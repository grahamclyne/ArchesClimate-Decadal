import logging
import math
import os
from collections.abc import Callable

import cartopy.crs as ccrs
import hydra
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import ma
from omegaconf import DictConfig, OmegaConf
from torchmetrics import Metric

from ArchesClimate.analysis.evaluate import (
    generate_monthly_dates,
    plot_headline_individual_metrics,
)
from ArchesClimate.paths import path as artifact_path

TAS_IDX = 10  # index of tas in surface_variables


from ArchesClimate.analysis.plot_climatology import load_diffesm_tas, load_mesmer_tas

LOGGER = logging.getLogger(__name__)

METRIC_NAMES = [
    "mse",
    "frmse",
    "var",
    "spskr",
    "crps",
    "fcrps",
    "energyscore",
    "fenergyscore",
]


def get_temporal_psd(data, sampling_rate=1.0):
    """Return the historical unnormalized Fourier amplitude (not power density).

    The legacy function name is retained for callers. Values have the input
    variable's units: the discrete FFT sum is neither squared nor normalized.
    """
    t, ny, nx = data.shape
    reshaped = data.reshape(t, ny * nx)
    fft_vals = np.fft.fft(reshaped, axis=0)
    psd = np.abs(fft_vals)
    freqs = np.fft.fftfreq(t, d=1 / sampling_rate)
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    psd = psd[pos_mask, :]
    mean_psd = np.nanmean(psd, axis=1)
    return freqs, mean_psd


class RankHistogramUltraFast(Metric):
    def __init__(
        self,
        n_members: int,
        data_shape: tuple = (118, 17),
        preprocess: Callable | None = None,
    ):
        Metric.__init__(self)
        self.preprocess = preprocess
        self.n_members = n_members
        self.data_shape = data_shape
        self.add_state("n_batches", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state(
            "histogram",
            default=torch.zeros(*data_shape, n_members + 1),
            dist_reduce_fx="sum",
        )

    def update(self, targets, preds) -> None:
        if isinstance(preds, list):
            preds = torch.stack(preds, dim=1)
        device = targets.device
        if self.preprocess:
            targets = self.preprocess(targets)
            preds = self.preprocess(preds)
        while targets.shape[0] == 1:
            targets = targets.squeeze(0)
        while preds.shape[0] == 1:
            preds = preds.squeeze(0)
        n_members = preds.shape[0]  # first dim after squeeze is members
        expected_dims = len(self.data_shape)
        bins = self.n_members + 1  # use constructor value to match histogram shape
        if expected_dims == 2:
            time_steps, n_vars = self.data_shape
            targets_flat = targets.reshape(time_steps, n_vars, -1)
            preds_flat = preds.reshape(n_members, time_steps, n_vars, -1)
        elif expected_dims == 3:
            time_steps, n_vars, n_levels = self.data_shape
            targets_flat = targets.reshape(time_steps, n_vars, n_levels, -1)
            preds_flat = preds.reshape(n_members, time_steps, n_vars, n_levels, -1)
        else:
            raise ValueError(f"Unsupported data_shape dimensions: {expected_dims}")
        targets_expanded = targets_flat.unsqueeze(0)
        ranks = (preds_flat < targets_expanded).sum(dim=0) + 1
        exact_matches = preds_flat == targets_expanded
        tie_counts = exact_matches.sum(dim=0)
        has_ties = tie_counts > 0
        if has_ties.any():
            random_offsets = torch.randint_like(tie_counts, 0, 1000000) % torch.clamp(
                tie_counts, min=1
            )
            ranks = torch.where(has_ties, ranks + random_offsets, ranks)
        histograms = torch.zeros(*self.data_shape, bins, device=device)
        rank_vals = ranks.long() - 1
        rank_vals = torch.clamp(rank_vals, 0, bins - 1)
        # exclude positions where target or any ensemble member is NaN
        valid = ~torch.isnan(targets_flat) & ~torch.isnan(preds_flat).any(dim=0)
        if expected_dims == 2:
            for t in range(time_steps):
                for v in range(n_vars):
                    mask = valid[t, v]
                    histograms[t, v] = torch.bincount(
                        rank_vals[t, v][mask], minlength=bins
                    ).float()
        elif expected_dims == 3:
            for t in range(time_steps):
                for v in range(n_vars):
                    for lev in range(n_levels):
                        mask = valid[t, v, lev]
                        histograms[t, v, lev] = torch.bincount(
                            rank_vals[t, v, lev][mask], minlength=bins
                        ).float()
        self.histogram += histograms

    def compute(self):
        total = self.histogram.sum(dim=-1, keepdim=True)
        normalized = self.histogram / (total + 1e-10)
        return {"rankhist": normalized * (self.n_members + 1)}


def _compute_psd_for_var(
    var_index,
    pred_dict_raw,
    target_surface_anomalies,
    target_level_anomalies,
    ocean_mask,
    var_filter,
):
    """Return {exp_name: (period, target_psd, pred_psd)} for a single var_index."""
    results = {}
    num_members = target_surface_anomalies.shape[0]
    for exp_idx, (exp_name, pred_anomalies) in enumerate(pred_dict_raw.items()):
        if (
            var_filter
            and exp_name in var_filter
            and var_index not in var_filter[exp_name]
        ):
            continue
        pred_surface_ano, pred_level_ano = pred_anomalies
        psd_list, psd1_list = [], []
        for member in range(num_members):
            if isinstance(var_index, tuple):
                freqs, psd = get_temporal_psd(
                    target_level_anomalies[member, :, var_index[0], var_index[1], :, :],
                    1,
                )
                _freqs1, psd1 = get_temporal_psd(
                    pred_level_ano[member, :, var_index[0], var_index[1], :, :], 1
                )
            elif isinstance(var_index, str) and var_index.startswith("land:"):
                vi = int(var_index.split(":")[1])
                tgt = target_surface_anomalies[member, :, vi, 0, :, :].copy()
                tgt[:, ocean_mask] = np.nan
                prd = pred_surface_ano[member, :, vi, 0, :, :].clone()
                prd[:, ocean_mask] = np.nan
                freqs, psd = get_temporal_psd(tgt, 1)
                _freqs1, psd1 = get_temporal_psd(prd, 1)
            else:
                freqs, psd = get_temporal_psd(
                    target_surface_anomalies[member, :, var_index, 0, :, :], 1
                )
                _freqs1, psd1 = get_temporal_psd(
                    pred_surface_ano[member, :, var_index, 0, :, :], 1
                )
            psd_list.append(psd)
            psd1_list.append(psd1)
        results[exp_name] = (
            1 / freqs,
            np.stack(psd_list).mean(0),
            np.stack(psd1_list).mean(0),
        )
    return results


def _draw_psd_axes(
    ax,
    psd_data,
    title,
    palette,
    exp_names,
    show_legend=False,
    yscale="linear",
    legend_handles=None,
):
    """Draw PSD lines onto ax from psd_data dict, auto-scaling y-axis.

    legend_handles: if provided, accumulate (handle, label) tuples into this list
                    so the caller can build a unified legend across subplots.
    """
    all_vals = []
    for exp_idx, exp_name in enumerate(exp_names):
        if exp_name not in psd_data:
            continue
        period, target_psd, pred_psd = psd_data[exp_name]
        color = palette[exp_idx % len(palette)]
        if exp_idx == 0:
            (h,) = ax.plot(
                period,
                target_psd,
                linewidth=1.5,
                color="black",
                label="IPSL-DCPP",
                linestyle=":",
            )
            all_vals.extend(target_psd.tolist())
            if legend_handles is not None and not any(
                l == "IPSL-DCPP" for _, l in legend_handles
            ):
                legend_handles.append((h, "IPSL-DCPP"))
        (h,) = ax.plot(period, pred_psd, linewidth=1.5, color=color, label=exp_name)
        all_vals.extend(pred_psd.tolist())
        if legend_handles is not None and not any(
            l == exp_name for _, l in legend_handles
        ):
            legend_handles.append((h, exp_name))
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale(yscale)
    if all_vals and yscale == "linear":
        vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
        margin = (vmax - vmin) * 0.05
        ax.set_ylim(vmin - margin, vmax + margin)
    if show_legend:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=2,
            fontsize=10,
            frameon=False,
        )


def plot_temporal_psd(
    pred_dict_raw,
    target_surface_anomalies,
    target_level_anomalies,
    surface_variables,
    level_variables,
    out_path,
    var_filter=None,
    precomputed_psd=None,
):
    """3x2 temporal PSD figure for headline variables.

    precomputed_psd: optional {var_index: psd_data} dict (psd_data in the same
        format _compute_psd_for_var returns) -- lets a caller pool PSD across
        multiple hindcast starts (see regen_psd_rankhist.py) instead of
        computing it fresh from a single (pred_dict_raw, target_*) pair here.
    """
    PLEV_LABELS = {0: "850hPa", 1: "700hPa", 2: "500hPa", 3: "250hPa"}
    headline_vars = [6, -5, -1, "land:10", (2, 1), (5, 1)]
    mpl.rcParams["mathtext.default"] = "regular"
    headline_var_names = []
    for i in headline_vars:
        if isinstance(i, tuple):
            headline_var_names.append(
                level_variables[i[0]] + " " + PLEV_LABELS.get(i[1], str(i[1]))
            )
        elif isinstance(i, str) and i.startswith("land:"):
            headline_var_names.append(
                surface_variables[int(i.split(":")[1])] + " (land)"
            )
        else:
            headline_var_names.append(surface_variables[i])

    ocean_mask = ~np.isnan(target_surface_anomalies[0, 0, -1, 0])
    _palette = [
        "#0173b2",
        "#de8f05",
        "#029e73",
        "#d55e00",
        "#cc78bc",
        "#ca9161",
        "#fbafe4",
        "#949494",
    ]
    exp_names = list(pred_dict_raw.keys())

    fig, ax = plt.subplots(3, 2, figsize=(10, 9))
    ax = ax.flatten()

    legend_handles = []
    for index, var_index in enumerate(headline_vars):
        if precomputed_psd is not None:
            psd_data = precomputed_psd[var_index]
        else:
            psd_data = _compute_psd_for_var(
                var_index,
                pred_dict_raw,
                target_surface_anomalies,
                target_level_anomalies,
                ocean_mask,
                var_filter,
            )
        _draw_psd_axes(
            ax[index],
            psd_data,
            headline_var_names[index],
            _palette,
            exp_names,
            show_legend=False,
            yscale="log",
            legend_handles=legend_handles,
        )
        # ax[index].yaxis.set_major_formatter(mpl.ticker.LogFormatterSciNotation(labelOnlyBase=True))
        ax[index].tick_params(axis="y", labelsize=10, which="both")

    if legend_handles:
        handles, labels = zip(*legend_handles)
        ax[0].legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(1.0, 1.05),
            ncol=2,
            fontsize=10,
            frameon=False,
        )

    plt.subplots_adjust(hspace=0.4, wspace=0.35)
    for axes in ax[:-2]:
        axes.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
    units = ["Pa", "K", "K", "K", "K", r"Pa s$^{-1}$"]
    for axis, unit in zip(ax, units):
        axis.set_ylabel(f"Fourier amplitude [{unit}]", fontsize=8)
    ax[-1].set_xlabel("Period [months]")
    ax[-2].set_xlabel("Period [months]")
    fig.savefig(out_path + ".pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_temporal_psd_tas(
    pred_dict_raw,
    target_surface_anomalies,
    target_level_anomalies,
    surface_variables,
    level_variables,
    out_path,
    var_filter=None,
    precomputed_psd=None,
):
    """Separate 1x2 temporal PSD figure for tas and tas (land only), log y-scale.

    precomputed_psd: see plot_temporal_psd.
    """
    tas_vars = [10, "land:10"]
    tas_names = [surface_variables[10], surface_variables[10] + " (land)"]

    ocean_mask = ~np.isnan(target_surface_anomalies[0, 0, -1, 0])
    _palette = [
        "#0173b2",
        "#de8f05",
        "#029e73",
        "#d55e00",
        "#cc78bc",
        "#ca9161",
        "#fbafe4",
        "#949494",
    ]
    exp_names = list(pred_dict_raw.keys())

    fig, ax = plt.subplots(1, 2, figsize=(7, 3))

    # Collect legend handles across both panels so all models appear even if
    # filtered out of one panel (e.g. mesmer/diffesm only in land panel).
    legend_handles = []
    for index, var_index in enumerate(tas_vars):
        if precomputed_psd is not None:
            psd_data = precomputed_psd[var_index]
        else:
            psd_data = _compute_psd_for_var(
                var_index,
                pred_dict_raw,
                target_surface_anomalies,
                target_level_anomalies,
                ocean_mask,
                var_filter,
            )
        _draw_psd_axes(
            ax[index],
            psd_data,
            tas_names[index],
            _palette,
            exp_names,
            show_legend=False,
            yscale="log",
            legend_handles=legend_handles,
        )
        ax[index].set_xlabel("Period [months]")
        ax[index].set_ylabel("Fourier amplitude [K]")

    if legend_handles:
        handles, labels = zip(*legend_handles)
        ax[0].legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(1.0, 1.05),
            ncol=len(legend_handles),
            fontsize=10,
            frameon=False,
        )

    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    fig.savefig(out_path + "_tas.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _compute_rank_histograms(
    pred_dict_raw_flat,
    target_surface,
    target_level,
    num_members,
    num_rollout_steps,
    start_lat,
    end_lat,
    start_lon,
    end_lon,
    n_surface_vars,
    n_level_vars,
    n_plevs=4,
):
    """Compute rank histograms: each predicted member ranked within the fixed target ensemble."""
    ocean_mask = ~torch.isnan(
        target_surface[0, 0, -1, 0, start_lat:end_lat, start_lon:end_lon]
    )
    tgt_surf = target_surface[
        :num_members, :num_rollout_steps, :, 0, start_lat:end_lat, start_lon:end_lon
    ]
    tgt_surf_land = tgt_surf.clone()
    tgt_surf_land[:, :, 10][..., ocean_mask] = float("nan")
    tgt_level = target_level[
        :num_members, :num_rollout_steps, :, :, start_lat:end_lat, start_lon:end_lon
    ]

    exp_surface_hists, exp_level_hists, exp_surface_land_hists = [], [], []
    for pred_surface, pred_level in pred_dict_raw_flat.values():
        surface_rh = RankHistogramUltraFast(
            n_members=num_members, data_shape=(num_rollout_steps, n_surface_vars)
        )
        surface_land_rh = RankHistogramUltraFast(
            n_members=num_members, data_shape=(num_rollout_steps, n_surface_vars)
        )
        level_rh = RankHistogramUltraFast(
            n_members=num_members, data_shape=(num_rollout_steps, n_level_vars, n_plevs)
        )
        for i in range(num_members):
            pred_surf_i = pred_surface[
                i, :, :, 0, start_lat:end_lat, start_lon:end_lon
            ][None, None]
            surface_rh.update(pred_surf_i, tgt_surf[None])
            pred_surf_land_i = pred_surf_i.clone()
            pred_surf_land_i[:, :, :, 10][..., ocean_mask] = float("nan")
            surface_land_rh.update(pred_surf_land_i, tgt_surf_land[None])
            level_rh.update(
                pred_level[i, :, :, :, start_lat:end_lat, start_lon:end_lon][
                    None, None
                ],
                tgt_level[None],
            )
        exp_surface_hists.append(surface_rh.compute()["rankhist"])
        exp_surface_land_hists.append(surface_land_rh.compute()["rankhist"])
        exp_level_hists.append(level_rh.compute()["rankhist"])
    return exp_surface_hists, exp_surface_land_hists, exp_level_hists


def _draw_rank_hist_axes(
    ax,
    index,
    var_index,
    exp_names,
    exp_surface_hists,
    exp_surface_land_hists,
    exp_level_hists,
    surface_variables,
    level_variables,
    palette,
    var_filter,
    legend_handles=None,
    itself_surface_hist=None,
    itself_surface_land_hist=None,
    itself_level_hist=None,
):
    """Draw rank histogram lines for one variable onto ax, auto-scaling y-axis."""
    all_vals = []
    allowed_map = var_filter or {}
    _plev_labels = {0: "850hPa", 1: "700hPa", 2: "500hPa", 3: "250hPa"}
    if isinstance(var_index, tuple):
        title = (
            level_variables[var_index[0]]
            + " "
            + _plev_labels.get(var_index[1], str(var_index[1]))
        )
    elif isinstance(var_index, str) and var_index.startswith("land:"):
        title = surface_variables[int(var_index.split(":")[1])] + " (land)"
    else:
        title = surface_variables[var_index]
    for exp_idx, exp_name in enumerate(exp_names):
        allowed = allowed_map.get(exp_name)
        if allowed is not None and var_index not in allowed:
            continue
        color = palette[exp_idx % len(palette)]
        if isinstance(var_index, tuple):
            data = exp_level_hists[exp_idx][:, var_index[0], var_index[1]]
        elif isinstance(var_index, str) and var_index.startswith("land:"):
            vi = int(var_index.split(":")[1])
            data = exp_surface_land_hists[exp_idx][:, vi]
        else:
            data = exp_surface_hists[exp_idx][:, var_index]
        y = data.mean(axis=0)
        (h,) = ax.plot(y, label=exp_name, color=color)
        all_vals.extend(y.tolist())
        if legend_handles is not None and not any(
            l == exp_name for _, l in legend_handles
        ):
            legend_handles.append((h, exp_name))

    # IPSL held-out comparison as dashed black line
    if itself_surface_hist is not None:
        if isinstance(var_index, tuple):
            data = itself_level_hist[:, var_index[0], var_index[1]]
        elif isinstance(var_index, str) and var_index.startswith("land:"):
            vi = int(var_index.split(":")[1])
            data = itself_surface_land_hist[:, vi]
        else:
            data = itself_surface_hist[:, var_index]
        y = data.mean(axis=0)
        (h,) = ax.plot(y, label="IPSL", color="black", linestyle="--", linewidth=1.2)
        all_vals.extend(y.tolist())
        if legend_handles is not None and not any(
            l == "IPSL" for _, l in legend_handles
        ):
            legend_handles.append((h, "IPSL"))

    ax.axhline(1, color="gray", linestyle=":", linewidth=0.8)
    if all_vals:
        vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
        margin = max((vmax - vmin) * 0.1, 0.05)
        ax.set_ylim(vmin - margin, vmax + margin)
    ax.set_title(title)


def _compute_itself_rank_histograms(
    target_surface,
    target_level,
    num_members,
    num_rollout_steps,
    start_lat,
    end_lat,
    start_lon,
    end_lon,
    n_surface_vars,
    n_level_vars,
    n_plevs=4,
):
    """Compute rank histograms for IPSL held-out: first 5 members as ensemble, remaining as truth."""
    ocean_mask = ~torch.isnan(
        target_surface[0, 0, -1, 0, start_lat:end_lat, start_lon:end_lon]
    )
    tgt_surf = target_surface[
        :, :num_rollout_steps, :, 0, start_lat:end_lat, start_lon:end_lon
    ]
    tgt_surf_land = tgt_surf.clone()
    tgt_surf_land[:, :, 10][..., ocean_mask] = float("nan")
    tgt_level = target_level[
        :, :num_rollout_steps, :, :, start_lat:end_lat, start_lon:end_lon
    ]

    n_total = target_surface.shape[0]
    pred_members = min(5, n_total // 2)
    n_truth_members = n_total - pred_members

    # first pred_members = "ensemble"; remaining members = truths iterated over
    group_a_surf = tgt_surf[:pred_members]
    group_a_surf_land = tgt_surf_land[:pred_members]
    group_a_level = tgt_level[:pred_members]

    surface_rh = RankHistogramUltraFast(
        n_members=pred_members, data_shape=(num_rollout_steps, n_surface_vars)
    )
    surface_land_rh = RankHistogramUltraFast(
        n_members=pred_members, data_shape=(num_rollout_steps, n_surface_vars)
    )
    level_rh = RankHistogramUltraFast(
        n_members=pred_members, data_shape=(num_rollout_steps, n_level_vars, n_plevs)
    )

    for i in range(n_truth_members):
        j = pred_members + i
        truth_surf_i = tgt_surf[j][None, None]
        truth_surf_land_i = tgt_surf_land[j][None, None]
        truth_level_i = tgt_level[j][None, None]
        surface_rh.update(truth_surf_i, group_a_surf[None])
        surface_land_rh.update(truth_surf_land_i, group_a_surf_land[None])
        level_rh.update(truth_level_i, group_a_level[None])

    return (
        surface_rh.compute()["rankhist"],
        surface_land_rh.compute()["rankhist"],
        level_rh.compute()["rankhist"],
    )


def plot_rank_histogram(
    pred_dict_raw_flat,
    target_surface,
    target_level,
    exp_names,
    surface_variables,
    level_variables,
    num_members,
    num_rollout_steps,
    out_path,
    start_lat=0,
    end_lat=144,
    start_lon=0,
    end_lon=144,
    var_filter=None,
):
    """3x2 rank histogram figure for headline variables."""
    headline_indices = [6, -5, -1, "land:10", (2, 1), (5, 1)]

    n_surface_vars = len(surface_variables)
    n_level_vars = len(level_variables)

    exp_surface_hists, exp_surface_land_hists, exp_level_hists = (
        _compute_rank_histograms(
            pred_dict_raw_flat,
            target_surface,
            target_level,
            num_members,
            num_rollout_steps,
            start_lat,
            end_lat,
            start_lon,
            end_lon,
            n_surface_vars,
            n_level_vars,
        )
    )
    itself_surf_hist, itself_surf_land_hist, itself_level_hist = (
        _compute_itself_rank_histograms(
            target_surface,
            target_level,
            num_members,
            num_rollout_steps,
            start_lat,
            end_lat,
            start_lon,
            end_lon,
            n_surface_vars,
            n_level_vars,
        )
    )

    _palette = [
        "#0173b2",
        "#de8f05",
        "#029e73",
        "#d55e00",
        "#cc78bc",
        "#ca9161",
        "#fbafe4",
        "#949494",
    ]

    fig, ax = plt.subplots(3, 2, figsize=(10, 9))
    ax = ax.flatten()

    legend_handles = []
    for index, var_index in enumerate(headline_indices):
        _draw_rank_hist_axes(
            ax[index],
            index,
            var_index,
            exp_names,
            exp_surface_hists,
            exp_surface_land_hists,
            exp_level_hists,
            surface_variables,
            level_variables,
            _palette,
            var_filter,
            legend_handles=legend_handles,
            itself_surface_hist=itself_surf_hist,
            itself_surface_land_hist=itself_surf_land_hist,
            itself_level_hist=itself_level_hist,
        )
        if index > 3:
            ax[index].set_xlabel("Rank")

    fig.text(
        0.04, 0.5, "Normalized Frequency", va="center", rotation="vertical", fontsize=12
    )
    if legend_handles:
        handles, labels = zip(*legend_handles)
        ax[0].legend(
            handles,
            labels,
            bbox_to_anchor=(1, 1.15),
            loc="lower center",
            ncol=2,
            frameon=False,
        )
    plt.subplots_adjust(
        hspace=0.4, wspace=0.35, left=0.1, right=0.95, top=0.92, bottom=0.08
    )
    fig.savefig(out_path + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rank_histogram_tas(
    pred_dict_raw_flat,
    target_surface,
    target_level,
    exp_names,
    surface_variables,
    level_variables,
    num_members,
    num_rollout_steps,
    out_path,
    start_lat=0,
    end_lat=144,
    start_lon=0,
    end_lon=144,
    var_filter=None,
):
    """Separate 1x2 rank histogram figure for tas and tas (land only)."""
    n_surface_vars = len(surface_variables)
    n_level_vars = len(level_variables)

    exp_surface_hists, exp_surface_land_hists, exp_level_hists = (
        _compute_rank_histograms(
            pred_dict_raw_flat,
            target_surface,
            target_level,
            num_members,
            num_rollout_steps,
            start_lat,
            end_lat,
            start_lon,
            end_lon,
            n_surface_vars,
            n_level_vars,
        )
    )
    itself_surf_hist, itself_surf_land_hist, itself_level_hist = (
        _compute_itself_rank_histograms(
            target_surface,
            target_level,
            num_members,
            num_rollout_steps,
            start_lat,
            end_lat,
            start_lon,
            end_lon,
            n_surface_vars,
            n_level_vars,
        )
    )

    _palette = [
        "#0173b2",
        "#de8f05",
        "#029e73",
        "#d55e00",
        "#cc78bc",
        "#ca9161",
        "#fbafe4",
        "#949494",
    ]

    fig, ax = plt.subplots(1, 2, figsize=(7, 3))

    legend_handles = []
    for index, var_index in enumerate([10, "land:10"]):
        _draw_rank_hist_axes(
            ax[index],
            index,
            var_index,
            exp_names,
            exp_surface_hists,
            exp_surface_land_hists,
            exp_level_hists,
            surface_variables,
            level_variables,
            _palette,
            var_filter,
            legend_handles=legend_handles,
            itself_surface_hist=itself_surf_hist,
            itself_surface_land_hist=itself_surf_land_hist,
            itself_level_hist=itself_level_hist,
        )
        ax[index].set_xlabel("Rank")
        ax[index].set_ylabel("Normalized Frequency")

    if legend_handles:
        handles, labels = zip(*legend_handles)
        ax[0].legend(
            handles,
            labels,
            bbox_to_anchor=(0.5, 1.15),
            loc="lower center",
            ncol=2,
            frameon=False,
        )
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    fig.savefig(out_path + "_tas.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_latex_tables(
    pred_dict,
    itself_metrics,
    var_names,
    out_path,
    last_n=None,
    pred_dict_land=None,
    itself_metrics_land=None,
    pred_dict_yearly=None,
    itself_metrics_yearly=None,
    pred_dict_land_yearly=None,
    itself_metrics_land_yearly=None,
    last_n_yearly=2,
):
    """
    Write a .tex file with one LaTeX table per metric.
    Rows = fixed variable order, columns = experiments + itself (IPSL).
    Best value per row is bolded (lowest, except for var where highest is bolded).
    """
    # Fixed variable order and pressure level for level vars (700hPa = index 2)
    SURFACE_VAR_ORDER = ["tas", "thetaot2000", "tos", "psl", "net_flux", "pr"]
    LEVEL_VAR_ORDER = [("hur", 1), ("ta", 1), ("wap", 1), ("zg", 1)]
    PLEV_LABELS = {0: "850hPa", 1: "700hPa", 2: "500hPa", 3: "250hPa"}

    num_surface_vars = itself_metrics["surface"].shape[2]
    num_level_vars = itself_metrics["level"].shape[2]
    surface_var_names = var_names[:num_surface_vars]
    level_var_names = var_names[num_surface_vars : num_surface_vars + num_level_vars]

    all_exps = dict(pred_dict)
    all_exps["IPSL"] = itself_metrics

    all_exps_land = None
    if pred_dict_land is not None and itself_metrics_land is not None:
        all_exps_land = dict(pred_dict_land)
        all_exps_land["IPSL"] = itself_metrics_land

    def _avg(metrics, kind, vi, pi, last_n_val):
        if kind == "surface":
            return (
                metrics["surface"][-last_n_val:].nanmean(dim=0)[:, vi]
                if last_n_val
                else metrics["surface"].nanmean(dim=0)[:, vi]
            )
        else:
            return (
                metrics["level"][-last_n_val:].nanmean(dim=0)[:, vi, pi]
                if last_n_val
                else metrics["level"].nanmean(dim=0)[:, vi, pi]
            )

    def _fmt(v):
        """Format to 2 significant decimal places, scientific for very small values."""
        if abs(v) < 0.01 and v != 0:
            # scientific notation with 2 decimal places: e.g. 3.10e-06
            s = f"{v:.2e}"
            # convert e+XX / e-XX to e+XX keeping compact form
            mantissa, exp = s.split("e")
            exp_int = int(exp)
            return f"{mantissa}e{exp_int:+03d}"
        else:
            return f"{v:.2f}" if abs(v) >= 10 else f"{v:.4g}"

    all_exps_yearly = None
    all_exps_land_yearly = None
    if pred_dict_yearly is not None and itself_metrics_yearly is not None:
        all_exps_yearly = dict(pred_dict_yearly)
        all_exps_yearly["IPSL"] = itself_metrics_yearly
    if pred_dict_land_yearly is not None and itself_metrics_land_yearly is not None:
        all_exps_land_yearly = dict(pred_dict_land_yearly)
        all_exps_land_yearly["IPSL"] = itself_metrics_land_yearly

    # Build row index: (label, kind, vi, pi, use_land, use_yearly)
    rows = []
    for vname in SURFACE_VAR_ORDER:
        if vname in surface_var_names:
            vi = list(surface_var_names).index(vname)
            label = r"\textit{" + vname.replace("_", r"\_") + "}"
            rows.append((label, "surface", vi, None, False, False))
            if vname == "tas" and all_exps_land is not None:
                rows.append((r"\textit{tas} (land)", "surface", vi, None, True, False))
            if vname == "tas" and all_exps_yearly is not None:
                rows.append(
                    (r"\textit{tas} (yearly)", "surface", vi, None, False, True)
                )
            if (
                vname == "tas"
                and all_exps_land_yearly is not None
                and pred_dict_land is not None
            ):
                rows.append(
                    (r"\textit{tas} land (yearly)", "surface", vi, None, True, True)
                )
    for vname, plev_idx in LEVEL_VAR_ORDER:
        if vname in level_var_names:
            vi = list(level_var_names).index(vname)
            plev_label = PLEV_LABELS.get(plev_idx, str(plev_idx))
            label = r"\textit{" + vname.replace("_", r"\_") + "} " + plev_label
            rows.append((label, "level", vi, plev_idx, False, False))

    exp_names = list(all_exps.keys())
    n_cols = len(exp_names)
    col_spec = "l|" + "c" * n_cols

    lines = []
    lines.append(r"% Auto-generated LaTeX metric tables")
    lines.append("")

    for mi, mname in enumerate(METRIC_NAMES):
        higher_is_better = mname == "var"

        lines.append(r"\begin{table}[h!]")
        lines.append(r"\centering")
        lines.append(
            r"\caption{"
            + mname.upper()
            + r" --- last "
            + (f"last {last_n} months" if last_n else "full period")
            + r" average}"
        )
        lines.append(r"\begin{tabular}{" + col_spec + r"}")
        lines.append(r"\toprule")

        header = (
            r"\textbf{Variable} & "
            + " & ".join(
                r"\textbf{" + str(n).replace("_", r"\_") + "}" for n in exp_names
            )
            + r" \\"
        )
        lines.append(header)
        lines.append(r"\midrule")

        ipsl_idx = exp_names.index("IPSL") if "IPSL" in exp_names else None

        for label, kind, vi, pi, use_land, use_yearly in rows:
            if use_yearly:
                if use_land:
                    exps = all_exps_land_yearly
                else:
                    exps = all_exps_yearly
                n_val = last_n_yearly
            else:
                exps = all_exps_land if use_land else all_exps
                n_val = last_n
            if exps is None:
                continue
            raw_vals = []
            for exp_name in exp_names:
                if exp_name not in exps:
                    raw_vals.append(float("nan"))
                    continue
                avgs = _avg(exps[exp_name], kind, vi, pi, n_val)
                raw_vals.append(avgs[mi].item())

            finite = [v for v in raw_vals if not math.isnan(v)]  # filter nan
            if not finite:
                continue

            if (
                mname == "spskr"
                and ipsl_idx is not None
                and not math.isnan(raw_vals[ipsl_idx])
            ):
                ipsl_val = 1.0
                non_ipsl_vals = [
                    (abs(v - ipsl_val), i)
                    for i, v in enumerate(raw_vals)
                    if i != ipsl_idx and not math.isnan(v)
                ]
                best_idx = min(non_ipsl_vals)[1] if non_ipsl_vals else None
                best = raw_vals[best_idx] if best_idx is not None else None
            else:
                best = max(finite) if higher_is_better else min(finite)

            cells = []
            for v in raw_vals:
                if math.isnan(v):  # nan
                    cells.append("--")
                else:
                    s = _fmt(v)
                    cells.append(r"\textbf{" + s + "}" if v == best else s)

            lines.append(label + " & " + " & ".join(cells) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

    # --- Extra table: RMSE (sqrt of MSE) ---
    MSE_IDX = METRIC_NAMES.index("mse")
    period_label_rmse = f"last {last_n} months" if last_n else "full period"
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\caption{RMSE --- " + period_label_rmse + r" average}")
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Variable} & "
        + " & ".join(r"\textbf{" + str(n).replace("_", r"\_") + "}" for n in exp_names)
        + r" \\"
    )
    lines.append(r"\midrule")

    for label, kind, vi, pi, use_land, use_yearly in rows:
        if use_yearly:
            exps = all_exps_land_yearly if use_land else all_exps_yearly
            n_val = last_n_yearly
        else:
            exps = all_exps_land if use_land else all_exps
            n_val = last_n
        if exps is None:
            continue
        raw_vals = []
        for exp_name in exp_names:
            if exp_name not in exps:
                raw_vals.append(float("nan"))
                continue
            avgs = _avg(exps[exp_name], kind, vi, pi, n_val)
            mse_val = avgs[MSE_IDX].item()
            raw_vals.append(
                mse_val**0.5
                if not math.isnan(mse_val) and mse_val >= 0
                else float("nan")
            )
        finite = [v for v in raw_vals if not math.isnan(v)]
        if not finite:
            continue
        best = min(finite)
        cells = []
        for v in raw_vals:
            if math.isnan(v):
                cells.append("--")
            else:
                s = _fmt(v)
                cells.append(r"\textbf{" + s + "}" if v == best else s)
        lines.append(label + " & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # --- Extra table: std dev ratio (pred / IPSL) ---
    VAR_IDX = METRIC_NAMES.index("var")
    ipsl_exp = all_exps.get("IPSL")
    non_ipsl_names = [n for n in exp_names if n != "IPSL"]
    if ipsl_exp is not None and non_ipsl_names:
        period_label = f"last {last_n} months" if last_n else "full period"
        ratio_lines = []
        ratio_lines.append(r"\begin{table}[h!]")
        ratio_lines.append(r"\centering")
        ratio_lines.append(
            r"\caption{Ensemble spread ratio (pred std / IPSL std) --- "
            + period_label
            + r"}"
        )
        col_spec_ratio = "l|" + "c" * len(non_ipsl_names)
        ratio_lines.append(r"\begin{tabular}{" + col_spec_ratio + r"}")
        ratio_lines.append(r"\toprule")
        header = (
            r"\textbf{Variable} & "
            + " & ".join(
                r"\textbf{" + str(n).replace("_", r"\_") + "}" for n in non_ipsl_names
            )
            + r" \\"
        )
        ratio_lines.append(header)
        ratio_lines.append(r"\midrule")

        n_rows_written = 0
        for label, kind, vi, pi, use_land, use_yearly in rows:
            if use_yearly:
                exps_r = all_exps_land_yearly if use_land else all_exps_yearly
                ipsl_r = (
                    (all_exps_land_yearly or {}).get("IPSL")
                    if use_land
                    else (all_exps_yearly or {}).get("IPSL")
                )
                n_val = last_n_yearly
            else:
                exps_r = all_exps_land if use_land else all_exps
                ipsl_r = (all_exps_land or {}).get("IPSL") if use_land else ipsl_exp
                n_val = last_n
            if exps_r is None or ipsl_r is None:
                continue

            try:
                ipsl_var = _avg(ipsl_r, kind, vi, pi, n_val)[VAR_IDX].item()
            except Exception as e:
                LOGGER.warning("Optional analysis input or plot failed", exc_info=True)
                print(f"[std ratio] _avg failed for IPSL {label}: {e}")
                continue
            print(f"[std ratio] {label} kind={kind} vi={vi} ipsl_var={ipsl_var:.6g}")
            if math.isnan(ipsl_var) or ipsl_var <= 0:
                continue
            ipsl_std = ipsl_var**0.5

            raw_vals = []
            for exp_name in non_ipsl_names:
                if exp_name not in exps_r:
                    raw_vals.append(float("nan"))
                    continue
                pred_var = _avg(exps_r[exp_name], kind, vi, pi, n_val)[VAR_IDX].item()
                raw_vals.append(
                    (pred_var**0.5) / ipsl_std
                    if not math.isnan(pred_var) and pred_var >= 0
                    else float("nan")
                )

            finite = [v for v in raw_vals if not math.isnan(v)]
            if not finite:
                continue
            best = min(finite, key=lambda v: abs(v - 1.0))

            cells = []
            for v in raw_vals:
                if math.isnan(v):
                    cells.append("--")
                else:
                    s = _fmt(v)
                    cells.append(r"\textbf{" + s + "}" if v == best else s)
            ratio_lines.append(label + " & " + " & ".join(cells) + r" \\")
            n_rows_written += 1

        ratio_lines.append(r"\bottomrule")
        ratio_lines.append(r"\end{tabular}")
        ratio_lines.append(r"\end{table}")
        ratio_lines.append("")
        print(f"[std ratio table] wrote {n_rows_written} rows")
        lines.extend(ratio_lines)

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_robinson_spatial_rmse(
    spatial_rmse_dict, itself_spatial_rmse, out_path, var_names, var_indices=None
):
    """
    Robinson projection maps of spatial RMSE (last-24-month average) per model.

    spatial_rmse_dict: {exp_name: tensor [num_vars, 1, nlat, nlon]}
    itself_spatial_rmse: tensor [num_vars, 1, nlat, nlon]
    var_indices: list of surface variable indices to plot (default: tas=10, tos=-1)
    """
    if var_indices is None:
        var_indices = [10, -1]  # tas, tos

    lat = np.linspace(-90, 90, 143)
    lon = np.linspace(-2, 358, 144)
    lon2d, lat2d = np.meshgrid(lon, lat)

    all_entries = list(spatial_rmse_dict.items()) + [("ipsl", itself_spatial_rmse)]
    n_models = len(all_entries)

    for vi in var_indices:
        vname = var_names[vi]
        fig, axes = plt.subplots(
            1,
            n_models,
            figsize=(5 * n_models, 3.5),
            subplot_kw={"projection": ccrs.Robinson(central_longitude=210)},
            constrained_layout=True,
        )
        if n_models == 1:
            axes = [axes]

        # Determine shared color range across all models
        all_vals = []
        for name, rmse in all_entries:
            v = rmse[vi, 0].numpy() if hasattr(rmse, "numpy") else np.array(rmse[vi, 0])
            all_vals.append(v)
        vmax = float(
            np.nanpercentile(np.concatenate([v.ravel() for v in all_vals]), 95)
        )
        levels = np.linspace(0, vmax, 11)

        cf = None
        for ax, (name, rmse), data in zip(axes, all_entries, all_vals):
            masked = ma.masked_invalid(data)
            cf = ax.contourf(
                lon2d,
                lat2d,
                masked,
                levels=levels,
                transform=ccrs.PlateCarree(),
                cmap="cividis",
                extend="max",
            )
            ax.coastlines(linewidth=0.5)
            ax.set_title(name, fontsize=10)

        plt.colorbar(
            cf, ax=axes[-1], orientation="vertical", shrink=0.8, label="RMSE", pad=0.02
        )
        fig.suptitle(f"Spatial RMSE — {vname} (last 24 months avg)", fontsize=12)
        fig.savefig(f"{out_path}_{vname}.pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)


def plot_single_var_metric(
    pred_dict, itself_metrics, out_path, var_idx, title, metric_index
):
    """Plot one surface variable's metric over time for all experiments."""
    colors = [
        "#0173b2",
        "#de8f05",
        "#029e73",
        "#d55e00",
        "#cc78bc",
        "#ca9161",
        "#fbafe4",
        "#949494",
        "#ece133",
        "#56b4e9",
    ]
    T = next(iter(pred_dict.values()))["surface"].shape[0]
    x = generate_monthly_dates(1969, 3, T)
    fig, ax = plt.subplots(1, figsize=(7, 3))
    for i, (exp_name, metrics) in enumerate(pred_dict.items()):
        ax.plot(
            x,
            metrics["surface"][:, metric_index, var_idx],
            color=colors[i % len(colors)],
            label=exp_name,
        )
    ax.plot(
        x,
        itself_metrics["surface"][:, metric_index, var_idx],
        color="black",
        linestyle="--",
        label="ipsl",
    )
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.legend(loc="upper left", ncol=1, fontsize=9, frameon=False)
    plt.subplots_adjust(hspace=0.4)
    fig.set_size_inches((7, 3))
    fig.savefig(out_path + ".pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def save_last_two_years_averages(
    pred_dict,
    itself_metrics,
    var_names,
    out_path,
    last_n=None,
    pred_dict_land=None,
    itself_metrics_land=None,
):
    TAS_IDX = 10

    num_surface_vars = itself_metrics["surface"].shape[2]
    num_level_vars = itself_metrics["level"].shape[2]
    num_plevs = itself_metrics["level"].shape[3]
    surface_var_names = var_names[:num_surface_vars]
    level_var_names = var_names[num_surface_vars : num_surface_vars + num_level_vars]

    lines = []
    period_label = f"last {last_n} months" if last_n else "full period"
    lines.append(f"{period_label} average metrics")
    lines.append("=" * 70)

    all_exps = dict(pred_dict)
    all_exps["itself"] = itself_metrics

    for exp_name, metrics in all_exps.items():
        lines.append(f"\nExperiment: {exp_name}")
        lines.append("-" * 50)

        surf = metrics["surface"][-last_n:] if last_n else metrics["surface"]
        lvl = metrics["level"][-last_n:] if last_n else metrics["level"]

        surf_avg = surf.nanmean(dim=0)
        lvl_avg = lvl.nanmean(dim=0)

        lines.append("\n  Surface variables:")
        for vi, vname in enumerate(surface_var_names):
            lines.append(f"    {vname}:")
            for mi, mname in enumerate(METRIC_NAMES):
                val = surf_avg[mi, vi].item()
                lines.append(f"      {mname:>14s}: {val:.6f}")

        lines.append("\n  Level variables:")
        for vi, vname in enumerate(level_var_names):
            for pi in range(num_plevs):
                lines.append(f"    {vname} [plev {pi}]:")
                for mi, mname in enumerate(METRIC_NAMES):
                    val = lvl_avg[mi, vi, pi].item()
                    lines.append(f"      {mname:>14s}: {val:.6f}")

    # --- TAS summary (global) ---
    lines.append("\n\n" + "=" * 70)
    lines.append(f"Surface Temp TAS (global, index {TAS_IDX}) — {period_label} avg")
    lines.append("=" * 70)
    for exp_name, metrics in all_exps.items():
        surf = metrics["surface"][-last_n:] if last_n else metrics["surface"]
        surf_avg = surf.nanmean(dim=0)
        lines.append(f"\n  {exp_name}:")
        for mi, mname in enumerate(METRIC_NAMES):
            val = surf_avg[mi, TAS_IDX].item()
            lines.append(f"    {mname:>14s}: {val:.6f}")

    # --- TAS summary (land only) ---
    if pred_dict_land is not None and itself_metrics_land is not None:
        all_exps_land = dict(pred_dict_land)
        all_exps_land["itself"] = itself_metrics_land
        lines.append("\n\n" + "=" * 70)
        lines.append(
            f"Surface Temp TAS (land only, index {TAS_IDX}) — {period_label} avg"
        )
        lines.append("=" * 70)
        for exp_name, metrics in all_exps_land.items():
            surf = metrics["surface"][-last_n:] if last_n else metrics["surface"]
            surf_avg = surf.nanmean(dim=0)
            lines.append(f"\n  {exp_name}:")
            for mi, mname in enumerate(METRIC_NAMES):
                val = surf_avg[mi, TAS_IDX].item()
                lines.append(f"    {mname:>14s}: {val:.6f}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_yearly_metrics(
    pred_dict,
    itself_metrics,
    out_path,
    var_names,
    metric_index,
    legend_title,
    pred_dict_land=None,
    itself_metrics_land=None,
):
    """Plot yearly-averaged headline metrics, one subplot per variable."""
    _palette = [
        "#0173b2",
        "#de8f05",
        "#029e73",
        "#d55e00",
        "#cc78bc",
        "#ca9161",
        "#fbafe4",
        "#949494",
        "#ece133",
        "#56b4e9",
    ]
    exp_colors = {exp: _palette[i % len(_palette)] for i, exp in enumerate(pred_dict)}
    TAS_IDX = 10
    headline_surface_indices = [1, 6, -5]
    headline_surface_names = ["net_flux", "psl", "tos"]
    headline_surface_ylabels = ["W m-2", "Pa", "degC"]

    has_tas = pred_dict_land is not None and itself_metrics_land is not None
    n_rows = 6 if has_tas else 4

    T = next(iter(pred_dict.values()))["surface"].shape[0]
    x = list(range(1, T + 1))

    fig, axes = plt.subplots(n_rows, sharex=True)
    axes = axes.flatten()

    for col, (vi, title, ylabel) in enumerate(
        zip(headline_surface_indices, headline_surface_names, headline_surface_ylabels)
    ):
        for exp, metrics in pred_dict.items():
            axes[col].plot(
                x,
                metrics["surface"][:, metric_index, vi],
                color=exp_colors[exp],
                label=exp,
            )
        axes[col].plot(
            x,
            itself_metrics["surface"][:, metric_index, vi],
            color="black",
            linestyle="--",
            label="ipsl",
        )
        axes[col].set_title(title)
        axes[col].set_ylabel(ylabel)
        axes[col].tick_params(axis="both", which="major")

    for exp, metrics in pred_dict.items():
        axes[3].plot(
            x, metrics["level"][:, metric_index, 2, 1], color=exp_colors[exp], label=exp
        )
    axes[3].plot(
        x,
        itself_metrics["level"][:, metric_index, 2, 1],
        color="black",
        linestyle="--",
        label="ipsl",
    )
    axes[3].set_title("ta")
    axes[3].set_ylabel("K")
    axes[3].tick_params(axis="both", which="major")

    if has_tas:
        for exp, metrics in pred_dict.items():
            axes[4].plot(
                x,
                metrics["surface"][:, metric_index, TAS_IDX],
                color=exp_colors[exp],
                label=exp,
            )
        axes[4].plot(
            x,
            itself_metrics["surface"][:, metric_index, TAS_IDX],
            color="black",
            linestyle="--",
            label="ipsl",
        )
        axes[4].set_title("tas (global)")
        axes[4].set_ylabel("K")
        axes[4].tick_params(axis="both", which="major")

        land_colors = {
            exp: _palette[i % len(_palette)] for i, exp in enumerate(pred_dict_land)
        }
        for exp, metrics in pred_dict_land.items():
            axes[5].plot(
                x,
                metrics["surface"][:, metric_index, TAS_IDX],
                color=land_colors[exp],
                label=exp,
            )
        axes[5].plot(
            x,
            itself_metrics_land["surface"][:, metric_index, TAS_IDX],
            color="black",
            linestyle="--",
            label="ipsl",
        )
        axes[5].set_title("tas (land)")
        axes[5].set_ylabel("K")
        axes[5].tick_params(axis="both", which="major")

    if metric_index == 4:
        axes[0].legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=2,
            fontsize=10,
            frameon=False,
        )
    axes[-1].set_xlabel("Prediction year")
    for ax in axes[:-1]:
        ax.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )

    plt.subplots_adjust(hspace=0.4)
    fig.set_size_inches((7, 1.5 * n_rows))
    fig.savefig(out_path + ".pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def save_last_two_years_text(
    pred_dict_yearly,
    itself_metrics_yearly,
    var_names,
    out_path,
    pred_dict_land_yearly=None,
    itself_metrics_land_yearly=None,
):
    """Write metric values averaged over the last 2 years of the yearly stats."""
    TAS_IDX = 10
    last_n = 2

    num_surface_vars = itself_metrics_yearly["surface"].shape[2]
    num_level_vars = itself_metrics_yearly["level"].shape[2]
    num_plevs = itself_metrics_yearly["level"].shape[3]
    surface_var_names = var_names[:num_surface_vars]
    level_var_names = var_names[num_surface_vars : num_surface_vars + num_level_vars]

    lines = []
    lines.append(f"Last {last_n} years average (from yearly metrics)")
    lines.append("=" * 70)

    all_exps = dict(pred_dict_yearly)
    all_exps["itself"] = itself_metrics_yearly

    for exp_name, metrics in all_exps.items():
        lines.append(f"\nExperiment: {exp_name}")
        lines.append("-" * 50)
        surf_avg = metrics["surface"][-last_n:].nanmean(dim=0)
        lvl_avg = metrics["level"][-last_n:].nanmean(dim=0)

        lines.append("\n  Surface variables:")
        for vi, vname in enumerate(surface_var_names):
            lines.append(f"    {vname}:")
            for mi, mname in enumerate(METRIC_NAMES):
                lines.append(f"      {mname:>14s}: {surf_avg[mi, vi].item():.6f}")

        lines.append("\n  Level variables:")
        for vi, vname in enumerate(level_var_names):
            for pi in range(num_plevs):
                lines.append(f"    {vname} [plev {pi}]:")
                for mi, mname in enumerate(METRIC_NAMES):
                    lines.append(
                        f"      {mname:>14s}: {lvl_avg[mi, vi, pi].item():.6f}"
                    )

    lines.append("\n\n" + "=" * 70)
    lines.append(
        f"Surface Temp TAS (global, index {TAS_IDX}) — last {last_n} years avg"
    )
    lines.append("=" * 70)
    for exp_name, metrics in all_exps.items():
        surf_avg = metrics["surface"][-last_n:].nanmean(dim=0)
        lines.append(f"\n  {exp_name}:")
        for mi, mname in enumerate(METRIC_NAMES):
            lines.append(f"    {mname:>14s}: {surf_avg[mi, TAS_IDX].item():.6f}")

    if pred_dict_land_yearly is not None and itself_metrics_land_yearly is not None:
        all_exps_land = dict(pred_dict_land_yearly)
        all_exps_land["itself"] = itself_metrics_land_yearly
        lines.append("\n\n" + "=" * 70)
        lines.append(
            f"Surface Temp TAS (land only, index {TAS_IDX}) — last {last_n} years avg"
        )
        lines.append("=" * 70)
        for exp_name, metrics in all_exps_land.items():
            surf_avg = metrics["surface"][-last_n:].nanmean(dim=0)
            lines.append(f"\n  {exp_name}:")
            for mi, mname in enumerate(METRIC_NAMES):
                lines.append(f"    {mname:>14s}: {surf_avg[mi, TAS_IDX].item():.6f}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):

    plt.style.use("seaborn-v0_8-paper")
    SMALL_SIZE = 12
    MEDIUM_SIZE = 13
    BIGGER_SIZE = 14

    # Set global font to Times New Roman
    # mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams["font.serif"] = ["Times New Roman"]

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc("lines", linewidth=1.1)

    plt.rcParams["font.family"] = ["sans"]

    print(cfg)
    # target = cfg.analysis.experiments_to_plot[0].split('_')[-2]
    # year = cfg.analysis.experiments_to_plot[0].split('_')[-1]
    num_members = cfg.analysis.num_members
    num_rollout_steps = cfg.analysis.num_rollout_steps
    var_names = (
        cfg.dataloader.dataset.surface_variables
        + cfg.dataloader.dataset.level_variables
    )
    experiment = cfg.analysis.experiments_to_plot[0]
    start_lat = cfg.analysis.start_lat
    end_lat = cfg.analysis.end_lat
    start_lon = cfg.analysis.start_lon
    end_lon = cfg.analysis.end_lon
    # climatology_metrics = torch.load(f'{cfg.paths.metrics}/climatology_metrics/climatology_metrics_{target}_{year}_{start_lat}_{end_lat}_{start_lon}_{end_lon}')
    # climatology_metrics['surface'] = climatology_metrics['surface'][:,:,cfg.module.surface_variable_indices]
    # only needed for denorm function, sometimes different from predicted (e.g. non spatial std)
    out_dir = experiment + f"_{num_members}_{num_rollout_steps}"
    os.makedirs(f"{cfg.paths.plots}/{out_dir}", exist_ok=True)

    itself_metrics = torch.load(
        f"{cfg.paths.metrics}/{cfg.analysis.experiment}_{num_members}_{num_rollout_steps}/itself_output_metrics_{start_lat}_{end_lat}_{start_lon}_{end_lon}"
    )
    # print('normalized',normalized_itself_metrics)
    # normalized_itself_metrics['surface'] = normalized_itself_metrics['surface'][:,:,cfg.module.surface_variable_indices]

    diffesm_year = OmegaConf.select(cfg, "analysis.diffesm_year", default=None)
    mesmer_year = OmegaConf.select(cfg, "analysis.mesmer_year", default=None)

    BASELINES = {"mesmer", "diffesm"}

    def resolve_experiment_dir(x):
        """Metrics cache directory: always has _{num_members}_{num_rollout_steps} suffix."""
        if x == "diffesm" and diffesm_year is not None:
            return f"diffesm_{diffesm_year}_{num_members}_{num_rollout_steps}"
        if x == "mesmer" and mesmer_year is not None:
            return f"mesmer_{mesmer_year}_{num_members}_{num_rollout_steps}"
        return f"{x}_{num_members}_{num_rollout_steps}"

    def resolve_raw_experiment_dir(x):
        """Raw rollout directory: no metrics suffix, just the experiment name with year."""
        if x == "diffesm" and diffesm_year is not None:
            return f"diffesm_{diffesm_year}"
        if x == "mesmer" and mesmer_year is not None:
            return f"mesmer_{mesmer_year}"
        return x

    experiment_mode = "default"
    if experiment_mode == "default":
        experiments = [
            torch.load(
                f"{cfg.paths.metrics}/{resolve_experiment_dir(x)}/predicted_output_metrics_{start_lat}_{end_lat}_{start_lon}_{end_lon}"
            )
            for x in cfg.analysis.experiments_to_plot
        ]
    elif experiment_mode == "ensemble_size":
        experiments = [
            torch.load(
                f"{cfg.paths.metrics}/{experiment}_{x}_{num_rollout_steps}/predicted_output_metrics_{start_lat}_{end_lat}"
            )
            for x in [2, 5, 10, 20, 40]
        ]
    elif experiment_mode == "latitude_comp":
        experiments = [
            torch.load(
                f"{cfg.paths.metrics}/{experiment}_{num_members}_{num_rollout_steps}/predicted_output_metrics_{start}_{end}"
            )
            for start, end in [(0, 24), (24, -24), (-24, None)]
        ]

    def _pad_metrics(m, target_T):
        """Pad a metrics TensorDict to target_T timesteps along dim 0 with NaN."""
        T = m["surface"].shape[0]
        if T >= target_T:
            return m
        from tensordict.tensordict import TensorDict as TD

        def _pad(t):
            pad_shape = (target_T - T, *t.shape[1:])
            return torch.cat([t, torch.full(pad_shape, float("nan"))], dim=0)

        padded = TD({k: _pad(v) for k, v in m.items()})
        padded.batch_size = [target_T]
        return padded

    pred_dict = {
        cfg.analysis.experiment_names[i]: _pad_metrics(
            experiments[i], num_rollout_steps
        )
        for i in range(len(experiments))
    }
    out_dir = experiment + f"_{num_members}_{num_rollout_steps}"
    # climatology_metrics.batch_size = [1]
    # plot_headline_metrics_agg(
    #     pred_dict,
    #     itself_metrics,
    #     f'{cfg.paths.plots}/{out_dir}/climatology_comp_agg',
    #     cfg.analysis.experiment_legend_title
    # )

    # --- Load land-only metrics for TAS comparisons ---
    land_suffix = (
        f"predicted_output_metrics_land_{start_lat}_{end_lat}_{start_lon}_{end_lon}"
    )
    itself_land_path = (
        f"{cfg.paths.metrics}"
        f"/{cfg.analysis.experiment}_{num_members}_{num_rollout_steps}"
        f"/itself_output_metrics_land_{start_lat}_{end_lat}_{start_lon}_{end_lon}"
    )
    itself_metrics_land = (
        torch.load(itself_land_path)
        if os.path.exists(itself_land_path)
        else itself_metrics
    )

    land_experiments = []
    for x in cfg.analysis.experiments_to_plot:
        land_path = f"{cfg.paths.metrics}/{resolve_experiment_dir(x)}/{land_suffix}"
        idx = list(cfg.analysis.experiments_to_plot).index(x)
        land_experiments.append(
            torch.load(land_path) if os.path.exists(land_path) else experiments[idx]
        )
    pred_dict_land = {
        cfg.analysis.experiment_names[i]: _pad_metrics(
            land_experiments[i], num_rollout_steps
        )
        for i in range(len(land_experiments))
    }

    plot_headline_individual_metrics(
        pred_dict,
        itself_metrics,
        f"{cfg.paths.plots}/{out_dir}/individual_headline_crps",
        var_names,
        4,
        cfg.analysis.experiment_legend_title,
        pred_dict_land=pred_dict_land,
        itself_metrics_land=itself_metrics_land,
    )
    plot_headline_individual_metrics(
        pred_dict,
        itself_metrics,
        f"{cfg.paths.plots}/{out_dir}/individual_headline_variance",
        var_names,
        2,
        cfg.analysis.experiment_legend_title,
        pred_dict_land=pred_dict_land,
        itself_metrics_land=itself_metrics_land,
    )

    save_last_two_years_averages(
        pred_dict,
        itself_metrics,
        var_names,
        f"{cfg.paths.plots}/{out_dir}/last_two_years_avg_metrics.txt",
        pred_dict_land=pred_dict_land,
        itself_metrics_land=itself_metrics_land,
    )

    # --- Yearly metrics ---
    yearly_suffix = (
        f"predicted_output_metrics_yearly_{start_lat}_{end_lat}_{start_lon}_{end_lon}"
    )
    itself_yearly_path = (
        f"{cfg.paths.metrics}"
        f"/{cfg.analysis.experiment}_{num_members}_{num_rollout_steps}"
        f"/itself_output_metrics_yearly_{start_lat}_{end_lat}_{start_lon}_{end_lon}"
    )
    itself_metrics_yearly = (
        torch.load(itself_yearly_path) if os.path.exists(itself_yearly_path) else None
    )

    yearly_experiments = []
    for x in cfg.analysis.experiments_to_plot:
        ypath = f"{cfg.paths.metrics}/{resolve_experiment_dir(x)}/{yearly_suffix}"
        yearly_experiments.append(torch.load(ypath) if os.path.exists(ypath) else None)
    pred_dict_yearly = {
        cfg.analysis.experiment_names[i]: yearly_experiments[i]
        for i in range(len(yearly_experiments))
        if yearly_experiments[i] is not None
    }

    land_yearly_suffix = f"predicted_output_metrics_land_yearly_{start_lat}_{end_lat}_{start_lon}_{end_lon}"
    itself_land_yearly_path = (
        f"{cfg.paths.metrics}"
        f"/{cfg.analysis.experiment}_{num_members}_{num_rollout_steps}"
        f"/itself_output_metrics_land_yearly_{start_lat}_{end_lat}_{start_lon}_{end_lon}"
    )
    itself_metrics_land_yearly = (
        torch.load(itself_land_yearly_path)
        if os.path.exists(itself_land_yearly_path)
        else None
    )

    land_yearly_experiments = []
    for x in cfg.analysis.experiments_to_plot:
        lypath = f"{cfg.paths.metrics}/{resolve_experiment_dir(x)}/{land_yearly_suffix}"
        land_yearly_experiments.append(
            torch.load(lypath) if os.path.exists(lypath) else None
        )
    pred_dict_land_yearly = {
        cfg.analysis.experiment_names[i]: land_yearly_experiments[i]
        for i in range(len(land_yearly_experiments))
        if land_yearly_experiments[i] is not None
    }

    if pred_dict_yearly and itself_metrics_yearly is not None:
        plot_yearly_metrics(
            pred_dict_yearly,
            itself_metrics_yearly,
            f"{cfg.paths.plots}/{out_dir}/individual_headline_crps_yearly",
            var_names,
            4,
            cfg.analysis.experiment_legend_title,
            pred_dict_land=pred_dict_land_yearly or None,
            itself_metrics_land=itself_metrics_land_yearly,
        )
        plot_yearly_metrics(
            pred_dict_yearly,
            itself_metrics_yearly,
            f"{cfg.paths.plots}/{out_dir}/individual_headline_variance_yearly",
            var_names,
            2,
            cfg.analysis.experiment_legend_title,
            pred_dict_land=pred_dict_land_yearly or None,
            itself_metrics_land=itself_metrics_land_yearly,
        )
        save_last_two_years_text(
            pred_dict_yearly,
            itself_metrics_yearly,
            var_names,
            f"{cfg.paths.plots}/{out_dir}/last_two_years_yearly_metrics.txt",
            pred_dict_land_yearly=pred_dict_land_yearly or None,
            itself_metrics_land_yearly=itself_metrics_land_yearly,
        )

    save_latex_tables(
        pred_dict,
        itself_metrics,
        var_names,
        f"{cfg.paths.plots}/{out_dir}/metric_tables.tex",
        pred_dict_land=pred_dict_land,
        itself_metrics_land=itself_metrics_land,
        pred_dict_yearly=pred_dict_yearly if pred_dict_yearly else None,
        itself_metrics_yearly=itself_metrics_yearly,
        pred_dict_land_yearly=pred_dict_land_yearly if pred_dict_land_yearly else None,
        itself_metrics_land_yearly=itself_metrics_land_yearly,
    )

    # --- Temporal PSD and rank histogram (requires raw rollout data) ---
    surface_variables = list(cfg.dataloader.dataset.surface_variables)
    level_variables = list(cfg.dataloader.dataset.level_variables)
    inference_target = OmegaConf.select(cfg, "inference.target", default="test")
    inference_year = OmegaConf.select(cfg, "inference.year", default=0)

    try:
        from ArchesClimate.analysis.gen_and_save_metrics import (
            load_experiment,
        )

        train_dataset = hydra.utils.instantiate(cfg.dataloader.dataset, domain="test")
        train_spatial_norm = hydra.utils.instantiate(
            cfg.dataloader.dataset, domain="test", norm_scheme="spatial_norm"
        )

        target_path = f"""{artifact_path("targets")}/{inference_target}_ensemble_{inference_year}_flow_net_flux_1"""
        target_normalized = torch.load(target_path)
        target_normalized["surface"] = target_normalized["surface"][
            :, :, cfg.module.surface_variable_indices
        ]
        target_normalized["level"] = target_normalized["level"][
            :, :, cfg.module.level_variable_indices
        ]
        target_denormed_all = train_spatial_norm.denormalize(
            target_normalized, stateless=True
        )[:, :num_rollout_steps]
        target_denormed = target_denormed_all[:num_members]

        n_surface_vars = len(cfg.dataloader.dataset.surface_variables)
        n_level_vars = len(cfg.dataloader.dataset.level_variables)
        n_plevs = len(cfg.dataloader.dataset.pressure_levels)

        pred_dict_raw = {}
        # var_filter: restrict baselines to their supported variables
        # mesmer → tas-land only; diffesm → tas + tas-land
        baseline_var_filter = {}
        for i, x in enumerate(cfg.analysis.experiments_to_plot):
            exp_label = cfg.analysis.experiment_names[i]
            try:
                if x == "mesmer" and mesmer_year is not None:
                    pred_dict_raw[exp_label] = load_mesmer_tas(
                        mesmer_year,
                        num_members,
                        num_rollout_steps,
                        n_surface_vars,
                        n_level_vars,
                        n_plevs,
                    )
                    baseline_var_filter[exp_label] = {"land:10"}
                elif x == "diffesm" and diffesm_year is not None:
                    pred_dict_raw[exp_label] = load_diffesm_tas(
                        diffesm_year,
                        num_members,
                        num_rollout_steps,
                        n_surface_vars,
                        n_level_vars,
                        n_plevs,
                    )
                    baseline_var_filter[exp_label] = {10, "land:10"}
                else:
                    raw_dir = resolve_raw_experiment_dir(x)
                    raw = load_experiment(cfg, raw_dir, num_members, num_rollout_steps)
                    pred_dict_raw[exp_label] = train_dataset.denormalize(
                        raw, stateless=True
                    )
            except Exception as e:
                LOGGER.warning("Optional analysis input or plot failed", exc_info=True)
                import traceback

                print(f"Skipping {x} ({exp_label}) for PSD/rank histogram: {e}")
                traceback.print_exc()

        # Compute climatology from target itself: monthly mean across all members and timesteps
        tgt_surf_full = target_denormed_all["surface"][
            :, :num_rollout_steps
        ]  # [M, T, vars, 1, lat, lon]
        tgt_lvl_full = target_denormed_all["level"][
            :, :num_rollout_steps
        ]  # [M, T, vars, plevs, lat, lon]
        month_indices = (
            torch.arange(num_rollout_steps) % 12
        )  # 0-based month label per timestep
        clima_surf = torch.zeros(12, *tgt_surf_full.shape[2:])
        clima_lvl = torch.zeros(12, *tgt_lvl_full.shape[2:])
        for m in range(12):
            mask = month_indices == m
            clima_surf[m] = tgt_surf_full[:, mask].nanmean(dim=(0, 1))
            clima_lvl[m] = tgt_lvl_full[:, mask].nanmean(dim=(0, 1))
        # Tile climatology to cover num_rollout_steps timesteps aligned to month_indices
        clima_surf_tiled = clima_surf[month_indices]  # [T, vars, 1, lat, lon]
        clima_lvl_tiled = clima_lvl[month_indices]  # [T, vars, plevs, lat, lon]

        target_surface_ano = (
            target_denormed["surface"][:, :num_rollout_steps] - clima_surf_tiled
        )
        target_level_ano = (
            target_denormed["level"][:, :num_rollout_steps] - clima_lvl_tiled
        )

        pred_anomalies = {}
        for exp_name, pred in pred_dict_raw.items():
            pred_surf_full = pred["surface"][:, :num_rollout_steps]
            pred_lvl_full = pred["level"][:, :num_rollout_steps]
            pred_clima_surf = torch.zeros(12, *pred_surf_full.shape[2:])
            pred_clima_lvl = torch.zeros(12, *pred_lvl_full.shape[2:])
            for m in range(12):
                mask = month_indices == m
                pred_clima_surf[m] = pred_surf_full[:, mask].nanmean(dim=(0, 1))
                pred_clima_lvl[m] = pred_lvl_full[:, mask].nanmean(dim=(0, 1))
            pred_anomalies[exp_name] = (
                pred_surf_full - pred_clima_surf[month_indices],
                pred_lvl_full - pred_clima_lvl[month_indices],
            )

        _target_surface_ano_np = (
            target_surface_ano.numpy()
            if hasattr(target_surface_ano, "numpy")
            else target_surface_ano
        )
        _target_level_ano_np = (
            target_level_ano.numpy()
            if hasattr(target_level_ano, "numpy")
            else target_level_ano
        )
        plot_temporal_psd(
            pred_anomalies,
            _target_surface_ano_np,
            _target_level_ano_np,
            surface_variables,
            level_variables,
            f"{cfg.paths.plots}/{out_dir}/temporal_psd_headline",
            var_filter=baseline_var_filter if baseline_var_filter else None,
        )
        plot_temporal_psd_tas(
            pred_anomalies,
            _target_surface_ano_np,
            _target_level_ano_np,
            surface_variables,
            level_variables,
            f"{cfg.paths.plots}/{out_dir}/temporal_psd_headline",
            var_filter=baseline_var_filter if baseline_var_filter else None,
        )

        from tensordict.tensordict import TensorDict

        from ArchesClimate.analysis.plot_climatology import (
            _alt_period,
            _load_target,
            plot_climatology_envelopes,
        )

        alt_target = "val" if inference_target == "test" else "test"
        try:
            alt_target_denormed = _load_target(
                cfg,
                alt_target,
                inference_year,
                train_spatial_norm,
                num_members,
                num_rollout_steps,
            )
            target_denormed_concat = TensorDict(
                {
                    "surface": torch.cat(
                        [target_denormed["surface"], alt_target_denormed["surface"]],
                        dim=1,
                    ),
                    "level": torch.cat(
                        [target_denormed["level"], alt_target_denormed["level"]], dim=1
                    ),
                },
                batch_size=[],
            )
        except Exception as e:
            LOGGER.warning("Optional analysis input or plot failed", exc_info=True)
            print(
                f"[clim] Could not load alt target ({alt_target}): {e}. Using single period."
            )
            target_denormed_concat = target_denormed

        BASELINES = {"mesmer", "diffesm"}
        pred_dict_raw_concat = {}
        for exp_name, pred in pred_dict_raw.items():
            exp_dir = resolve_raw_experiment_dir(
                cfg.analysis.experiments_to_plot[
                    list(pred_dict_raw.keys()).index(exp_name)
                ]
            )
            if exp_dir in BASELINES:
                pred_dict_raw_concat[exp_name] = pred
                continue
            alt_dir = _alt_period(exp_dir)
            if alt_dir is not None:
                try:
                    alt_raw = load_experiment(
                        cfg, alt_dir, num_members, num_rollout_steps
                    )
                    alt_pred = train_dataset.denormalize(alt_raw, stateless=True)
                    pred_dict_raw_concat[exp_name] = TensorDict(
                        {
                            "surface": torch.cat(
                                [pred["surface"], alt_pred["surface"]], dim=1
                            ),
                            "level": torch.cat(
                                [pred["level"], alt_pred["level"]], dim=1
                            ),
                        },
                        batch_size=[],
                    )
                except Exception as e:
                    LOGGER.warning(
                        "Optional analysis input or plot failed", exc_info=True
                    )
                    print(
                        f"[clim] Could not load alt pred {alt_dir}: {e}. Using single period."
                    )
                    pred_dict_raw_concat[exp_name] = pred
            else:
                pred_dict_raw_concat[exp_name] = pred

        plot_climatology_envelopes(
            pred_dict_raw_concat,
            target_denormed_concat,
            surface_variables,
            level_variables,
            f"{cfg.paths.plots}/{out_dir}/climatology_envelopes",
        )

        # Anomaly versions for rank histogram — same self-climatology removal as PSD
        tgt_surf_ano_all = (
            target_denormed_all["surface"][:, :num_rollout_steps] - clima_surf_tiled
        )
        tgt_lvl_ano_all = (
            target_denormed_all["level"][:, :num_rollout_steps] - clima_lvl_tiled
        )
        pred_dict_raw_flat = {
            name: (surf_ano, lvl_ano)
            for name, (surf_ano, lvl_ano) in pred_anomalies.items()
        }
        _rh_kwargs = {
            "target_surface": tgt_surf_ano_all,
            "target_level": tgt_lvl_ano_all,
            "exp_names": list(cfg.analysis.experiment_names),
            "surface_variables": surface_variables,
            "level_variables": level_variables,
            "num_members": num_members,
            "num_rollout_steps": num_rollout_steps,
            "start_lat": start_lat,
            "end_lat": end_lat if end_lat != "None" else None,
            "start_lon": start_lon,
            "end_lon": end_lon if end_lon != "None" else 144,
            "var_filter": baseline_var_filter if baseline_var_filter else None,
        }
        plot_rank_histogram(
            pred_dict_raw_flat,
            **_rh_kwargs,
            out_path=f"{cfg.paths.plots}/{out_dir}/rank_histogram_headline",
        )
        plot_rank_histogram_tas(
            pred_dict_raw_flat,
            **_rh_kwargs,
            out_path=f"{cfg.paths.plots}/{out_dir}/rank_histogram_headline",
        )
    except Exception as e:
        LOGGER.warning("Optional analysis input or plot failed", exc_info=True)
        import traceback

        print(f"Skipping temporal PSD / rank histogram plots: {e}")
        traceback.print_exc()

    # --- Robinson spatial RMSE maps ---
    spatial_suffix = (
        f"predicted_spatial_rmse_{start_lat}_{end_lat}_{start_lon}_{end_lon}"
    )
    itself_spatial_path = (
        f"{cfg.paths.metrics}"
        f"/{cfg.analysis.experiment}_{num_members}_{num_rollout_steps}"
        f"/itself_spatial_rmse_{start_lat}_{end_lat}_{start_lon}_{end_lon}"
    )
    itself_spatial_rmse = (
        torch.load(itself_spatial_path) if os.path.exists(itself_spatial_path) else None
    )

    if itself_spatial_rmse is not None:
        spatial_rmse_exps = {}
        for i, x in enumerate(cfg.analysis.experiments_to_plot):
            spath = f"{cfg.paths.metrics}/{resolve_experiment_dir(x)}/{spatial_suffix}"
            if os.path.exists(spath):
                spatial_rmse_exps[cfg.analysis.experiment_names[i]] = torch.load(spath)

        if spatial_rmse_exps:
            plot_robinson_spatial_rmse(
                spatial_rmse_exps,
                itself_spatial_rmse,
                f"{cfg.paths.plots}/{out_dir}/robinson_spatial_rmse",
                var_names,
            )


if __name__ == "__main__":
    main()
