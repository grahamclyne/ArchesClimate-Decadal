"""Regenerate temporal_psd_headline.pdf and rank_histogram_headline.pdf, both
pooled across ALL 11 available hindcast starts, to address a reviewer's
small-sample concern -- previously only one start (val_0) was used for both.

Both are pooled by incrementally accumulating each start's contribution and
discarding that start's raw anomaly tensors before moving to the next, rather
than concatenating raw per-start tensors in memory first -- keeps peak memory
at roughly one start's worth of data instead of all 11 at once, which is what
made an earlier concatenation-based version of this script OOM on a 40-cpu
(~160GB) node. Rank histograms accumulate into a running RankHistogramUltraFast
sum (it sums per-rollout-step histograms across update() calls -- this is
exactly how plot_metrics.py's own methodology text already describes "pooling
ranks over every available initialization period"). PSD accumulates each
start's small per-frequency-bin curve (see _compute_psd_for_var) into a list,
averaged across starts at the end -- equivalent to widening a Welch-style
periodogram average from 5 ensemble members to 5 members x 11 starts.

Usage: python regen_psd_rankhist.py
"""

import warnings

from ArchesClimate.paths import config_directory
from ArchesClimate.paths import path as artifact_path

warnings.filterwarnings("ignore")
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from tensordict.tensordict import TensorDict

from ArchesClimate.analysis.gen_and_save_metrics import load_experiment
from ArchesClimate.analysis.gen_baseline_metrics import load_diffesm, load_mesmer
from ArchesClimate.analysis.plot_metrics import (
    RankHistogramUltraFast,
    _compute_psd_for_var,
    _draw_rank_hist_axes,
    plot_temporal_psd,
    plot_temporal_psd_tas,
)

CD = str(config_directory())
TARGET_BASE = str(artifact_path("targets"))
OUT_DIR = str(artifact_path("plots") / "headline_figures_pooled")
DIFFESM_SAVE_NAME = "new_monthcond"

CK = "epoch-epoch=19.ckpt"
NUM_MEMBERS = 5
NUM_ROLLOUT_STEPS = 118

MODELS = [
    ("2D-E192-CL", "flow_ensemble_small", "flow_ensemble_small_cl"),
    ("2D-E768-CL", "flow_ensemble_large", "flow_ensemble_large_cl"),
    ("2D-E768-NCL", "flow_ensemble", "flow_ensemble_ncl_v2"),
    ("2D-S768-CL", "flow_ensemble_single_small", "flow_ensemble_single_small_cl"),
    (
        "2D-S768-NCL",
        "flow_ensemble_single_not_spectral",
        "flow_ensemble_single_not_spectral_v2",
    ),
    ("3D-S768-CL", "flow_3d_prior", "flow_3d_prior_cl"),
]

# (inference.target, inference.year, diffesm/mesmer calendar year)
STARTS = [
    ("test", 0, 1969),
    ("test", 10, 1979),
    ("test", 20, 1989),
    ("test", 30, 1999),
    ("test", 40, 2009),
    ("val", 0, 2010),
    ("val", 10, 2011),
    ("val", 20, 2012),
    ("val", 30, 2013),
    ("val", 40, 2014),
    ("val", 50, 2015),
]

BASELINE_VAR_FILTER = {"MESMER-M": {"land:10"}, "DiffESM": {10, "land:10"}}

# Union of plot_temporal_psd's headline_vars and plot_temporal_psd_tas's tas_vars,
# so both figures' PSD data can be accumulated in the same per-start pass below.
ALL_PSD_VARS = [6, -5, -1, "land:10", (2, 1), (5, 1), 10]


def compose_cfg():
    with initialize_config_dir(version_base=None, config_dir=CD):
        return compose(
            config_name="config",
            overrides=[
                "module=flow_ensemble_single_small",
                "inference=inference_25",
                "inference.num_inference_steps=24",
            ],
        )


def load_target(cfg, train_spatial_norm, tgt, yr):
    path = f"{TARGET_BASE}/{tgt}_ensemble_{yr}_flow_net_flux_1"
    raw = torch.load(path, map_location="cpu")
    surf = raw["surface"]
    if surf.shape[2] > len(cfg.module.surface_variable_indices):
        surf = surf[:, :, cfg.module.surface_variable_indices]
    lvl = raw["level"]
    if lvl.shape[2] > len(cfg.module.level_variable_indices):
        lvl = lvl[:, :, cfg.module.level_variable_indices]
    raw = TensorDict({"surface": surf, "level": lvl})
    denormed = train_spatial_norm.denormalize(raw, stateless=True)
    return denormed["surface"][:, :NUM_ROLLOUT_STEPS], denormed["level"][
        :, :NUM_ROLLOUT_STEPS
    ]


def climatology_anomaly(surf, lvl):
    """Per-series self-climatology removal, identical recipe to plot_metrics.py's main()."""
    month_indices = torch.arange(NUM_ROLLOUT_STEPS) % 12
    clima_s = torch.zeros(12, *surf.shape[2:])
    clima_l = torch.zeros(12, *lvl.shape[2:])
    for m in range(12):
        mask = month_indices == m
        clima_s[m] = surf[:, mask].nanmean(dim=(0, 1))
        clima_l[m] = lvl[:, mask].nanmean(dim=(0, 1))
    return surf - clima_s[month_indices], lvl - clima_l[month_indices]


def update_model_rankhist(
    surf_rh,
    surf_land_rh,
    level_rh,
    ocean_mask,
    p_surf_ano,
    p_lvl_ano,
    tgt_surf,
    tgt_surf_land,
    tgt_level,
):
    """One start's worth of updates for one model, mirroring _compute_rank_histograms's inner loop."""
    for i in range(NUM_MEMBERS):
        pred_surf_i = p_surf_ano[i, :, :, 0][None, None]
        surf_rh.update(pred_surf_i, tgt_surf[None])
        pred_surf_land_i = pred_surf_i.clone()
        pred_surf_land_i[:, :, :, 10][..., ocean_mask] = float("nan")
        surf_land_rh.update(pred_surf_land_i, tgt_surf_land[None])
        level_rh.update(p_lvl_ano[i, :, :, :][None, None], tgt_level[None])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ref_cfg = compose_cfg()
    train_dataset = hydra.utils.instantiate(ref_cfg.dataloader.dataset, domain="test")
    train_spatial_norm = hydra.utils.instantiate(
        ref_cfg.dataloader.dataset, domain="test", norm_scheme="spatial_norm"
    )
    surface_variables = list(ref_cfg.dataloader.dataset.surface_variables)
    level_variables = list(ref_cfg.dataloader.dataset.level_variables)
    n_surface_vars = len(surface_variables)
    n_level_vars = len(level_variables)
    n_plevs = len(ref_cfg.dataloader.dataset.pressure_levels)
    target_lats = np.linspace(-90, 90, 143)
    target_lons = np.linspace(0, 360, 144)

    exp_names = [label for label, _, _ in MODELS] + ["DiffESM", "MESMER-M"]

    surf_rh = {
        n: RankHistogramUltraFast(
            NUM_MEMBERS, data_shape=(NUM_ROLLOUT_STEPS, n_surface_vars)
        )
        for n in exp_names
    }
    surf_land_rh = {
        n: RankHistogramUltraFast(
            NUM_MEMBERS, data_shape=(NUM_ROLLOUT_STEPS, n_surface_vars)
        )
        for n in exp_names
    }
    level_rh = {
        n: RankHistogramUltraFast(
            NUM_MEMBERS, data_shape=(NUM_ROLLOUT_STEPS, n_level_vars, n_plevs)
        )
        for n in exp_names
    }
    itself_surf_rh = RankHistogramUltraFast(
        NUM_MEMBERS, data_shape=(NUM_ROLLOUT_STEPS, n_surface_vars)
    )
    itself_surf_land_rh = RankHistogramUltraFast(
        NUM_MEMBERS, data_shape=(NUM_ROLLOUT_STEPS, n_surface_vars)
    )
    itself_level_rh = RankHistogramUltraFast(
        NUM_MEMBERS, data_shape=(NUM_ROLLOUT_STEPS, n_level_vars, n_plevs)
    )

    single_pred_anom = {}
    single_tgt_surf_ano = single_tgt_lvl_ano = None
    n_starts_used_for_itself = 0

    # PSD pooled across all 11 starts: var_index -> exp_name -> {'period', 'target_psds', 'pred_psds'}.
    # Each entry is a small 1D array (one per frequency bin), so accumulating these
    # across starts (unlike the raw anomaly tensors) is cheap and needs no streaming
    # care beyond what's already here.
    psd_accum = {v: {} for v in ALL_PSD_VARS}

    for tgt, yr, cal_yr in STARTS:
        print(f"=== start {tgt}_{yr} (cal {cal_yr}) ===", flush=True)
        t_surf, t_lvl = load_target(ref_cfg, train_spatial_norm, tgt, yr)
        t_surf_ano, t_lvl_ano = climatology_anomaly(t_surf, t_lvl)

        ocean_mask = ~torch.isnan(t_surf_ano[0, 0, -1, 0])
        tgt_surf = t_surf_ano[:NUM_MEMBERS, :, :, 0]
        tgt_surf_land = tgt_surf.clone()
        tgt_surf_land[:, :, 10][..., ocean_mask] = float("nan")
        tgt_level = t_lvl_ano[:NUM_MEMBERS]

        this_start_anom = {}
        for label, module, name in MODELS:
            exp = f"{name}_24_5_1_118_1_{tgt}_{yr}_{CK}"
            raw = load_experiment(ref_cfg, exp, NUM_MEMBERS, NUM_ROLLOUT_STEPS)
            pred = train_dataset.denormalize(raw, stateless=True)
            p_surf, p_lvl = (
                pred["surface"][:, :NUM_ROLLOUT_STEPS],
                pred["level"][:, :NUM_ROLLOUT_STEPS],
            )
            p_surf_ano, p_lvl_ano = climatology_anomaly(p_surf, p_lvl)
            update_model_rankhist(
                surf_rh[label],
                surf_land_rh[label],
                level_rh[label],
                ocean_mask,
                p_surf_ano,
                p_lvl_ano,
                tgt_surf,
                tgt_surf_land,
                tgt_level,
            )
            this_start_anom[label] = (p_surf_ano, p_lvl_ano)
            del raw, pred, p_surf, p_lvl

        diffesm_raw = load_diffesm(
            cal_yr,
            NUM_ROLLOUT_STEPS,
            n_surface_vars,
            n_level_vars,
            n_plevs,
            0,
            None,
            0,
            None,
            save_name=DIFFESM_SAVE_NAME,
        )
        d_surf = diffesm_raw["surface"][0, :NUM_MEMBERS, :NUM_ROLLOUT_STEPS]
        d_lvl = diffesm_raw["level"][0, :NUM_MEMBERS, :NUM_ROLLOUT_STEPS]
        actual_T = d_surf.shape[1]
        if actual_T < NUM_ROLLOUT_STEPS:
            pad = NUM_ROLLOUT_STEPS - actual_T
            d_surf = torch.cat(
                [
                    d_surf,
                    torch.full((d_surf.shape[0], pad, *d_surf.shape[2:]), float("nan")),
                ],
                dim=1,
            )
            d_lvl = torch.cat(
                [
                    d_lvl,
                    torch.full((d_lvl.shape[0], pad, *d_lvl.shape[2:]), float("nan")),
                ],
                dim=1,
            )
        d_surf_ano, d_lvl_ano = climatology_anomaly(d_surf, d_lvl)
        update_model_rankhist(
            surf_rh["DiffESM"],
            surf_land_rh["DiffESM"],
            level_rh["DiffESM"],
            ocean_mask,
            d_surf_ano,
            d_lvl_ano,
            tgt_surf,
            tgt_surf_land,
            tgt_level,
        )
        this_start_anom["DiffESM"] = (d_surf_ano, d_lvl_ano)

        mesmer_raw = load_mesmer(
            cal_yr,
            NUM_ROLLOUT_STEPS,
            n_surface_vars,
            n_level_vars,
            n_plevs,
            target_lats,
            target_lons,
            0,
            None,
            0,
            None,
        )
        m_surf = mesmer_raw["surface"][0, :NUM_MEMBERS, :NUM_ROLLOUT_STEPS]
        m_lvl = mesmer_raw["level"][0, :NUM_MEMBERS, :NUM_ROLLOUT_STEPS]
        m_surf_ano, m_lvl_ano = climatology_anomaly(m_surf, m_lvl)
        update_model_rankhist(
            surf_rh["MESMER-M"],
            surf_land_rh["MESMER-M"],
            level_rh["MESMER-M"],
            ocean_mask,
            m_surf_ano,
            m_lvl_ano,
            tgt_surf,
            tgt_surf_land,
            tgt_level,
        )
        this_start_anom["MESMER-M"] = (m_surf_ano, m_lvl_ano)

        # PSD compares member-for-member against the predicted (5-member) ensembles,
        # so cap the target back down to NUM_MEMBERS here (unlike the rank-histogram
        # path above, which deliberately uses every available target member).
        start_tgt_surf_ano = t_surf_ano[:NUM_MEMBERS]
        start_tgt_lvl_ano = t_lvl_ano[:NUM_MEMBERS]
        if (tgt, yr) == ("val", 0):
            single_pred_anom = this_start_anom
            single_tgt_surf_ano, single_tgt_lvl_ano = (
                start_tgt_surf_ano,
                start_tgt_lvl_ano,
            )

        # --- Accumulate PSD for this start (pooled across all 11 at the end) ---
        # DiffESM's later val starts (2011-2015) are NaN-padded up to NUM_ROLLOUT_STEPS
        # (see the padding above); an FFT over NaN-padded data is all-NaN, so any
        # (var, exp) result containing NaN is dropped rather than pooled -- MESMER-M
        # and every ArchesClimate config have full real data for every start.
        start_tgt_surf_np = start_tgt_surf_ano.numpy()
        start_tgt_lvl_np = start_tgt_lvl_ano.numpy()
        start_ocean_mask_np = ~np.isnan(start_tgt_surf_np[0, 0, -1, 0])
        for var_index in ALL_PSD_VARS:
            var_results = _compute_psd_for_var(
                var_index,
                this_start_anom,
                start_tgt_surf_np,
                start_tgt_lvl_np,
                start_ocean_mask_np,
                BASELINE_VAR_FILTER,
            )
            for exp_name, (period, tgt_psd, prd_psd) in var_results.items():
                if np.isnan(tgt_psd).any() or np.isnan(prd_psd).any():
                    print(
                        f"  [psd] skipping {exp_name} var={var_index} start {tgt}_{yr}: NaN in PSD (short/padded data)",
                        flush=True,
                    )
                    continue
                acc = psd_accum[var_index].setdefault(
                    exp_name, {"period": period, "target_psds": [], "pred_psds": []}
                )
                acc["target_psds"].append(tgt_psd)
                acc["pred_psds"].append(prd_psd)

        # "itself" (IPSL vs itself) accumulation, if this start's target has spare members
        n_total = t_surf_ano.shape[0]
        pred_members = min(NUM_MEMBERS, n_total // 2)
        if pred_members == NUM_MEMBERS:
            n_truth_members = n_total - pred_members
            group_a_surf = t_surf_ano[:pred_members, :, :, 0]
            group_a_surf_land = group_a_surf.clone()
            group_a_surf_land[:, :, 10][..., ocean_mask] = float("nan")
            group_a_level = t_lvl_ano[:pred_members]
            full_surf = t_surf_ano[:, :, :, 0]
            full_surf_land = full_surf.clone()
            full_surf_land[:, :, 10][..., ocean_mask] = float("nan")
            for i in range(n_truth_members):
                j = pred_members + i
                itself_surf_rh.update(full_surf[j][None, None], group_a_surf[None])
                itself_surf_land_rh.update(
                    full_surf_land[j][None, None], group_a_surf_land[None]
                )
                itself_level_rh.update(t_lvl_ano[j][None, None], group_a_level[None])
            n_starts_used_for_itself += 1
        else:
            print(
                f"  [itself] skipping start {tgt}_{yr}: only {n_total} target members available",
                flush=True,
            )

    print(
        f"[itself] pooled from {n_starts_used_for_itself}/{len(STARTS)} starts",
        flush=True,
    )

    # --- Temporal PSD: pooled across all 11 starts, corrected checkpoints ---
    final_psd = {}
    for var_index in ALL_PSD_VARS:
        final_psd[var_index] = {
            exp_name: (
                acc["period"],
                np.mean(acc["target_psds"], axis=0),
                np.mean(acc["pred_psds"], axis=0),
            )
            for exp_name, acc in psd_accum[var_index].items()
        }
        n_pooled = {
            exp_name: len(acc["target_psds"])
            for exp_name, acc in psd_accum[var_index].items()
        }
        print(f"[psd] var={var_index} pooled starts per exp: {n_pooled}", flush=True)

    plot_temporal_psd(
        single_pred_anom,
        single_tgt_surf_ano.numpy(),
        single_tgt_lvl_ano.numpy(),
        surface_variables,
        level_variables,
        f"{OUT_DIR}/temporal_psd_headline",
        var_filter=BASELINE_VAR_FILTER,
        precomputed_psd=final_psd,
    )
    plot_temporal_psd_tas(
        single_pred_anom,
        single_tgt_surf_ano.numpy(),
        single_tgt_lvl_ano.numpy(),
        surface_variables,
        level_variables,
        f"{OUT_DIR}/temporal_psd_headline",
        var_filter=BASELINE_VAR_FILTER,
        precomputed_psd=final_psd,
    )
    print("Saved temporal_psd_headline(.pdf/_tas.pdf)", flush=True)

    # --- Rank histogram: pooled across all 11 hindcast starts (streamed accumulation) ---
    exp_surface_hists = [surf_rh[n].compute()["rankhist"] for n in exp_names]
    exp_surface_land_hists = [surf_land_rh[n].compute()["rankhist"] for n in exp_names]
    exp_level_hists = [level_rh[n].compute()["rankhist"] for n in exp_names]
    itself_surf_hist = itself_surf_rh.compute()["rankhist"]
    itself_surf_land_hist = itself_surf_land_rh.compute()["rankhist"]
    itself_level_hist = itself_level_rh.compute()["rankhist"]

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

    # headline (3x2) figure
    headline_indices = [6, -5, -1, "land:10", (2, 1), (5, 1)]
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
            BASELINE_VAR_FILTER,
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
    fig.savefig(f"{OUT_DIR}/rank_histogram_headline.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # tas-only (1x2) figure
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
            BASELINE_VAR_FILTER,
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
            ncol=len(legend_handles),
            frameon=False,
        )
    plt.subplots_adjust(wspace=0.4)
    fig.savefig(
        f"{OUT_DIR}/rank_histogram_headline_tas.pdf", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    print(
        f"Pooled rank-histogram sample size: {len(STARTS)} starts x {NUM_ROLLOUT_STEPS} = {len(STARTS) * NUM_ROLLOUT_STEPS} timesteps per model",
        flush=True,
    )
    print("Saved rank_histogram_headline(.pdf/_tas.pdf)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
