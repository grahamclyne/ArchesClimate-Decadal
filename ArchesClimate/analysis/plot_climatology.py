"""Climatology envelope plotting: load data (test+val) and produce the figure."""

import logging
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict.tensordict import TensorDict

from ArchesClimate.paths import path as artifact_path

LOGGER = logging.getLogger(__name__)


def load_mesmer_tas(
    year, num_members, num_rollout_steps, n_surface_vars, n_level_vars, n_plevs
):
    from ArchesClimate.analysis.gen_baseline_metrics import load_mesmer

    # Use the same 143×144 grid as the rest of the project
    target_lats = np.linspace(-90, 90, 143)
    target_lons = np.linspace(0, 360, 144)
    td = load_mesmer(
        year,
        num_rollout_steps,
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
    out = TensorDict(
        {
            "surface": td["surface"][0, :num_members],
            "level": td["level"][0, :num_members],
        }
    )
    out.batch_size = [num_members, num_rollout_steps]
    return out


def load_diffesm_tas(
    year,
    num_members,
    num_rollout_steps,
    n_surface_vars,
    n_level_vars,
    n_plevs,
    save_name="new_monthcond",
):
    from ArchesClimate.analysis.gen_baseline_metrics import load_diffesm

    td = load_diffesm(
        year,
        num_rollout_steps,
        n_surface_vars,
        n_level_vars,
        n_plevs,
        0,
        None,
        0,
        None,
        save_name=save_name,
    )
    out = TensorDict(
        {
            "surface": td["surface"][0, :num_members],
            "level": td["level"][0, :num_members],
        }
    )
    out.batch_size = [num_members, num_rollout_steps]
    return out


# The 6 headline variables used in rank histogram and temporal PSD:
# psl, thetaot2000, tos, tas (land), ta 700hPa, wap 700hPa.
HEADLINE_VARS = [6, -5, -1, "land:10", (2, 1), (5, 1)]
VAR_TITLES = ["psl", "thetaot2000", "tos", "tas (land)", "ta 700hPa", "wap 700hPa"]
VAR_YLABELS = ["Pa", "°C", "°C", "°C", "K", "Pa s⁻¹"]

REGIONS = [(0, 56), (56, 86), (86, 143)]
REGION_NAMES = ["90°S–30°S", "30°S–30°N", "30°N–90°N"]

_PALETTE = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#999999",
]
_MONTH_LABELS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


def _extract(
    data, var_index, ocean_mask, lat_start=None, lat_end=None, common_mask=None
):
    """Pull one headline variable out of a raw {'surface','level'} TensorDict.

    lat_start/lat_end select a latitude band (None = the full globe, matching
    the non-regional climatology plot).
    """
    ls = slice(lat_start, lat_end)
    if isinstance(var_index, tuple):
        return data["level"][:, :, var_index[0], var_index[1], ls]
    elif isinstance(var_index, str) and var_index.startswith("land:"):
        vi = int(var_index.split(":")[1])
        t = data["surface"][:, :, vi, 0, ls].clone()
        t[:, :, ocean_mask[ls]] = float("nan")
        if common_mask is not None:
            t[:, :, ~common_mask] = float("nan")
        return t
    else:
        return data["surface"][:, :, var_index, 0, ls]


def _member_clima(tensor, start_month, sm=None):
    """tensor: [members, T, ...spatial] -> [members, 12] spatial nanmean by calendar month."""
    if sm is None:
        sm = start_month
    spatial = tensor.flatten(start_dim=2).nanmean(dim=-1)  # [M, T]
    T = spatial.shape[1]
    t_month_idx = (torch.arange(T) + sm - 1) % 12
    return torch.stack([spatial[:, t_month_idx == m].mean(1) for m in range(12)], dim=1)


def _reduce_to_climatology(data, exp_name, ocean_mask, mesmer_land_mask, start_month):
    """Reduce one model's full raw {'surface','level'} tensor down to the tiny
    per-headline-variable [members, 12] monthly climatology (global, and per
    latitude band), so the big raw tensor can be discarded immediately after.
    """
    # MESMER-M/DiffESM's raw data starts at real January, while ArchesClimate
    # rollouts/target save t=0 as real February (see start_month's definition
    # in flow_ensemble_comparison.yaml) -- a 1-month gap, not 2.
    sm = (
        start_month - 1
        if ("mesmer" in exp_name.lower() or "diffesm" in exp_name.lower())
        else None
    )
    global_cl = {}
    region_cl = {}
    for var_index, title in zip(HEADLINE_VARS, VAR_TITLES):
        is_land = isinstance(var_index, str) and var_index.startswith("land:")
        land_common = mesmer_land_mask if is_land else None
        global_cl[title] = _member_clima(
            _extract(data, var_index, ocean_mask, common_mask=land_common),
            start_month,
            sm=sm,
        )

        region_cl[title] = {}
        for (lat_s, lat_e), reg_name in zip(REGIONS, REGION_NAMES):
            lc = (
                mesmer_land_mask[lat_s:lat_e]
                if (is_land and mesmer_land_mask is not None)
                else None
            )
            region_cl[title][reg_name] = _member_clima(
                _extract(
                    data,
                    var_index,
                    ocean_mask,
                    lat_start=lat_s,
                    lat_end=lat_e,
                    common_mask=lc,
                ),
                start_month,
                sm=sm,
            )
    return global_cl, region_cl


def plot_climatology_envelopes(pred_global_cl, target_global_cl, out_path):
    """Plot 12-month climatology envelope (mean ± std across members) for all experiments.

    Layout is 3x2 to match the rank-histogram/PSD figures.
    Takes precomputed [members, 12] climatologies (see _reduce_to_climatology),
    not raw prediction tensors.
    """
    x = list(range(1, 13))

    def draw_members(ax, cl, color, label):
        cl_np = cl.numpy()
        for m in range(cl_np.shape[0]):
            ax.plot(x, cl_np[m], color=color, linewidth=0.5, alpha=0.4)
        ax.plot(x, cl_np.mean(0), color=color, linewidth=2.5, label=label)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()

    for idx, (title, ylabel) in enumerate(zip(VAR_TITLES, VAR_YLABELS)):
        ax = axes[idx]
        draw_members(ax, target_global_cl[title], "black", "IPSL")

        for exp_idx, (exp_name, cl_by_title) in enumerate(pred_global_cl.items()):
            draw_members(
                ax, cl_by_title[title], _PALETTE[exp_idx % len(_PALETTE)], exp_name
            )

        ax.set_title(title, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(axis="y", labelsize=10)

    for ax in axes[-2:]:
        ax.set_xticks(x)
        ax.set_xticklabels(_MONTH_LABELS, fontsize=11)
        ax.set_xlabel("Month", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        fontsize=11,
        frameon=False,
    )
    plt.subplots_adjust(hspace=0.4, wspace=0.35, top=0.88)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path + ".pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_climatology_envelopes_by_region(pred_region_cl, target_region_cl, out_path):
    """Same as plot_climatology_envelopes but split into 3 latitude bands.

    Layout: 6 rows (variables) x 3 cols (south 0-56, tropics 56-86, north 86-143).
    Takes precomputed climatologies (see _reduce_to_climatology).
    """
    x = list(range(1, 13))

    def draw_diff_members(ax, diff_cl, color, label):
        cl_np = diff_cl.numpy()
        for m in range(cl_np.shape[0]):
            ax.plot(x, cl_np[m], color=color, linewidth=0.5, alpha=0.4)
        ax.plot(x, cl_np.mean(0), color=color, linewidth=2.5, label=label)

    n_vars = len(VAR_TITLES)
    n_regions = len(REGIONS)
    fig, axes = plt.subplots(
        n_vars,
        n_regions,
        figsize=(4 * n_regions, 2.5 * n_vars),
        sharex=True,
        sharey="row",
    )

    for row, (title, ylabel) in enumerate(zip(VAR_TITLES, VAR_YLABELS)):
        for col, reg_name in enumerate(REGION_NAMES):
            ax = axes[row, col]
            tgt_cl = target_region_cl[title][reg_name]
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            for exp_idx, (exp_name, cl_by_title) in enumerate(pred_region_cl.items()):
                pred_cl = cl_by_title[title][reg_name]
                diff_cl = pred_cl - tgt_cl
                draw_diff_members(
                    ax, diff_cl, _PALETTE[exp_idx % len(_PALETTE)], exp_name
                )
            if row == 0:
                ax.set_title(reg_name, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"Δ {title}\n[{ylabel}]", fontsize=8)
            if row == n_vars - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(_MONTH_LABELS)
                ax.set_xlabel("Month")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        fontsize=9,
        frameon=False,
    )
    plt.subplots_adjust(hspace=0.35, wspace=0.2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path + ".pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _load_target(
    cfg, target_name, year, train_spatial_norm, num_members, num_rollout_steps
):
    path = (
        f"""{artifact_path("targets")}/{target_name}_ensemble_{year}_flow_net_flux_1"""
    )
    data = torch.load(path)
    # Some target files are already sliced down to the selected variable subset (not the
    # full legacy 19-var pool); indexing those again with surface_variable_indices (which
    # goes up to 18) throws IndexError. Guard exactly like gen_and_save_metrics.py's
    # load_predicted_and_target_ensembles and regen_psd_rankhist.py's load_target do --
    # without this, 4 of the 11 hindcast starts (test_40, val_30/40/50) were silently
    # dropped from every target pooling in this file, while predictions still pooled all 11.
    if data["surface"].shape[2] > len(cfg.module.surface_variable_indices):
        data["surface"] = data["surface"][:, :, cfg.module.surface_variable_indices]
    if data["level"].shape[2] > len(cfg.module.level_variable_indices):
        data["level"] = data["level"][:, :, cfg.module.level_variable_indices]
    return train_spatial_norm.denormalize(data, stateless=True)[
        :num_members, :num_rollout_steps
    ]


def _alt_period(name):
    """Swap val_0 <-> test_0 in an experiment directory name."""
    if "_val_" in name:
        return name.replace("_val_", "_test_", 1)
    if "_test_" in name:
        return name.replace("_test_", "_val_", 1)
    return None


# All 11 available hindcast starts (see sec:hindcast_bootstrap), with the
# calendar year each corresponds to for the mesmer/diffesm baselines.
ALL_STARTS = [
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

_START_RE = __import__("re").compile(r"_(test|val)_\d+_")


def _start_variant(name, tgt, yr):
    """Replace whichever _test_N_/_val_N_ segment is in an experiment directory
    name with _{tgt}_{yr}_, so we can load any of the 11 hindcast starts for a
    given model checkpoint."""
    new, n = _START_RE.subn(f"_{tgt}_{yr}_", name, count=1)
    if n == 0:
        raise ValueError(f"could not find a _test_N_/_val_N_ segment in {name!r}")
    return new


def _load_pooled_pred(
    cfg,
    x,
    mesmer_year,
    diffesm_year,
    num_members,
    num_rollout_steps,
    n_surface_vars,
    n_level_vars,
    n_plevs,
    train_dataset,
    load_experiment,
):
    """Load one experiment, pooled across all 11 hindcast starts. Returns a
    {'surface','level'} TensorDict (float32) or None if nothing could be loaded.

    Kept in float32 throughout -- an earlier version accumulated parts in half
    precision to save memory, but float16's ~3-4 significant digits silently
    rounded away psl's genuine ~10-28 Pa bias signal (psl's absolute values are
    ~101,000+ Pa, where float16's rounding step is ~64-128 Pa, larger than the
    signal itself), while leaving smaller-magnitude variables (tos, thetaot2000,
    ta, wap, tas) apparently unaffected. Peak memory is no longer a problem here
    since main() already reduces each model to its tiny climatology immediately
    after loading it (see _reduce_to_climatology) rather than holding all 8
    models' pooled raw tensors simultaneously.
    """
    surf_parts, lvl_parts = [], []
    for tgt, yr, cal_yr in ALL_STARTS:
        try:
            if x == "mesmer" and mesmer_year is not None:
                pred = load_mesmer_tas(
                    cal_yr,
                    num_members,
                    num_rollout_steps,
                    n_surface_vars,
                    n_level_vars,
                    n_plevs,
                )
            elif x == "diffesm" and diffesm_year is not None:
                pred = load_diffesm_tas(
                    cal_yr,
                    num_members,
                    num_rollout_steps,
                    n_surface_vars,
                    n_level_vars,
                    n_plevs,
                )
            else:
                variant = _start_variant(x, tgt, yr)
                raw = load_experiment(cfg, variant, num_members, num_rollout_steps)
                pred = train_dataset.denormalize(raw, stateless=True)
            surf_parts.append(pred["surface"])
            lvl_parts.append(pred["level"])
        except Exception as e:
            LOGGER.warning("Optional analysis input or plot failed", exc_info=True)
            print(f"[clim] Skipping {x} start {tgt}_{yr}: {e}")
    if not surf_parts:
        return None, 0
    n_parts = len(surf_parts)
    out = TensorDict(
        {
            "surface": torch.cat(surf_parts, dim=1),
            "level": torch.cat(lvl_parts, dim=1),
        },
        batch_size=[],
    )
    del surf_parts, lvl_parts
    return out, n_parts


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    from ArchesClimate.analysis.gen_and_save_metrics import load_experiment

    num_members = cfg.analysis.num_members
    num_rollout_steps = cfg.analysis.num_rollout_steps
    mesmer_year = OmegaConf.select(cfg, "analysis.mesmer_year", default=None)
    diffesm_year = OmegaConf.select(cfg, "analysis.diffesm_year", default=None)
    start_month = OmegaConf.select(cfg, "analysis.start_month", default=1)
    out_dir = cfg.analysis.experiment

    train_dataset = hydra.utils.instantiate(cfg.dataloader.dataset, domain="test")
    train_spatial_norm = hydra.utils.instantiate(
        cfg.dataloader.dataset, domain="test", norm_scheme="spatial_norm"
    )

    n_surface_vars = len(cfg.dataloader.dataset.surface_variables)
    n_level_vars = len(cfg.dataloader.dataset.level_variables)
    n_plevs = len(cfg.dataloader.dataset.pressure_levels)

    # --- Load target, pooled across all 11 hindcast starts (sec:hindcast_bootstrap) ---
    # Kept in float32 throughout -- see _load_pooled_pred's docstring for why half
    # precision silently destroyed psl's genuine bias signal.
    target_parts = []
    for tgt, yr, _cal in ALL_STARTS:
        try:
            t = _load_target(
                cfg, tgt, yr, train_spatial_norm, num_members, num_rollout_steps
            )
            target_parts.append(
                TensorDict(
                    {"surface": t["surface"], "level": t["level"]}, batch_size=[]
                )
            )
        except Exception as e:
            LOGGER.warning("Optional analysis input or plot failed", exc_info=True)
            print(f"[clim] Could not load target {tgt}_{yr}: {e}. Skipping this start.")
    n_target_parts = len(target_parts)
    target_denormed = TensorDict(
        {
            "surface": torch.cat([p["surface"] for p in target_parts], dim=1),
            "level": torch.cat([p["level"] for p in target_parts], dim=1),
        },
        batch_size=[],
    )
    del target_parts
    print(
        f"[clim] Pooled {n_target_parts}/{len(ALL_STARTS)} target starts -> {target_denormed['surface'].shape[1]} timesteps"
    )

    # TOS is NaN over land -> isnan(tos) is True over land, False over ocean
    land_mask = torch.isnan(target_denormed["surface"][0, 0, -1, 0])  # True over land
    ocean_mask = ~land_mask  # True over ocean

    # --- Load + immediately reduce each experiment to its tiny monthly
    # climatology, one at a time, so we never hold more than one model's full
    # raw pooled tensor (11 starts x all variables/levels) in memory at once.
    # Holding all 8 models' raw tensors simultaneously (even in half precision)
    # OOMs; the two plotting functions only ever need 6 headline variables'
    # worth of [members, 12] numbers per model, so we throw the rest away here.
    exp_pairs = list(
        zip(cfg.analysis.experiments_to_plot, cfg.analysis.experiment_names)
    )

    # MESMER's land-pixel mask is needed to reduce every OTHER model's
    # tas(land) channel, so MESMER (if present) is loaded and reduced first.
    mesmer_label = None
    computed = {}
    for x, lbl in exp_pairs:
        if x == "mesmer" and mesmer_year is not None:
            raw, n_parts = _load_pooled_pred(
                cfg,
                x,
                mesmer_year,
                diffesm_year,
                num_members,
                num_rollout_steps,
                n_surface_vars,
                n_level_vars,
                n_plevs,
                train_dataset,
                load_experiment,
            )
            if raw is None:
                print(f"[clim] No starts loaded for {lbl}, dropping it entirely.")
                mesmer_land_mask = None
                break
            print(
                f"[clim] Pooled {n_parts}/{len(ALL_STARTS)} starts for {lbl} -> {raw['surface'].shape[1]} timesteps"
            )
            vi_tas = 10
            mesmer_tas = raw["surface"][0, 0, vi_tas, 0].clone()
            mesmer_tas[ocean_mask] = float("nan")
            mesmer_land_mask = ~mesmer_tas.isnan() & land_mask  # [lat, lon] True = keep
            print(
                f"[clim] tas(land) common mask: {mesmer_land_mask.sum().item()} pixels "
                f"(IPSL land: {land_mask.sum().item()}, MESMER: {(~mesmer_tas.isnan()).sum().item()})"
            )
            mesmer_label = lbl
            computed[lbl] = _reduce_to_climatology(
                raw, x, ocean_mask, mesmer_land_mask, start_month
            )
            del raw
            break
    else:
        mesmer_land_mask = None

    # Target's own climatology (needs mesmer_land_mask settled first).
    target_global_cl, target_region_cl = _reduce_to_climatology(
        target_denormed, "__target__", ocean_mask, mesmer_land_mask, start_month
    )
    del target_denormed

    for x, lbl in exp_pairs:
        if lbl == mesmer_label:
            continue  # already computed above
        raw, n_parts = _load_pooled_pred(
            cfg,
            x,
            mesmer_year,
            diffesm_year,
            num_members,
            num_rollout_steps,
            n_surface_vars,
            n_level_vars,
            n_plevs,
            train_dataset,
            load_experiment,
        )
        if raw is None:
            print(f"[clim] No starts loaded for {lbl}, dropping it entirely.")
            continue
        print(
            f"[clim] Pooled {n_parts}/{len(ALL_STARTS)} starts for {lbl} -> {raw['surface'].shape[1]} timesteps"
        )
        computed[lbl] = _reduce_to_climatology(
            raw, x, ocean_mask, mesmer_land_mask, start_month
        )
        del raw

    # Preserve original config ordering (mesmer, if present, may not be first).
    pred_global_cl, pred_region_cl = {}, {}
    for x, lbl in exp_pairs:
        if lbl not in computed:
            continue
        pred_global_cl[lbl], pred_region_cl[lbl] = computed[lbl]

    out_path = f"{cfg.paths.plots}/{out_dir}/climatology_envelopes"
    print(f"[clim] Saving to {out_path}.pdf")
    plot_climatology_envelopes(pred_global_cl, target_global_cl, out_path)
    plot_climatology_envelopes_by_region(
        pred_region_cl, target_region_cl, out_path + "_by_region"
    )


if __name__ == "__main__":
    main()
