"""
Combined regional analysis script: regional means, North Atlantic seasonal maps,
and Pearson correlation maps. Driven by hydra config + CLI overrides.

Usage:
    python analysis/regional_analysis.py \
        module=flow_ensemble \
        analysis.experiment=flow_ensemble_2_24_10_1_118_1 \
        analysis.test_0_year=1969 \
        analysis.test_10_year=1979 \
        analysis.val_year=2010
"""

import datetime
import os

import cartopy.crs as ccrs
import hydra
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from numpy import ma
from omegaconf import DictConfig

from ArchesClimate.analysis.gen_and_save_metrics import (
    load_climatology,
    load_experiment,
)
from ArchesClimate.paths import path as artifact_path


def compute_self_climatology(data, key):
    """Compute monthly climatology from data[key] itself.

    data[key] shape: [members, T, vars, spatial...]
    Returns tiled climatology of shape [T, vars, spatial...].
    """
    tensor = data[key]  # [M, T, ...]
    T = tensor.shape[1]
    month_indices = torch.arange(T) % 12
    clima = torch.zeros(12, *tensor.shape[2:])
    for m in range(12):
        mask = month_indices == m
        clima[m] = tensor[:, mask].nanmean(dim=(0, 1))
    return clima[month_indices]  # [T, ...]


# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------

plt.style.use("seaborn-v0_8-paper")
mpl.rcParams["font.serif"] = ["Times New Roman"]
plt.rc("font", size=12)
plt.rc("axes", titlesize=13, labelsize=12)
plt.rc("xtick", labelsize=12)
plt.rc("ytick", labelsize=12)
plt.rc("legend", fontsize=12)
plt.rc("figure", titlesize=14)
plt.rc("lines", linewidth=1.1)
plt.rcParams["font.family"] = ["sans"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_monthly_dates(start_year, start_month, num_months):
    dates, year, month = [], start_year, start_month
    for _ in range(num_months):
        dates.append(datetime.datetime(year, month, 1))  # noqa: DTZ001 -- timezone-naive climate calendar
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return dates


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    return mcolors.LinearSegmentedColormap.from_list(
        f"{cmap.name}_trunc", cmap(np.linspace(minval, maxval, n))
    )


def pearson_r(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x * y).sum(axis=0) / (
        torch.sqrt((x**2).sum(axis=0)) * torch.sqrt((y**2).sum(axis=0))
    )


def load_target(cfg, path, train_spatial_norm, num_rollout_steps):
    data = torch.load(path)
    data["surface"] = data["surface"][:, :, cfg.module.surface_variable_indices]
    data["level"] = data["level"][:, :, cfg.module.level_variable_indices]
    return train_spatial_norm.denormalize(data, stateless=True)[:10, :num_rollout_steps]


# ---------------------------------------------------------------------------
# Plot 1: regional ensemble mean rollouts
# ---------------------------------------------------------------------------


def plot_headline_ensemble_mean_rollouts(
    generated_data,
    ipsl_data,
    file_name,
    var_names,
    start_year,
    start_lat,
    end_lat,
    start_lon,
    end_lon,
    add_legend=True,
):

    x = generate_monthly_dates(start_year, 1, generated_data["surface"].shape[2])
    surface_indices = [1, 6, -1]
    level_index, plev = 2, 1
    var_ylabels = ["W m-2", "Pa", "degC", "K"]

    fig, axes = plt.subplots(len(surface_indices) + 1, sharex=True)
    axes = axes.flatten()

    for idx, vi in enumerate(surface_indices):
        agg = generated_data["surface"][
            0, :, :, vi, 0, start_lat:end_lat, start_lon:end_lon
        ].nanmean(axis=(-1, -2))
        means, std = agg.mean(axis=0), agg.std(axis=0)
        axes[idx].plot(x, means, color="blue", lw=0.5)
        axes[idx].fill_between(
            x, means - std, means + std, color="blue", alpha=0.3, label="ArchesClimate"
        )
        axes[idx].plot(
            x, agg.min(axis=0).values, color="blue", lw=0.5, linestyle="dotted"
        )
        axes[idx].plot(
            x, agg.max(axis=0).values, color="blue", lw=0.5, linestyle="dotted"
        )

        agg_ipsl = ipsl_data["surface"][
            0, :, :, vi, 0, start_lat:end_lat, start_lon:end_lon
        ].nanmean(axis=(-1, -2))
        means_i, std_i = agg_ipsl.mean(axis=0), agg_ipsl.std(axis=0)
        axes[idx].plot(x, means_i, color="black", lw=0.5)
        axes[idx].fill_between(
            x,
            means_i - std_i,
            means_i + std_i,
            color="black",
            alpha=0.2,
            label="IPSL-DCPP",
        )
        axes[idx].plot(
            x, agg_ipsl.min(axis=0).values, color="black", lw=0.5, linestyle="dotted"
        )
        axes[idx].plot(
            x, agg_ipsl.max(axis=0).values, color="black", lw=0.5, linestyle="dotted"
        )

        if idx == 0 and add_legend:
            axes[idx].legend(
                handles=[
                    Line2D([], [], color="blue", label="ArchesClimate mean"),
                    Line2D([], [], color="black", label="IPSL-DCPP mean"),
                    Line2D(
                        [], [], color="gray", linestyle="dotted", label="Member min/max"
                    ),
                    Patch(facecolor="gray", alpha=0.3, label="Mean ±1 member SD"),
                ],
                loc="lower center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=2,
                fontsize=9,
                frameon=False,
            )
        axes[idx].set_title(var_names[idx], pad=0.1, fontsize=10)
        axes[idx].set_ylabel(var_ylabels[idx])

    # level variable
    ax_lv = axes[len(surface_indices)]
    for data, color, label in [
        (generated_data, "blue", "ArchesClimate"),
        (ipsl_data, "black", "IPSL-DCPP"),
    ]:
        agg = data["level"][
            0, :, :, level_index, plev, start_lat:end_lat, start_lon:end_lon
        ].nanmean(axis=(-1, -2))
        means = agg.mean(axis=0)
        std = agg.std(axis=0)
        ax_lv.plot(x, means, color=color, lw=0.5)
        ax_lv.fill_between(
            x, means - std, means + std, color=color, alpha=0.3, label=label
        )
        ax_lv.plot(x, agg.min(axis=0).values, color=color, lw=0.5, linestyle="dotted")
        ax_lv.plot(x, agg.max(axis=0).values, color=color, lw=0.5, linestyle="dotted")
    ax_lv.set_title(var_names[len(surface_indices)] + " 700hPa", pad=0.1, fontsize=10)
    ax_lv.set_ylabel(var_ylabels[len(surface_indices)])
    ax_lv.set_xlabel("Time")
    ax_lv.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-%m"))
    ax_lv.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonth=[1, 7]))
    plt.setp(ax_lv.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    for ax in axes[:-1]:
        ax.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )

    fig.set_size_inches((7, 5))
    fig.savefig(f"{file_name}.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: North Atlantic seasonal maps
# ---------------------------------------------------------------------------


def plot_robinson_global(ax, data, vmin, vmax):
    lat = np.linspace(-90, 90, 143)
    lon = np.linspace(-2, 358, 144)
    lon2d, lat2d = np.meshgrid(lon, lat)
    levels = np.linspace(vmin, vmax, 10)
    contour = ax.contourf(
        lon2d,
        lat2d,
        ma.masked_invalid(data),
        levels=levels,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        extend="both",
    )
    ax.coastlines()
    return contour


def plot_global_seasonal_means(pred_data, target_data, out_path):
    """Plot global seasonal std dev ratio (pred / target) per grid cell.

    For each season, computes the std dev across ensemble members at each grid point
    for pred and target separately, then plots the ratio pred_std / target_std.
    Values > 1 mean the model is overdispersed; < 1 means underdispersed.
    """
    length = pred_data["surface"].shape[1]
    start_month = 3
    seasonal_indices = [
        torch.tensor([i for i in range(length) if (i + start_month - 1) % 12 + 1 in s])
        for s in [[3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 1, 2]]
    ]
    season_names = ["MAM", "JJA", "SON", "DJF"]

    all_data = []
    for si in seasonal_indices:
        pred_std = pred_data["surface"][:, si, -1, 0].std(dim=(0, 1))  # [lat, lon]
        tgt_std = target_data["surface"][:, si, -1, 0].std(dim=(0, 1))  # [lat, lon]
        ratio = pred_std / tgt_std.clamp(min=1e-8)
        all_data.append(ratio)

    vmin = np.nanpercentile([d.numpy() for d in all_data], 1)
    vmax = np.nanpercentile([d.numpy() for d in all_data], 99)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 9),
        subplot_kw={"projection": ccrs.Robinson(central_longitude=210)},
        constrained_layout=True,
    )
    axes = axes.flatten()
    im_last = None
    for row, (ratio, si) in enumerate(zip(all_data, seasonal_indices)):
        ax = axes[row]
        im = plot_robinson_global(ax, ratio.numpy(), vmin, vmax)
        ax.set_title(f"{season_names[row]} std ratio (pred / IPSL)", fontsize=12)
        im_last = im

    cbar = fig.colorbar(
        im_last, ax=axes.tolist(), orientation="vertical", fraction=0.025, pad=0.02
    )
    cbar.set_label("Std dev ratio (pred / IPSL)", fontsize=12)
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Per-member climatology
# ---------------------------------------------------------------------------


def plot_member_climatologies(
    pred_data, target_data, out_path, surface_var_names, level_var_name
):
    """Plot 12-month climatology envelope (mean ± std across members) for pred and target."""
    surface_indices = [1, 6, -1, 10]
    level_index, plev = 2, 1
    pred_color = "#0173b2"
    tgt_color = "#de8f05"
    month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    x = list(range(1, 13))

    T = pred_data["surface"].shape[1]
    month_idx = torch.arange(T) % 12

    def member_clima(tensor):
        """tensor: [members, T, ...spatial] → spatial nanmean → [members, 12]"""
        spatial = tensor.flatten(start_dim=2).nanmean(dim=-1)  # [M, T]
        return torch.stack(
            [spatial[:, month_idx == m].mean(1) for m in range(12)], dim=1
        )

    def extract_surface(data, vi):
        """Return [M, T, lat, lon], handling both [M,T,vars,lat,lon] and [M,T,vars,1,lat,lon]."""
        t = data["surface"][:5, :, vi]  # [M, T, lat, lon] or [M, T, 1, lat, lon]
        if t.ndim == 4 and t.shape[2] == 1:
            t = t[:, :, 0]  # drop singleton → [M, T, lat, lon]
        return t

    def draw_envelope(ax, cl, color, label):
        mean = cl.mean(0).numpy()
        std = cl.std(0).numpy()
        ax.plot(x, mean, color=color, linewidth=1.5, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.25)

    n_plots = len(surface_indices) + 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(7, 2.5 * n_plots), sharex=True)

    print(f"[clim debug] pred surface shape: {pred_data['surface'].shape}")
    print(f"[clim debug] tgt  surface shape: {target_data['surface'].shape}")
    for ax_idx, vi in enumerate(surface_indices):
        ax = axes[ax_idx]
        pred_slice = extract_surface(pred_data, vi)
        tgt_slice = extract_surface(target_data, vi)
        print(
            f"[clim debug] vi={vi} pred_slice shape={pred_slice.shape} mean={pred_slice.nanmean().item():.4f}"
        )
        print(
            f"[clim debug] vi={vi} tgt_slice  shape={tgt_slice.shape}  mean={tgt_slice.nanmean().item():.4f}"
        )
        pred_cl = member_clima(pred_slice)
        tgt_cl = member_clima(tgt_slice)
        print(f"[clim debug] vi={vi} pred_cl monthly means: {pred_cl.mean(0).tolist()}")
        print(f"[clim debug] vi={vi} tgt_cl  monthly means: {tgt_cl.mean(0).tolist()}")
        draw_envelope(ax, pred_cl, pred_color, "ArchesClimate")
        draw_envelope(ax, tgt_cl, tgt_color, "IPSL")
        ax.set_title(surface_var_names[ax_idx], fontsize=10)
        ax.set_ylabel("Value")

    # level variable
    ax = axes[-1]
    pred_cl = member_clima(pred_data["level"][:5, :, level_index, plev])
    tgt_cl = member_clima(target_data["level"][:5, :, level_index, plev])
    draw_envelope(ax, pred_cl, pred_color, "ArchesClimate")
    draw_envelope(ax, tgt_cl, tgt_color, "IPSL")
    ax.set_title(f"{level_var_name} 700hPa", fontsize=10)
    ax.set_ylabel("Value")
    ax.set_xticks(x)
    ax.set_xticklabels(month_labels)
    ax.set_xlabel("Month")

    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        fontsize=10,
        frameon=False,
    )
    plt.subplots_adjust(hspace=0.4)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: Pearson correlation maps
# ---------------------------------------------------------------------------


def plot_robinson_pearson(ax, data):
    lat = np.linspace(-90, 90, 143)
    lon = np.linspace(-2, 358, 144)
    lon2d, lat2d = np.meshgrid(lon, lat)
    levels = np.linspace(-1, 1, 10)
    c = ax.contourf(
        lon2d,
        lat2d,
        ma.masked_invalid(data.numpy()),
        levels=levels,
        transform=ccrs.PlateCarree(),
        cmap=plt.get_cmap("RdBu_r"),
        extend="both",
    )
    ax.coastlines()
    return c


def plot_pearson_maps(
    pred_test0,
    pred_test10,
    pred_val,
    target_test0,
    target_test10,
    target_val,
    clima,
    out_path,
    year_test0=1969,
    year_test10=1979,
    year_val=2010,
):
    def anomaly(data, clim_slice):
        return data["surface"][:, :118] - clim_slice

    clim = clima["surface"]
    pred_ano_0 = anomaly(pred_test0, clim[:, :5])
    pred_ano_val = anomaly(pred_val, clim[:, :5])
    tgt_ano_0 = anomaly(target_test0, clim)
    tgt_ano_val = anomaly(target_val, clim)

    vi = -1  # SST

    def pr(pred_ano, tgt_ano):
        return pearson_r(
            pred_ano[0, :5, :117, vi, 0].mean(0), tgt_ano[0, 5:, :117, vi, 0].mean(0)
        )

    maps = [
        (
            pr(pred_ano_0, tgt_ano_0),
            f"ArchesClimate vs IPSL {year_test0}-{year_test0 + 10}",
        ),
        (pr(tgt_ano_0, tgt_ano_0), f"IPSL vs IPSL {year_test0}-{year_test0 + 10}"),
        (
            pr(pred_ano_val, tgt_ano_val),
            f"ArchesClimate vs IPSL {year_val}-{year_val + 10}",
        ),
        (pr(tgt_ano_val, tgt_ano_val), f"IPSL vs IPSL {year_val}-{year_val + 10}"),
    ]

    fig, ax = plt.subplots(
        2,
        2,
        figsize=(14, 7),
        subplot_kw={"projection": ccrs.Robinson(central_longitude=210)},
        constrained_layout=True,
    )
    ax = ax.flatten()
    c_last = None
    for i, (rmap, title) in enumerate(maps):
        c = plot_robinson_pearson(ax[i], rmap)
        ax[i].set_title(title, fontsize=11)
        c_last = c

    for i in [2, 3]:
        cbar = plt.colorbar(
            c_last, ax=ax[i], orientation="horizontal", shrink=0.8, pad=0.05
        )
        cbar.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        cbar.ax.tick_params(labelsize=10)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    num_rollout_steps = cfg.inference.num_rollout_steps
    num_members = cfg.analysis.num_members
    experiment_base = cfg.analysis.experiment
    ckpt = getattr(cfg.analysis, "ckpt_fname", "None")

    def _exp(target, year):
        return (
            f"{experiment_base}_test_{year}_{ckpt}"
            if target == "test"
            else f"{experiment_base}_{target}_{year}_{ckpt}"
        )

    out_dir = f"{cfg.paths.plots}/{experiment_base}"
    os.makedirs(out_dir, exist_ok=True)

    train = hydra.utils.instantiate(cfg.dataloader.dataset, domain="test")
    train_spatial_norm = hydra.utils.instantiate(
        cfg.dataloader.dataset, domain="test", norm_scheme="spatial_norm"
    )

    # --- Load targets ---
    generated_data_path = str(artifact_path("targets"))
    target_test_0 = load_target(
        cfg,
        f"{generated_data_path}/test_ensemble_0_flow_net_flux_1",
        train_spatial_norm,
        num_rollout_steps,
    )
    target_test_10 = load_target(
        cfg,
        f"{generated_data_path}/test_ensemble_10_flow_net_flux_1",
        train_spatial_norm,
        num_rollout_steps,
    )
    target_val_0 = load_target(
        cfg,
        f"{generated_data_path}/val_ensemble_0_flow_net_flux_1",
        train_spatial_norm,
        num_rollout_steps,
    )

    # --- Load predictions ---
    first_test = train.denormalize(
        load_experiment(cfg, _exp("test", 0), num_members, num_rollout_steps),
        stateless=True,
    )
    second_test = train.denormalize(
        load_experiment(cfg, _exp("test", 10), num_members, num_rollout_steps),
        stateless=True,
    )
    first_val = train.denormalize(
        load_experiment(cfg, _exp("val", 0), num_members, num_rollout_steps),
        stateless=True,
    )

    # --- Headline var names ---
    # NB: train.level_variables is the dataset's expanded (var x plev) list (e.g.
    # 'hur 500' at index 2), while plot_headline_ensemble_mean_rollouts's hardcoded
    # level_index=2 refers to the compact per-variable list (module.level_variables,
    # index 2 = 'ta') and separately appends its own '700hPa' suffix -- use the
    # compact list here to match, or the label and the plotted data disagree.
    surface_var_names = [train.surface_variables[i] for i in [1, 6, -1]]
    level_var_name = cfg.module.level_variables[2]
    headline_var_names = surface_var_names + [level_var_name]

    year_test0 = getattr(cfg.analysis, "test_0_year", 1969)
    year_test10 = getattr(cfg.analysis, "test_10_year", 1979)
    year_val = getattr(cfg.analysis, "val_year", 2010)

    # --- Per-dataset self climatologies for anomaly plots ---
    pred_clima_surf = compute_self_climatology(first_test[:5], "surface")
    pred_clima_level = compute_self_climatology(first_test[:5], "level")
    tgt_clima_surf = compute_self_climatology(target_test_0[:5], "surface")
    tgt_clima_level = compute_self_climatology(target_test_0[:5], "level")

    from tensordict.tensordict import TensorDict as TD

    pred_ano = TD(
        {
            "surface": first_test[:5]["surface"] - pred_clima_surf[None],
            "level": first_test[:5]["level"] - pred_clima_level[None],
        }
    )
    tgt_ano = TD(
        {
            "surface": target_test_0[:5]["surface"] - tgt_clima_surf[None],
            "level": target_test_0[:5]["level"] - tgt_clima_level[None],
        }
    )

    # --- Regional means: tropics ---
    print("Plotting tropics rollouts...")
    plot_headline_ensemble_mean_rollouts(
        first_test[None, :5],
        target_test_0[None, :5],
        f"{out_dir}/headline_var_rollout_tropics",
        headline_var_names,
        year_test0,
        56,
        86,
        0,
        144,
        add_legend=True,
    )
    plot_headline_ensemble_mean_rollouts(
        pred_ano[None],
        tgt_ano[None],
        f"{out_dir}/headline_var_rollout_tropics_anomalies",
        headline_var_names,
        year_test0,
        56,
        86,
        0,
        144,
        add_legend=False,
    )

    # --- Regional means: North Atlantic ---
    print("Plotting North Atlantic rollouts...")
    plot_headline_ensemble_mean_rollouts(
        first_test[None, :5],
        target_test_0[None, :5],
        f"{out_dir}/headline_var_rollout_north_atlantic",
        headline_var_names,
        year_test0,
        71,
        122,
        112,
        144,
        add_legend=True,
    )
    plot_headline_ensemble_mean_rollouts(
        pred_ano[None],
        tgt_ano[None],
        f"{out_dir}/headline_var_rollout_north_atlantic_anomalies",
        headline_var_names,
        year_test0,
        71,
        122,
        112,
        144,
        add_legend=False,
    )

    # --- Per-member climatologies (test + val concatenated along time) ---
    print("Plotting member climatologies...")
    pred_all = TD(
        {
            "surface": torch.cat(
                [first_test[:5]["surface"], first_val[:5]["surface"]], dim=1
            ),
            "level": torch.cat(
                [first_test[:5]["level"], first_val[:5]["level"]], dim=1
            ),
        }
    )
    tgt_all = TD(
        {
            "surface": torch.cat(
                [target_test_0[:5]["surface"], target_val_0[:5]["surface"]], dim=1
            ),
            "level": torch.cat(
                [target_test_0[:5]["level"], target_val_0[:5]["level"]], dim=1
            ),
        }
    )
    plot_member_climatologies(
        pred_all,
        tgt_all,
        f"{out_dir}/member_climatologies.pdf",
        surface_var_names=[train.surface_variables[i] for i in [1, 6, -1, 10]],
        level_var_name=train.level_variables[2],
    )

    # --- Global seasonal maps ---
    print("Plotting global seasonal maps...")
    plot_global_seasonal_means(
        first_test, target_test_0, f"{out_dir}/global_seasonal_means.pdf"
    )

    # --- Pearson correlation maps ---
    print("Plotting Pearson maps...")
    clima = load_climatology(
        cfg, min(num_members, 10), num_rollout_steps, target_test_0, 0, None
    )
    plot_pearson_maps(
        first_test,
        second_test,
        first_val,
        target_test_0,
        target_test_10,
        target_val_0,
        clima,
        f"{out_dir}/sst_pearsons.pdf",
        year_test0=year_test0,
        year_test10=year_test10,
        year_val=year_val,
    )

    print(f"All figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
