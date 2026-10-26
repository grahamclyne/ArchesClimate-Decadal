from ArchesClimate.paths import path as artifact_path
import os

os.environ["PROJ_NETWORK"] = "OFF"

import matplotlib

matplotlib.use("Agg")  # no display on cluster

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import filefisher
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import mesmer

# =============================================================================
# Configuration — must match calibration/eval scripts
# =============================================================================

MODEL = "IPSL-CM6A-LR"
THRESHOLD_LAND = 1 / 3
REFERENCE_PERIOD = slice("1960", "1970")
REF_INIT_YEAR = 1960
EVAL_INIT_YEARS = [1969, 1979, 2010]
DCPP_DATA_PATH = Path(str(artifact_path('data') / 'dcpp_dataset/'))
PARAM_PATH = Path("./output/calibrated_parameters/")
EMU_PATH = Path("./output/emulations/")
PLOT_PATH = Path("./output/plots/")
RUN_LABEL = "dcpp"

PLOT_PATH.mkdir(exist_ok=True, parents=True)

# =============================================================================
# Helpers
# =============================================================================

def load_init_year(init_yr):
    members = []
    for fpath in sorted(DCPP_DATA_PATH.glob(f"{init_yr}_*.nc")):
        member_id = int(fpath.stem.split("_")[1])
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        ds = xr.open_dataset(str(fpath), decode_times=time_coder, engine="netcdf4")
        ds = ds.drop_vars(["height", "time_bnds", "file_qf"], errors="ignore")
        ds = ds.assign_coords({"member": member_id})
        members.append(ds[["tas"]])
    return xr.concat(members, dim="member")


def to_year_float(time_coord):
    """Convert cftime/datetime time coord to decimal year array for plotting."""
    return np.array([t.year + (t.month - 0.5) / 12 for t in time_coord.values])


def annual_mean(da, time_dim="time"):
    return da.groupby(f"{time_dim}.year").mean(time_dim)


# =============================================================================
# Load reference and grid
# =============================================================================

print("Loading reference data...", flush=True)
ref_ds = load_init_year(REF_INIT_YEAR)
ref_mean = ref_ds.sel(time=REFERENCE_PERIOD).mean("time")

PARAM_FILEFINDER = filefisher.FileFinder(
    path_pattern=PARAM_PATH / "{esm}_{scen}",
    file_pattern="params_{module}_{esm}_{scen}.nc",
)
grid_orig = xr.open_dataset(
    PARAM_FILEFINDER.create_full_name(module="grid-orig", esm=MODEL, scen=RUN_LABEL)
)

# =============================================================================
# Load and process all data
# =============================================================================

records = {}

for init_yr in EVAL_INIT_YEARS:
    key = f"init_{init_yr}"
    print(f"Loading {key}...", flush=True)

    # --- emulations (stacked gridcell) ---
    emu_ds = xr.open_dataset(EMU_PATH / f"tas_emulations_{MODEL}_{key}_{RUN_LABEL}.nc")
    emu_tas = emu_ds.tas  # (gridcell, time, realisation)

    # global land mean: weight by cos(lat)
    lat_weights = np.cos(np.deg2rad(emu_tas.lat))
    emu_globmean = emu_tas.weighted(lat_weights).mean("gridcell")  # (time, realisation)
    emu_globmean_annual = annual_mean(emu_globmean)  # (year, realisation)

    # --- actual ESM data ---
    actual_ds = load_init_year(init_yr)
    tas_anom = (actual_ds - ref_mean).tas  # (member, time, lat, lon)

    # mask ocean and antarctica, then compute land mean
    masked = mesmer.mask.mask_ocean_fraction(tas_anom.to_dataset(name="tas"), THRESHOLD_LAND)
    masked = mesmer.mask.mask_antarctica(masked)
    lat_weights_2d = np.cos(np.deg2rad(masked.lat))
    actual_globmean = masked.tas.weighted(lat_weights_2d).mean(["lat", "lon"])  # (member, time)
    actual_globmean_annual = annual_mean(actual_globmean)  # (member, year)

    # --- spatial: unstack emulation mean and actual mean ---
    emu_mean_stacked = emu_tas.mean("realisation").mean("time")  # (gridcell,)
    emu_mean_spatial = mesmer.grid.unstack_lat_lon_and_align(emu_mean_stacked, grid_orig)

    actual_mean_spatial = tas_anom.mean(["member", "time"])  # (lat, lon)

    records[init_yr] = {
        "emu_globmean_annual": emu_globmean_annual,
        "actual_globmean_annual": actual_globmean_annual,
        "emu_mean_spatial": emu_mean_spatial,
        "actual_mean_spatial": actual_mean_spatial,
        "emu_tas": emu_tas,
    }

print("Data loaded. Making plots...", flush=True)

# =============================================================================
# Figure 1: Global land mean time series
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
fig.suptitle(f"Global land mean temperature anomaly — {MODEL}", fontsize=12)

for ax, init_yr in zip(axes, EVAL_INIT_YEARS):
    r = records[init_yr]
    emu_ann = r["emu_globmean_annual"]   # (year, realisation)
    act_ann = r["actual_globmean_annual"]  # (member, year)

    years_emu = emu_ann.year.values
    years_act = act_ann.year.values

    # emulation realisations (thin, transparent)
    for i in range(emu_ann.sizes["realisation"]):
        ax.plot(years_emu, emu_ann.isel(realisation=i).values,
                color="steelblue", alpha=0.15, lw=0.6)

    # emulation spread (5–95 percentile)
    p05 = emu_ann.quantile(0.05, "realisation")
    p95 = emu_ann.quantile(0.95, "realisation")
    ax.fill_between(years_emu, p05.values, p95.values,
                    color="steelblue", alpha=0.25, label="Emulation 5–95%")

    # emulation mean
    ax.plot(years_emu, emu_ann.mean("realisation").values,
            color="steelblue", lw=2, label="Emulation mean")

    # actual ESM members (grey)
    for m in act_ann.member.values:
        ax.plot(years_act, act_ann.sel(member=m).values,
                color="0.6", lw=0.8, alpha=0.7)

    # actual ESM ensemble mean
    ax.plot(years_act, act_ann.mean("member").values,
            color="firebrick", lw=2, label="ESM ensemble mean")

    ax.axhline(0, color="0.3", lw=0.5, ls="--")
    ax.set_title(f"init_{init_yr}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Tas anomaly (K)")
    ax.legend(fontsize=7)

plt.tight_layout()
out = PLOT_PATH / f"timeseries_globmean_{MODEL}_{RUN_LABEL}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}", flush=True)
plt.close(fig)

# =============================================================================
# Figure 2: Spatial maps — ESM mean | Emulation mean | Difference
# =============================================================================

proj = ccrs.Robinson()
pc = ccrs.PlateCarree()

fig, axes = plt.subplots(
    len(EVAL_INIT_YEARS), 3,
    figsize=(15, 4 * len(EVAL_INIT_YEARS)),
    subplot_kw={"projection": proj},
)
fig.suptitle(
    f"Mean temperature anomaly over prediction window — {MODEL}", fontsize=12
)

col_titles = ["ESM ensemble mean", "Emulation mean", "Emulation − ESM"]

for col, title in enumerate(col_titles):
    axes[0, col].set_title(title, fontsize=10)

for row, init_yr in enumerate(EVAL_INIT_YEARS):
    r = records[init_yr]
    esm_map = r["actual_mean_spatial"]    # (lat, lon)
    emu_map = r["emu_mean_spatial"]        # (lat, lon)
    diff_map = emu_map - esm_map

    # shared colour scale for ESM and emulation
    vmax = float(max(abs(esm_map).max(), abs(emu_map).max()))
    opt_mean = {"transform": pc, "cmap": "RdBu_r", "vmin": -vmax, "vmax": vmax,
                    "add_colorbar": False}
    vdiff = float(abs(diff_map).quantile(0.98))
    opt_diff = {"transform": pc, "cmap": "RdBu_r", "vmin": -vdiff, "vmax": vdiff,
                    "add_colorbar": False}

    for col, (data, opt) in enumerate(
        [(esm_map, opt_mean), (emu_map, opt_mean), (diff_map, opt_diff)]
    ):
        ax = axes[row, col]
        h = data.plot(ax=ax, **opt)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
        ax.set_global()
        ax.set_ylabel(f"init_{init_yr}" if col == 0 else "")

        # colorbar
        plt.colorbar(h, ax=ax, orientation="horizontal", pad=0.04,
                     fraction=0.046, label="K")

plt.tight_layout()
out = PLOT_PATH / f"spatial_maps_{MODEL}_{RUN_LABEL}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}", flush=True)
plt.close(fig)

# =============================================================================
# Figure 3: Emulation spread (std across realisations) per init year
# =============================================================================

fig, axes = plt.subplots(
    1, len(EVAL_INIT_YEARS),
    figsize=(15, 4),
    subplot_kw={"projection": proj},
)
fig.suptitle(f"Emulation spread (std across realisations) — {MODEL}", fontsize=12)

for ax, init_yr in zip(axes, EVAL_INIT_YEARS):
    r = records[init_yr]
    emu_tas = r["emu_tas"]  # (gridcell, time, realisation)

    std_stacked = emu_tas.std("realisation").mean("time")  # (gridcell,)
    std_spatial = mesmer.grid.unstack_lat_lon_and_align(std_stacked, grid_orig)

    vmax = float(std_spatial.quantile(0.98))
    h = std_spatial.plot(
        ax=ax, transform=pc, cmap="YlOrRd", vmin=0, vmax=vmax,
        add_colorbar=False,
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    ax.set_global()
    ax.set_title(f"init_{init_yr}", fontsize=10)
    plt.colorbar(h, ax=ax, orientation="horizontal", pad=0.04,
                 fraction=0.046, label="K")

plt.tight_layout()
out = PLOT_PATH / f"emulation_spread_{MODEL}_{RUN_LABEL}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}", flush=True)
plt.close(fig)

print("\nAll plots saved to", PLOT_PATH, flush=True)
