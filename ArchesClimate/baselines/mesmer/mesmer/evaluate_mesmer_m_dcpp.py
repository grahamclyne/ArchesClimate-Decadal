from ArchesClimate.paths import path as artifact_path
import os

os.environ["PROJ_NETWORK"] = "OFF"

import pathlib
from pathlib import Path

import filefisher
import numpy as np
import xarray as xr

import mesmer

# =============================================================================
# Configuration — must match calibration settings
# =============================================================================

MODEL = "IPSL-CM6A-LR"
THRESHOLD_LAND = 1 / 3
REFERENCE_PERIOD = slice("1960", "1970")
REF_INIT_YEAR = 1960
EVAL_INIT_YEARS = [1969, 1979, 1989, 1999, 2009, 2010, 2011, 2012, 2013, 2014, 2015]
N_REALISATIONS = 30
BUFFER = 20

DCPP_DATA_PATH = Path(str(artifact_path('data') / 'dcpp_dataset/'))
PARAM_PATH = pathlib.Path("./output/calibrated_parameters/")
EMU_PATH = pathlib.Path("./output/emulations/")
RUN_LABEL = "dcpp_m"

# Fixed seeds for reproducibility — one per eval init year
SEEDS = {1969: 100, 1979: 200, 1989: 300, 1999: 400, 2009: 500,
         2010: 600, 2011: 700, 2012: 800, 2013: 900, 2014: 1000, 2015: 1100}

# =============================================================================
# Helper functions
# =============================================================================


def load_init_year(init_yr):
    """Load all members for one init year as (member, time, lat, lon) Dataset."""
    members = []
    for fpath in sorted(DCPP_DATA_PATH.glob(f"{init_yr}_*.nc")):
        member_id = int(fpath.stem.split("_")[1])
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        ds = xr.open_dataset(str(fpath), decode_times=time_coder, engine="netcdf4")
        ds = ds.drop_vars(["height", "time_bnds", "file_qf"], errors="ignore")
        ds = ds.assign_coords({"member": member_id})
        members.append(ds[["tas"]])
    if not members:
        raise FileNotFoundError(f"No files found for init year {init_yr}")
    return xr.concat(members, dim="member")


def clip_to_complete_years(ds):
    """Keep only months from years that have all 12 months present."""
    years = ds.time.dt.year.values
    unique_years, counts = np.unique(years, return_counts=True)
    complete = unique_years[counts == 12]
    mask = np.isin(years, complete)
    return ds.isel(time=mask)


def mask_and_stack_ds(ds, threshold_land):
    """Mask ocean/Antarctica and stack lat/lon into gridcell for a single Dataset."""
    dt = xr.DataTree.from_dict({"node": ds})
    dt = mesmer.mask.mask_ocean_fraction(dt, threshold_land)
    dt = mesmer.mask.mask_antarctica(dt)
    dt = mesmer.grid.stack_lat_lon(dt)
    return dt["node"].ds


# =============================================================================
# Load reference data (init_1960) for anomaly baseline
# =============================================================================

print("Loading reference data (init_1960)...", flush=True)
ref_raw = load_init_year(REF_INIT_YEAR)
ref_raw = clip_to_complete_years(ref_raw)
ref_raw_annual = ref_raw.resample(time="YE").mean()

# Per-member reference means used to compute anomalies (same as calibration)
ref_mean_m = ref_raw.sel(time=REFERENCE_PERIOD).mean("time")         # (member, lat, lon)
ref_mean_y = ref_raw_annual.sel(time=REFERENCE_PERIOD).mean("time")  # (member, lat, lon)
print(f"  Reference mean computed from {REF_INIT_YEAR} over {REFERENCE_PERIOD}", flush=True)

# =============================================================================
# Load calibrated parameters
# =============================================================================

print("Loading calibrated parameters...", flush=True)
PARAM_FILEFINDER = filefisher.FileFinder(
    path_pattern=PARAM_PATH / "{esm}_{scen}",
    file_pattern="params_{module}_{esm}_{scen}.nc",
)

harmonic_params = xr.open_dataset(
    PARAM_FILEFINDER.create_full_name(module="harmonic-model", esm=MODEL, scen=RUN_LABEL)
)
pt_ds = xr.open_dataset(
    PARAM_FILEFINDER.create_full_name(module="power-transformer", esm=MODEL, scen=RUN_LABEL)
)
ar1_params = xr.open_dataset(
    PARAM_FILEFINDER.create_full_name(module="local-ar1-monthly", esm=MODEL, scen=RUN_LABEL)
)
cov_params = xr.open_dataset(
    PARAM_FILEFINDER.create_full_name(module="covariance-monthly", esm=MODEL, scen=RUN_LABEL)
)
grid_orig = xr.open_dataset(
    PARAM_FILEFINDER.create_full_name(module="grid-orig", esm=MODEL, scen=RUN_LABEL)
)

lambda_function = pt_ds.lambda_coeffs.attrs.get("lambda_function", "constant")
yj_transformer = mesmer.stats.YeoJohnsonTransformer(lambda_function)
pt_coefficients = pt_ds.lambda_coeffs
print("  All parameters loaded.", flush=True)

# =============================================================================
# Generate emulations for each held-out init year
# =============================================================================

EMU_PATH.mkdir(exist_ok=True, parents=True)

for init_yr in EVAL_INIT_YEARS:
    key = f"init_{init_yr}"
    print(f"\nGenerating emulations for {key}...", flush=True)

    # --- Load and prepare anomalies ---
    eval_raw = load_init_year(init_yr)
    eval_raw = clip_to_complete_years(eval_raw)
    eval_raw_annual = eval_raw.resample(time="YE").mean()

    tas_anom_m = eval_raw - ref_mean_m
    tas_anom_y = eval_raw_annual - ref_mean_y

    # Mask and stack
    tas_m_stacked = mask_and_stack_ds(tas_anom_m, THRESHOLD_LAND)
    tas_y_stacked = mask_and_stack_ds(tas_anom_y, THRESHOLD_LAND)

    # --- Predictor and time axis ---
    # Ensemble mean annual anomaly as yearly predictor: (time, gridcell)
    yearly_predictor = tas_y_stacked.tas.mean("member")
    m_time = tas_m_stacked.time

    print(
        f"  Annual steps: {yearly_predictor.sizes['time']}, "
        f"monthly steps: {m_time.size}",
        flush=True,
    )

    # --- 1. Harmonic model (seasonal cycle) ---
    print("  Predicting harmonic model...", flush=True)
    monthly_harmonic_emu = mesmer.stats.predict_harmonic_model(
        yearly_predictor, harmonic_params.coeffs, m_time
    )

    # --- 2. Draw local variability with cyclo-stationary AR(1) ---
    print("  Drawing local variability...", flush=True)
    local_variability_transformed = mesmer.stats.draw_auto_regression_monthly(
        ar1_params,
        cov_params.localized_covariance,
        time=m_time,
        n_realisations=N_REALISATIONS,
        seed=SEEDS[init_yr],
        buffer=BUFFER,
    )

    # --- 3. Invert Yeo-Johnson power transformation ---
    print("  Inverting power transformation...", flush=True)
    local_variability_inverted = yj_transformer.inverse_transform(
        yearly_predictor,
        local_variability_transformed.samples,
        pt_coefficients,
    )

    # --- 4. Combine seasonal cycle + variability ---
    emulations = monthly_harmonic_emu + local_variability_inverted.inverted
    # shape: (realisation, time, gridcell) — monthly anomaly emulations

    # --- 5. Save ---
    emu_ds = emulations.to_dataset(name="tas")
    emu_ds["seed"] = SEEDS[init_yr]
    emu_ds["reference_period_start"] = str(REFERENCE_PERIOD.start)
    emu_ds["reference_period_stop"] = str(REFERENCE_PERIOD.stop)

    out_path = EMU_PATH / f"tas_monthly_emulations_{MODEL}_{key}_{RUN_LABEL}.nc"
    emu_ds.to_netcdf(out_path)
    print(f"  Saved {out_path}", flush=True)

print("\nDone!", flush=True)
