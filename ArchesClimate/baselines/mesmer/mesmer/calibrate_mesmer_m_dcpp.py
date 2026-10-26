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
# Configuration
# =============================================================================

MODEL = "IPSL-CM6A-LR"
THRESHOLD_LAND = 1 / 3
REFERENCE_PERIOD = slice("1960", "1970")
REF_SCENARIO = "init_1960"
LOCALISATION_RADII = list(range(7_500, 12_501, 500))
K_FOLDS = 10

DCPP_DATA_PATH = Path(str(artifact_path('data') / 'dcpp_dataset/'))
PARAM_PATH = pathlib.Path("./output/calibrated_parameters/")
RUN_LABEL = "dcpp_m"

EXCLUDED_INIT_YEARS = [1969, 1979, 1989, 2010]
MAX_INIT_YEAR = 2010

# =============================================================================
# Load data
# =============================================================================

print("Loading data...", flush=True)

DCPP_FILEFINDER = filefisher.FileFinder(
    path_pattern=DCPP_DATA_PATH,
    file_pattern="{init_year}_{member}.nc",
)

fc_all = DCPP_FILEFINDER.find_files()

df = fc_all.df.copy()
df["init_year"] = df["init_year"].astype(int)
df["member"] = df["member"].astype(int)

training_mask = (~df["init_year"].isin(EXCLUDED_INIT_YEARS)) & (
    df["init_year"] <= MAX_INIT_YEAR
)
fc_all.df = (
    df[training_mask]
    .sort_values(by=["init_year", "member"])
    .reset_index(drop=True)
)

print(f"  {len(fc_all.df)} files selected for training.", flush=True)


def load_data(filecontainer):
    out = xr.DataTree()
    init_years = filecontainer.df.init_year.unique().tolist()

    for init_yr in init_years:
        year_df = filecontainer.df[filecontainer.df["init_year"] == init_yr]
        if year_df.empty:
            continue

        members = []
        for _, row in year_df.iterrows():
            file_path = (
                DCPP_DATA_PATH / f"{int(row['init_year'])}_{int(row['member'])}.nc"
            )
            if not file_path.exists():
                print(f"  Warning: missing {file_path}, skipping.", flush=True)
                continue

            time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
            ds = xr.open_dataset(
                str(file_path), decode_times=time_coder, engine="netcdf4"
            )
            ds = ds.drop_vars(["height", "time_bnds", "file_qf"], errors="ignore")
            ds = ds.assign_coords({"member": int(row["member"])})
            members.append(ds[["tas"]])

        if members:
            out[f"init_{init_yr}"] = xr.DataTree(xr.concat(members, dim="member"))

    return out


dt_monthly = load_data(fc_all)
print(f"  DataTree loaded with {len(dt_monthly.children)} init years.", flush=True)

# Use only the first two member per init year to reduce memory and compute cost.
# MESMER-M fits are pooled across all init years, so one member per year still
# provides a wide range of climate states for the harmonic/AR/covariance fits.
dt_monthly = mesmer.datatree.map_over_datasets(
    lambda ds: ds.isel(member=slice(0, 2)), dt_monthly
)
print("  Reduced to one member per init year.", flush=True)

# =============================================================================
# Clip to complete calendar years
# =============================================================================

print("Clipping to complete calendar years...", flush=True)


def clip_to_complete_years(ds):
    """Keep only months from years that have all 12 months present."""
    years = ds.time.dt.year.values
    unique_years, counts = np.unique(years, return_counts=True)
    complete = unique_years[counts == 12]
    mask = np.isin(years, complete)
    return ds.isel(time=mask)


dt_monthly = mesmer.datatree.map_over_datasets(clip_to_complete_years, dt_monthly)

# =============================================================================
# Annual means from monthly
# =============================================================================

print("Computing annual means from monthly data...", flush=True)
dt_annual = mesmer.datatree.map_over_datasets(
    lambda ds: ds.resample(time="YE").mean(), dt_monthly
)

# =============================================================================
# Anomalies
# =============================================================================

print("Calculating anomalies...", flush=True)
tas_anoms_m = mesmer.anomaly.calc_anomaly(
    dt_monthly, reference_period=REFERENCE_PERIOD, ref_scenario=REF_SCENARIO
)
tas_anoms_y = mesmer.anomaly.calc_anomaly(
    dt_annual, reference_period=REFERENCE_PERIOD, ref_scenario=REF_SCENARIO
)

# =============================================================================
# Mask, stack, pool
# =============================================================================

print("Masking and stacking...", flush=True)
grid_orig = tas_anoms_y[REF_SCENARIO].ds[["lat", "lon"]]


def mask_and_stack(dt, threshold_land):
    dt = mesmer.mask.mask_ocean_fraction(dt, threshold_land)
    dt = mesmer.mask.mask_antarctica(dt)
    dt = mesmer.grid.stack_lat_lon(dt)
    return dt


tas_y = mask_and_stack(tas_anoms_y, THRESHOLD_LAND)
tas_m = mask_and_stack(tas_anoms_m, THRESHOLD_LAND)

print("Pooling...", flush=True)
tas_pooled_y = mesmer.datatree.pool_scen_ens(tas_y)
tas_pooled_m = mesmer.datatree.pool_scen_ens(tas_m)

# =============================================================================
# Fit harmonic model
# =============================================================================

print("Fitting harmonic model...", flush=True)
harmonic_model_fit = mesmer.stats.fit_harmonic_model(
    tas_pooled_y.tas, tas_pooled_m.tas
)
print("  Harmonic model done.", flush=True)

# =============================================================================
# Yeo-Johnson power transformer
# =============================================================================

print("Fitting Yeo-Johnson power transformer...", flush=True)
yj_transformer = mesmer.stats.YeoJohnsonTransformer("constant")
pt_coefficients = yj_transformer.fit(
    tas_pooled_y.tas, harmonic_model_fit.residuals
)
transformed_resids = yj_transformer.transform(
    tas_pooled_y.tas, harmonic_model_fit.residuals, pt_coefficients
)
print("  Power transformer done.", flush=True)

# =============================================================================
# Cyclo-stationary AR(1)
# =============================================================================

print("Fitting cyclo-stationary AR(1) — this can take several minutes...", flush=True)
ar1_fit = mesmer.stats.fit_auto_regression_monthly(transformed_resids.transformed)
print("  AR(1) done.", flush=True)

# =============================================================================
# Localized empirical covariance (monthly)
# =============================================================================

print("Computing geodistance matrix...", flush=True)
first_key = next(iter(tas_y.children.keys()))
grid_stacked = tas_y[first_key].ds[["lat", "lon"]]
geodist = mesmer.geospatial.geodist_exact(grid_stacked.lon, grid_stacked.lat)

print("Building Gaspari-Cohn localizers...", flush=True)
phi_gc_localizer = mesmer.stats.gaspari_cohn_correlation_matrices(
    geodist, localisation_radii=LOCALISATION_RADII
)

print("Computing weights...", flush=True)
weights = mesmer.weighted.equal_scenario_weights_from_datatree(tas_anoms_m)
weights = mesmer.datatree.pool_scen_ens(weights)
# AR(1) drops the first timestep; match that here
weights = weights.isel(sample=slice(1, None))

print(
    f"Finding localized empirical covariance (k_folds={K_FOLDS}) — "
    "this can take a while...",
    flush=True,
)
localized_ecov = mesmer.stats.find_localized_empirical_covariance_monthly(
    ar1_fit.residuals,
    weights.weights,
    phi_gc_localizer,
    dim="time",
    k_folds=K_FOLDS,
)
print("  Localized covariance done.", flush=True)

# =============================================================================
# Save parameters
# =============================================================================

print("Saving parameters...", flush=True)

PARAM_FILEFINDER = filefisher.FileFinder(
    path_pattern=PARAM_PATH / "{esm}_{scen}",
    file_pattern="params_{module}_{esm}_{scen}.nc",
)

folder = PARAM_FILEFINDER.create_path_name(esm=MODEL, scen=RUN_LABEL)
pathlib.Path(folder).mkdir(exist_ok=True, parents=True)

params = {
    # residuals are large intermediates not needed for emulation
    "harmonic-model": harmonic_model_fit[["coeffs", "selected_order"]],
    "power-transformer": pt_coefficients.to_dataset(name="lambda_coeffs"),
    "local-ar1-monthly": ar1_fit[["intercept", "slope"]],
    "covariance-monthly": localized_ecov,
    "grid-orig": grid_orig,
}

for module, param in params.items():
    filename = PARAM_FILEFINDER.create_full_name(
        module=module, esm=MODEL, scen=RUN_LABEL
    )
    param.to_netcdf(filename)
    print(f"  Saved {filename}", flush=True)

print("Done!", flush=True)
