from ArchesClimate.paths import path as artifact_path
import os

os.environ["PROJ_NETWORK"] = "OFF"

import pathlib
from pathlib import Path

import filefisher
import xarray as xr

import mesmer

# =============================================================================
# Configuration
# =============================================================================

MODEL = "IPSL-CM6A-LR"
THRESHOLD_LAND = 1 / 3
REFERENCE_PERIOD = slice("1960", "1970")
REF_SCENARIO = "init_1960"
N_STEPS_LOWESS = 50
DCPP_DATA_PATH = Path(str(artifact_path('data') / 'dcpp_dataset/'))
PARAM_PATH = pathlib.Path("./output/calibrated_parameters/")
RUN_LABEL = "dcpp"  # used for naming output files

EXCLUDED_INIT_YEARS = [1969, 1979, 1989]
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
fc_all.df = df[training_mask].sort_values(by=["init_year", "member"]).reset_index(
    drop=True
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
            file_path = DCPP_DATA_PATH / f"{int(row['init_year'])}_{int(row['member'])}.nc"
            if not file_path.exists():
                print(f"  Warning: missing {file_path}, skipping.", flush=True)
                continue

            time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
            ds = xr.open_dataset(str(file_path), decode_times=time_coder, engine="netcdf4")
            ds = ds.drop_vars(["height", "time_bnds", "file_qf"], errors="ignore")
            ds = ds.assign_coords({"member": int(row["member"])})
            members.append(ds[["tas"]])

        if members:
            out[f"init_{init_yr}"] = xr.DataTree(xr.concat(members, dim="member"))

    return out


dt = load_data(fc_all)
print(f"  DataTree loaded with {len(dt.children)} init years.", flush=True)

# =============================================================================
# Anomalies
# =============================================================================

print("Calculating anomalies...", flush=True)
tas_anom = mesmer.anomaly.calc_anomaly(
    dt, reference_period=REFERENCE_PERIOD, ref_scenario=REF_SCENARIO
)

# =============================================================================
# Global mean and smooth trend
# =============================================================================

print("Calculating global mean...", flush=True)
tas_globmean = mesmer.weighted.global_mean(tas_anom)

print("Smoothing global mean (lowess)...", flush=True)
tas_globmean_ensmean = tas_globmean.mean(dim="member")
tas_globmean_smoothed = mesmer.stats.lowess(
    tas_globmean_ensmean,
    dim="time",
    n_steps=N_STEPS_LOWESS,
    use_coords=False,
)

# =============================================================================
# Global variability
# =============================================================================

print("Computing global variability residuals...", flush=True)
tas_globmean_resids = tas_globmean - tas_globmean_smoothed
tas_globmean_resids = mesmer.datatree.map_over_datasets(
    lambda ds: ds.rename({"tas": "tas_resids"}), tas_globmean_resids
)

print("Fitting global AR parameters...", flush=True)
# fit_auto_regression_scen_ens produces intercept/coeffs/variance, which is
# the format draw_auto_regression_uncorrelated expects in the eval script.
global_ar_params = mesmer.stats.fit_auto_regression_scen_ens(
    tas_globmean_resids, dim="time", ens_dim="member", lags=1
)

# =============================================================================
# Local forced response
# =============================================================================

print("Masking and stacking local data...", flush=True)
grid_orig = tas_anom[REF_SCENARIO].ds[["lat", "lon"]]


def mask_and_stack(dt, threshold_land):
    dt = mesmer.mask.mask_ocean_fraction(dt, threshold_land)
    dt = mesmer.mask.mask_antarctica(dt)
    dt = mesmer.grid.stack_lat_lon(dt)
    return dt


tas_stacked = mask_and_stack(tas_anom, THRESHOLD_LAND)

print("Pooling predictors and target...", flush=True)
predictors = xr.merge([tas_globmean_smoothed, tas_globmean_resids])
target = tas_stacked.copy()
weights = mesmer.weighted.equal_scenario_weights_from_datatree(tas_stacked)

predictors_pooled, target_pooled, weights_pooled = (
    mesmer.datatree.broadcast_and_pool_scen_ens(predictors, target, weights)
)

print("Fitting local linear regression...", flush=True)
local_lin_reg = mesmer.stats.LinearRegression()
local_lin_reg.fit(
    predictors=predictors_pooled,
    target=target_pooled.tas,
    dim="sample",
    weights=weights_pooled.weights,
)

# =============================================================================
# Local variability
# =============================================================================

print("Computing local residuals...", flush=True)
resids = local_lin_reg.residuals(predictors, target)
resids_pooled = mesmer.datatree.pool_scen_ens(resids)

print("Fitting local AR(1) — this can take several minutes...", flush=True)
# Select one member per init year to reduce the AR fitting cost.
# fit_auto_regression_scen_ens averages over members anyway, so using one
# member gives the same result structure with ~10x less work.
resids_one_member = mesmer.datatree.map_over_datasets(
    lambda ds: ds.isel(member=0, drop=False), resids
)
local_ar = mesmer.stats.fit_auto_regression_scen_ens(
    resids_one_member,
    ens_dim="member",
    dim="time",
    lags=1,
)
print("  AR(1) done.", flush=True)

print("Computing geodistance matrix...", flush=True)
grid_stacked = resids[REF_SCENARIO].ds[["lat", "lon"]]
geodist = mesmer.geospatial.geodist_exact(grid_stacked.lon, grid_stacked.lat)

print("Building Gaspari-Cohn localizers...", flush=True)
phi_gc_localizer = mesmer.stats.gaspari_cohn_correlation_matrices(
    geodist, range(5_000, 15_001, 2_500)
)

print("Finding localized empirical covariance (cross-validation)...", flush=True)
localized_ecov = mesmer.stats.find_localized_empirical_covariance(
    resids_pooled.residuals,
    weights_pooled.weights,
    phi_gc_localizer,
    dim="sample",
    k_folds=5,
)
print(f"  Selected localization radius: {localized_ecov.localization_radius.item()} km", flush=True)

localized_covariance_adjusted = mesmer.stats.adjust_covariance_ar1(
    localized_ecov.localized_covariance, local_ar.coeffs
)

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
    "global-variability": global_ar_params,
    "local-trends": local_lin_reg,
    "local-variability": local_ar,
    "covariance": localized_ecov,
    "grid-orig": grid_orig,
}

for module, param in params.items():
    filename = PARAM_FILEFINDER.create_full_name(
        module=module,
        esm=MODEL,
        scen=RUN_LABEL,
    )
    param.to_netcdf(filename)
    print(f"  Saved {filename}", flush=True)

print("Done!", flush=True)
