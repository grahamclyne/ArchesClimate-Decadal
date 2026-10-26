from ArchesClimate.paths import path as artifact_path
import os

os.environ["PROJ_NETWORK"] = "OFF"

import pathlib
from pathlib import Path

import filefisher
import xarray as xr

import mesmer

# =============================================================================
# Configuration — must match the calibration script settings
# =============================================================================

MODEL = "IPSL-CM6A-LR"
THRESHOLD_LAND = 1 / 3
REFERENCE_PERIOD = slice("1960", "1970")
REF_INIT_YEAR = 1960
EVAL_INIT_YEARS = [1969, 1979, 2010]
N_STEPS_LOWESS = 50
N_REALISATIONS = 30

DCPP_DATA_PATH = Path(str(artifact_path('data') / 'dcpp_dataset/'))
PARAM_PATH = pathlib.Path("./output/calibrated_parameters/")
EMU_PATH = pathlib.Path("./output/emulations/")
RUN_LABEL = "dcpp"

BUFFER_GLOBAL_VARIABILITY = 50
BUFFER_LOCAL_VARIABILITY = 20

# Fixed seeds for reproducibility — one per eval init year
SEEDS_GLOBAL = {1969: 111, 1979: 222, 2010: 333}
SEEDS_LOCAL  = {1969: 444, 1979: 555, 2010: 666}

# =============================================================================
# Helper: load all members for one init year
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
    if not members:
        raise FileNotFoundError(f"No files found for init year {init_yr}")
    return xr.concat(members, dim="member")

# =============================================================================
# Load reference data (init_1960) for anomaly baseline
# =============================================================================

print("Loading reference data (init_1960)...", flush=True)
ref_ds = load_init_year(REF_INIT_YEAR)
ref_mean = ref_ds.sel(time=REFERENCE_PERIOD).mean("time")
print(f"  Reference mean computed from {REF_INIT_YEAR} over {REFERENCE_PERIOD}", flush=True)

# =============================================================================
# Load calibrated parameters
# =============================================================================

print("Loading calibrated parameters...", flush=True)
PARAM_FILEFINDER = filefisher.FileFinder(
    path_pattern=PARAM_PATH / "{esm}_{scen}",
    file_pattern="params_{module}_{esm}_{scen}.nc",
)

params = {}
for module in ["global-variability", "local-trends", "local-variability", "covariance"]:
    fname = PARAM_FILEFINDER.create_full_name(module=module, esm=MODEL, scen=RUN_LABEL)
    params[module] = xr.open_dataset(fname)
    print(f"  Loaded {module}", flush=True)

lr = mesmer.stats.LinearRegression.from_params(params["local-trends"])
grid_orig = xr.open_dataset(
    PARAM_FILEFINDER.create_full_name(module="grid-orig", esm=MODEL, scen=RUN_LABEL)
)

# =============================================================================
# Generate emulations for each held-out init year
# =============================================================================

EMU_PATH.mkdir(exist_ok=True, parents=True)

for init_yr in EVAL_INIT_YEARS:
    key = f"init_{init_yr}"
    print(f"\nGenerating emulations for {key}...", flush=True)

    # --- anomalies (same reference as training) ---
    eval_ds = load_init_year(init_yr)
    tas_anom_ds = eval_ds - ref_mean

    # wrap in a single-node DataTree so mesmer utilities work
    dt = xr.DataTree.from_dict({key: tas_anom_ds})

    # --- global mean and smooth forcing ---
    tas_globmean = mesmer.weighted.global_mean(dt)
    tas_globmean_ensmean = tas_globmean.mean(dim="member")
    tas_globmean_forcing = mesmer.stats.lowess(
        tas_globmean_ensmean,
        dim="time",
        n_steps=N_STEPS_LOWESS,
        use_coords=False,
    )

    time = tas_globmean_forcing[key].ds.time
    print(f"  Time span: {time.values[0]} → {time.values[-1]}", flush=True)

    # --- seeds for this init year (single-node DataTree) ---
    seed_gv = xr.DataTree.from_dict({key: xr.Dataset({"seed": SEEDS_GLOBAL[init_yr]})})
    seed_lv = xr.DataTree.from_dict({key: xr.Dataset({"seed": SEEDS_LOCAL[init_yr]})})

    # --- 1. draw global variability ---
    print("  Drawing global variability...", flush=True)
    global_variability = mesmer.stats.draw_auto_regression_uncorrelated(
        params["global-variability"],
        time=time,
        realisation=N_REALISATIONS,
        seed=seed_gv,
        buffer=BUFFER_GLOBAL_VARIABILITY,
    )
    global_variability = mesmer.datatree.map_over_datasets(
        lambda ds: ds.rename({"samples": "tas_resids"}), global_variability
    )

    # --- 2. local forced response ---
    print("  Computing local forced response...", flush=True)
    predictors = xr.merge([tas_globmean_forcing, global_variability])

    local_forced_response = lr.predict(predictors, exclude={"tas_resids"})
    local_variability_from_global_var = lr.predict(predictors, only={"tas_resids"})

    # --- 3. draw local variability ---
    print("  Drawing local variability...", flush=True)
    local_variability = mesmer.stats.draw_auto_regression_correlated(
        params["local-variability"],
        params["covariance"].localized_covariance,
        time=time,
        realisation=N_REALISATIONS,
        seed=seed_lv,
        buffer=BUFFER_LOCAL_VARIABILITY,
    )
    local_variability = mesmer.datatree.map_over_datasets(
        lambda ds: ds.rename({"samples": "prediction"}), local_variability
    )

    # --- 4. combine ---
    local_variability_total = local_variability_from_global_var + local_variability
    emulations = local_forced_response + local_variability_total
    emulations = mesmer.datatree.map_over_datasets(
        lambda ds: ds.rename({"prediction": "tas"}), emulations
    )

    # --- save (stacked gridcell dim) ---
    emu_ds = emulations[key].ds.copy()
    emu_ds["seed_global_variability"] = SEEDS_GLOBAL[init_yr]
    emu_ds["seed_local_variability"] = SEEDS_LOCAL[init_yr]

    out_path = EMU_PATH / f"tas_emulations_{MODEL}_{key}_{RUN_LABEL}.nc"
    emu_ds.to_netcdf(out_path)
    print(f"  Saved {out_path}", flush=True)

print("\nDone!", flush=True)
