"""
Compute metrics for DiffESM and MESMER-M baselines, saving in the same
format as gen_and_save_metrics.py so they can be added to plot_metrics.py.

Saves to:
  {work_path}/computed_metrics/diffesm_{N}_{T}/predicted_output_metrics_{lat}_{lon}
  {work_path}/computed_metrics/mesmer_{N}_{T}/predicted_output_metrics_{lat}_{lon}

Usage:
  sbatch analysis/gen_baseline_metrics.sh \\
    inference.target=test inference.year=0 \\
    analysis.diffesm_year=1970 analysis.mesmer_year=1969

  DiffESM dir: gen_tas_{diffesm_year}-{diffesm_year+10}/member_x.nc
  MESMER file: tas_monthly_emulations_IPSL-CM6A-LR_init_{mesmer_year}_dcpp_m.nc
"""

import logging
import os

import hydra
import numpy as np
import pandas as pd
import torch
import xarray as xr
from omegaconf import DictConfig, OmegaConf
from tensordict.tensordict import TensorDict

from ArchesClimate.analysis.evaluate import generate_metrics
from ArchesClimate.paths import path as artifact_path

LOGGER = logging.getLogger(__name__)

DIFFESM_BASE_DIR = str(artifact_path("rollouts") / "IPSL/dcpp")
MESMER_DIR = str(artifact_path("mesmer"))
CLIMATOLOGY_PATH = str(artifact_path("stats") / "climatology_stats_net_flux.pt")

NUM_BASELINE_MEMBERS = 5
TAS_SURFACE_IDX = 10  # index of 'tas' in surface_variables list


def _yearly_mean(td):
    """Collapse monthly TensorDict to annual by averaging each 12-month block."""
    T = td["surface"].shape[2]
    n_years = T // 12

    def _avg(t):
        s = t.shape
        return (
            t[:, :, : n_years * 12].reshape(s[0], s[1], n_years, 12, *s[3:]).mean(dim=3)
        )

    out = TensorDict({"surface": _avg(td["surface"]), "level": _avg(td["level"])})
    out.batch_size = [td.batch_size[0], td.batch_size[1], n_years]
    return out


def _surface_only_tensordict(
    tas_array,
    num_surface_vars,
    num_level_vars,
    num_plevs,
    start_lat,
    end_lat,
    start_lon,
    end_lon,
):
    """
    Build a TensorDict from (members, T, lat, lon) tas_array.
    All surface variables except tas are NaN; level is all NaN.
    """
    n_members, T, nlat, nlon = tas_array.shape
    surface = torch.full(
        (1, n_members, T, num_surface_vars, 1, nlat, nlon), float("nan")
    )
    surface[0, :, :, TAS_SURFACE_IDX, 0, :, :] = tas_array
    level = torch.full(
        (1, n_members, T, num_level_vars, num_plevs, nlat, nlon), float("nan")
    )
    return TensorDict(
        {
            "surface": surface[:, :, :, :, :, start_lat:end_lat, start_lon:end_lon],
            "level": level[:, :, :, :, :, start_lat:end_lat, start_lon:end_lon],
        }
    )


def load_diffesm(
    diffesm_year,
    num_rollout_steps,
    num_surface_vars,
    num_level_vars,
    num_plevs,
    start_lat,
    end_lat,
    start_lon,
    end_lon,
    save_name=None,
):
    prefix = f"{save_name}_" if save_name else ""
    diffesm_dir = (
        f"{DIFFESM_BASE_DIR}/gen_{prefix}tas_{diffesm_year}-{diffesm_year + 10}"
    )
    members = []
    for m in range(NUM_BASELINE_MEMBERS):
        ds = xr.open_dataset(f"{diffesm_dir}/member_{m}.nc")
        tas = ds["tas"].values[:num_rollout_steps].astype(np.float32)
        if np.nanmean(tas) < 100:  # file labels units='K' incorrectly; detect by value
            tas = tas + 273.15
        members.append(tas)
        ds.close()
    tas_array = np.stack(members)  # (5, T, 143, 144)
    return _surface_only_tensordict(
        torch.tensor(tas_array),
        num_surface_vars,
        num_level_vars,
        num_plevs,
        start_lat,
        end_lat,
        start_lon,
        end_lon,
    )


def load_mesmer(
    mesmer_init_year,
    num_rollout_steps,
    num_surface_vars,
    num_level_vars,
    num_plevs,
    target_lats,
    target_lons,
    start_lat,
    end_lat,
    start_lon,
    end_lon,
):
    mesmer_file = (
        f"tas_monthly_emulations_IPSL-CM6A-LR_init_{mesmer_init_year}_dcpp_m.nc"
    )
    ds = xr.open_dataset(f"{MESMER_DIR}/{mesmer_file}")
    # 1. Define the target grid coordinates exactly as you requested
    target_lat = np.linspace(-90, 90, 143)
    target_lon = np.linspace(0, 360, 144)

    # 2. Prepare the empty NaN array (Time, Realisation, Lat, Lon)
    # Assuming ds['tas'] is (time: 120, realisation: 30, gridcell: 5121)
    ntime = ds.time.size
    nreal = ds.realisation.size
    grid_shape = (ntime, nreal, 143, 144)
    gridded_array = np.full(grid_shape, np.nan)

    # 3. Find the nearest grid indices for your 5121 points
    # We map each coordinate in 'ds' to the closest index in our 143x144 arrays
    def get_loc(value, target_array):
        # Returns the index of the closest value in the target array
        return np.abs(target_array[:, None] - value).argmin(axis=0)

    mesmer_lons = ds.lon.values % 360  # normalise -180→180 to 0→360
    print(
        f"[MESMER] lon range before norm: {ds.lon.values.min():.1f} → {ds.lon.values.max():.1f}"
    )
    print(
        f"[MESMER] lon range after norm:  {mesmer_lons.min():.1f} → {mesmer_lons.max():.1f}"
    )
    lat_idxs = get_loc(ds.lat.values, target_lat)
    lon_idxs = get_loc(mesmer_lons, target_lon)

    data_values = ds["tas"].transpose(..., "gridcell").values

    print("Mapping points to land surface buckets...")
    for t in range(ntime):
        for r in range(nreal):
            # This line maps the 5121 points to their specific [lat, lon] coordinates
            gridded_array[t, r, lat_idxs, lon_idxs] = data_values[t, r, :]

    # # 5. Create final Dataset
    # ds_final = xr.Dataset(
    #     data_vars={
    #         "tas": (("time", "realisation", "lat", "lon"), gridded_array)
    #     },
    #     coords={
    #         "time": ds.time,
    #         "realisation": ds.realisation,
    #         "lat": target_lat,
    #         "lon": target_lon
    #     }
    # )
    # Load 2010 file if it exists and concatenate along time
    mesmer_file_2010 = "tas_monthly_emulations_IPSL-CM6A-LR_init_2010_dcpp_m.nc"
    alt_path = f"{MESMER_DIR}/{mesmer_file_2010}"
    try:
        ds2 = xr.open_dataset(alt_path)
        data2 = ds2["tas"].transpose(..., "gridcell").values
        gridded_array2 = np.full(
            (ds2.time.size, ds2.realisation.size, 143, 144), np.nan
        )
        for t in range(ds2.time.size):
            for r in range(ds2.realisation.size):
                gridded_array2[t, r, lat_idxs, lon_idxs] = data2[t, r, :]
        gridded_array = np.concatenate([gridded_array, gridded_array2], axis=0)
        mesmer_times = np.concatenate([ds.time.values, ds2.time.values])
        mesmer_times = pd.to_datetime(mesmer_times)
        print(
            f"[MESMER] concatenated with 2010 file → total timesteps: {len(mesmer_times)}"
        )
    except Exception as e:
        LOGGER.warning("Optional analysis input or plot failed", exc_info=True)
        mesmer_times = pd.to_datetime(ds.time.values)
        print(f"[MESMER] could not load 2010 file: {e}")

    print(
        f"[MESMER] file time range: {mesmer_times[0]} → {mesmer_times[-1]}  (n={len(mesmer_times)})"
    )
    print(f"[MESMER] start month (1=Jan): {mesmer_times[0].month}")

    ds_clim = xr.open_dataset(str(artifact_path("data") / "dcpp_dataset/1960_1.nc"))
    climatology = torch.tensor(
        ds_clim["tas"].groupby("time.month").mean("time").values
    )  # [12, lat, lon]
    print(f"[MESMER] climatology shape (months, lat, lon): {climatology.shape}")

    # Build per-timestep climatology aligned to the actual calendar months
    month_indices = torch.tensor(
        [pd.Timestamp(t).month - 1 for t in mesmer_times]
    )  # 0-indexed
    ntime = len(mesmer_times)
    clim_per_timestep = climatology[month_indices]  # [ntime, lat, lon]

    gridded_array = torch.tensor(gridded_array)

    # Find first non-NaN land point for diagnostics
    flat = gridded_array[0, 0].reshape(-1)
    nnan_idx = (~flat.isnan()).nonzero(as_tuple=True)[0]
    if len(nnan_idx) > 0:
        diag_flat = nnan_idx[0].item()
        diag_lat, diag_lon = diag_flat // 144, diag_flat % 144
        print(
            f"[MESMER] first non-NaN land point: lat_idx={diag_lat}, lon_idx={diag_lon}"
        )
        print(
            f"[MESMER] raw anomaly at land point, first 12 months (r=0): {gridded_array[:12, 0, diag_lat, diag_lon].tolist()}"
        )
        print(
            f"[MESMER] climatology at land point, first 12 months: {clim_per_timestep[:12, diag_lat, diag_lon].tolist()}"
        )
    else:
        print("[MESMER] WARNING: no non-NaN land points found in gridded_array!")

    gridded_array = (
        gridded_array + clim_per_timestep[:120].nanmean(axis=0)[None, None]
    )  # broadcast over realisations

    if len(nnan_idx) > 0:
        print(
            f"[MESMER] TAS after adding clim, first 12 months (r=0): {gridded_array[:12, 0, diag_lat, diag_lon].tolist()}"
        )

    print(
        f"[MESMER] gridded TAS first 12 months at (r=0, lat=70, lon=72): {gridded_array[:12, 0, 70, 72].tolist()}"
    )
    ntime_available = gridded_array.shape[0]
    print(
        f"[MESMER] available timesteps: {ntime_available}, requested: {num_rollout_steps}"
    )
    if ntime_available < num_rollout_steps:
        print(
            f"[MESMER] padding {num_rollout_steps - ntime_available} missing timesteps with NaN"
        )
        pad = torch.full(
            (
                num_rollout_steps - ntime_available,
                gridded_array.shape[1],
                gridded_array.shape[2],
                gridded_array.shape[3],
            ),
            float("nan"),
        )
        gridded_array = torch.cat([gridded_array, pad], dim=0)
    return _surface_only_tensordict(
        gridded_array[:num_rollout_steps, :5].swapdims(0, 1),
        num_surface_vars,
        num_level_vars,
        num_plevs,
        start_lat,
        end_lat,
        start_lon,
        end_lon,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    num_rollout_steps = cfg.analysis.num_rollout_steps
    target = cfg.inference.target
    year = cfg.inference.year
    start_lat = int(cfg.analysis.start_lat)
    end_lat = None if cfg.analysis.end_lat == "None" else int(cfg.analysis.end_lat)
    start_lon = int(cfg.analysis.start_lon)
    end_lon = None if cfg.analysis.end_lon == "None" else int(cfg.analysis.end_lon)

    num_surface_vars = len(cfg.module.surface_variables)
    num_level_vars = len(cfg.module.level_variables)
    num_plevs = len(cfg.module.pressure_levels)

    # Load test ensemble and denormalize; use members 5-9 as reference (same as gen_and_save_metrics.py)
    train_spatial_norm = hydra.utils.instantiate(
        cfg.dataloader.dataset,
        domain="all",
        limit_examples=100,
        norm_scheme="spatial_norm",
    )
    test_path = (
        f"""{artifact_path("targets")}/{target}_ensemble_{year}_flow_net_flux_1"""
    )
    test_ensemble = torch.load(test_path)[:, : cfg.inference.num_rollout_steps]
    test_ensemble["surface"] = test_ensemble["surface"][
        :, :, cfg.module.surface_variable_indices
    ]
    denorm_test_full = train_spatial_norm.denormalize(
        test_ensemble[None], stateless=True
    )
    print("--- baseline ref shape audit ---")
    print("raw test_ensemble surface:", test_ensemble["surface"].shape)
    print("denorm_test_full surface:", denorm_test_full["surface"].shape)
    print(
        "denorm_test_full surface nans:",
        denorm_test_full["surface"].isnan().sum().item(),
    )
    denorm_test = denorm_test_full[:, 5:10, :num_rollout_steps]
    print("denorm_test after slice [:,5:10] surface:", denorm_test["surface"].shape)
    print(
        "denorm_test tas sample (step 0, gridpoint 70,72):",
        denorm_test["surface"][0, :, 0, 10, 0, 70, 72]
        if denorm_test["surface"].shape[1] > 0
        else "EMPTY",
    )
    print("--------------------------------")
    denorm_test = TensorDict(
        {
            "surface": denorm_test["surface"][
                :, :, :, :, :, start_lat:end_lat, start_lon:end_lon
            ],
            "level": denorm_test["level"][
                :, :, :, :, :, start_lat:end_lat, start_lon:end_lon
            ],
        }
    )
    denorm_test.batch_size = [1, 5, num_rollout_steps]

    diffesm_year = int(OmegaConf.select(cfg, "analysis.diffesm_year", default=1970))
    mesmer_year = int(OmegaConf.select(cfg, "analysis.mesmer_year", default=1969))

    # Lat/lon grid from DiffESM (matches model output grid)
    diffesm_dir = f"{DIFFESM_BASE_DIR}/gen_tas_{diffesm_year}-{diffesm_year + 10}"
    ref_ds = xr.open_dataset(f"{diffesm_dir}/member_5.nc")
    target_lats = ref_ds["lat"].values
    target_lons = ref_ds["lon"].values
    ref_ds.close()

    # num_members from analysis config drives directory naming to match plot_metrics.py convention
    num_members = cfg.analysis.num_members
    metric_suffix = (
        f"predicted_output_metrics_{start_lat}_{end_lat}_{start_lon}_{end_lon}"
    )

    # --- DiffESM ---
    print(f"Processing DiffESM (year={diffesm_year})...")
    diffesm_data = load_diffesm(
        diffesm_year,
        num_rollout_steps,
        num_surface_vars,
        num_level_vars,
        num_plevs,
        start_lat,
        end_lat,
        start_lon,
        end_lon,
    )
    diffesm_data.batch_size = [1, NUM_BASELINE_MEMBERS, num_rollout_steps]
    print("diffesm surface shape:", diffesm_data["surface"].shape)
    print(
        "diffesm tas nans:",
        diffesm_data["surface"][:, :, :, TAS_SURFACE_IDX].isnan().sum().item(),
    )
    print(
        "diffesm tas sample (member 0, step 0, gridpoint 70,72):",
        diffesm_data["surface"][0, 0, 0, TAS_SURFACE_IDX, 0, 70, 72],
    )
    print(
        "ref tas sample  (member 0, step 0, gridpoint 70,72):",
        denorm_test["surface"][0, 0, 0, TAS_SURFACE_IDX, 0, 70, 72],
    )
    diffesm_metrics = generate_metrics(diffesm_data, denorm_test)
    out_dir = (
        f"{cfg.paths.metrics}/diffesm_{diffesm_year}_{num_members}_{num_rollout_steps}"
    )
    os.makedirs(out_dir, exist_ok=True)
    torch.save(diffesm_metrics, f"{out_dir}/{metric_suffix}")
    print(f"Saved DiffESM metrics -> {out_dir}/{metric_suffix}")

    # --- MESMER-M ---
    print(f"Processing MESMER-M (init_year={mesmer_year})...")
    mesmer_data = load_mesmer(
        mesmer_year,
        num_rollout_steps,
        num_surface_vars,
        num_level_vars,
        num_plevs,
        target_lats,
        target_lons,
        start_lat,
        end_lat,
        start_lon,
        end_lon,
    )
    mesmer_data.batch_size = [1, NUM_BASELINE_MEMBERS, num_rollout_steps]
    print(
        "mesmer tas sample (member 0, step 0, gridpoint 70,72):",
        mesmer_data["surface"][0, 0, 0, TAS_SURFACE_IDX, 0, 70, 72],
    )
    mesmer_metrics = generate_metrics(mesmer_data[:, :, 1:], denorm_test)
    out_dir = (
        f"{cfg.paths.metrics}/mesmer_{mesmer_year}_{num_members}_{num_rollout_steps}"
    )
    os.makedirs(out_dir, exist_ok=True)
    torch.save(mesmer_metrics, f"{out_dir}/{metric_suffix}")
    print(f"Saved MESMER-M metrics -> {out_dir}/{metric_suffix}")
    print("mesmer tas mse at step 0:", mesmer_metrics["surface"][0, 0, TAS_SURFACE_IDX])

    # Land-only TAS: mask ocean (TOS non-NaN = ocean) so DiffESM is comparable to MESMER
    _ocean = ~denorm_test["surface"][0, 0, 0, -1, 0].isnan()  # [nlat, nlon] True=ocean

    def _land_mask_bl(td):
        s = td["surface"].clone()
        s[:, :, :, TAS_SURFACE_IDX, 0] = s[:, :, :, TAS_SURFACE_IDX, 0].masked_fill(
            _ocean, float("nan")
        )
        out = TensorDict({"surface": s, "level": td["level"]})
        out.batch_size = list(td.batch_size)
        return out

    land_suffix = metric_suffix.replace(
        "predicted_output_metrics", "predicted_output_metrics_land"
    )

    diffesm_land_metrics = generate_metrics(_land_mask_bl(diffesm_data), denorm_test)
    diffesm_out_dir = (
        f"{cfg.paths.metrics}/diffesm_{diffesm_year}_{num_members}_{num_rollout_steps}"
    )
    torch.save(diffesm_land_metrics, f"{diffesm_out_dir}/{land_suffix}")

    # MESMER is already land-only (ocean NaN by construction); save under land name too
    mesmer_out_dir = (
        f"{cfg.paths.metrics}/mesmer_{mesmer_year}_{num_members}_{num_rollout_steps}"
    )
    torch.save(mesmer_metrics, f"{mesmer_out_dir}/{land_suffix}")
    print("Saved land-only TAS metrics for DiffESM and MESMER")

    # --- Yearly analysis ---
    yearly_suffix = metric_suffix.replace(
        "predicted_output_metrics", "predicted_output_metrics_yearly"
    )
    land_yearly_suffix = metric_suffix.replace(
        "predicted_output_metrics", "predicted_output_metrics_land_yearly"
    )
    test_yearly = _yearly_mean(denorm_test)

    diffesm_metrics_yearly = generate_metrics(_yearly_mean(diffesm_data), test_yearly)
    torch.save(diffesm_metrics_yearly, f"{diffesm_out_dir}/{yearly_suffix}")
    diffesm_land_metrics_yearly = generate_metrics(
        _land_mask_bl(_yearly_mean(diffesm_data)), test_yearly
    )
    torch.save(diffesm_land_metrics_yearly, f"{diffesm_out_dir}/{land_yearly_suffix}")

    mesmer_metrics_yearly = generate_metrics(_yearly_mean(mesmer_data), test_yearly)
    torch.save(mesmer_metrics_yearly, f"{mesmer_out_dir}/{yearly_suffix}")
    torch.save(mesmer_metrics_yearly, f"{mesmer_out_dir}/{land_yearly_suffix}")
    print("Saved yearly metrics for DiffESM and MESMER")

    # Spatial RMSE maps averaged over all timesteps: shape [num_vars, 1, nlat, nlon]
    spatial_suffix = (
        f"predicted_spatial_rmse_{start_lat}_{end_lat}_{start_lon}_{end_lon}"
    )
    _tgt_mean = denorm_test["surface"][0].mean(dim=0)  # [T, vars, 1, lat, lon]

    def _spatial_rmse(td):
        pred_mean = td["surface"][0].mean(dim=0)  # [T, vars, 1, lat, lon]
        return (((pred_mean - _tgt_mean) ** 2).mean(dim=0)).sqrt()

    torch.save(_spatial_rmse(diffesm_data), f"{diffesm_out_dir}/{spatial_suffix}")
    torch.save(_spatial_rmse(mesmer_data), f"{mesmer_out_dir}/{spatial_suffix}")
    print("Saved spatial RMSE maps for DiffESM and MESMER")


if __name__ == "__main__":
    main()
