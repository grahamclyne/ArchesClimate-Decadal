import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tensordict.tensordict import TensorDict

from ArchesClimate.analysis.evaluate import generate_metrics
from ArchesClimate.paths import path as artifact_path


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


def load_experiment(cfg, experiment, num_members, num_rollout_steps):
    member_level_rollouts = []
    member_surface_rollouts = []
    for member_index in range(num_members):
        member_level_rollouts.append(
            torch.load(
                f"{cfg.paths.rollouts}/{experiment}/rollout_{member_index}_level.pt",
                map_location=torch.device("cpu"),
            )
            .cpu()
            .numpy()
        )
        member_surface_rollouts.append(
            torch.load(
                f"{cfg.paths.rollouts}/{experiment}/rollout_{member_index}_surface.pt",
                map_location=torch.device("cpu"),
            )
            .cpu()
            .numpy()
        )
    level_rollouts = np.stack(member_level_rollouts).squeeze(1)
    surface_rollouts = np.stack(member_surface_rollouts).squeeze(1)
    print(level_rollouts.shape)
    print(surface_rollouts.shape)
    data = TensorDict({"surface": surface_rollouts, "level": level_rollouts})
    data["surface"] = data["surface"][:, :num_rollout_steps]
    data["level"] = data["level"][:, :num_rollout_steps]
    data.batch_size = [num_members, num_rollout_steps]
    return data


def load_climatology(
    cfg,
    num_members,
    num_rollout_steps,
    denormalized_predicted_ensemble,
    start_lat,
    end_lat,
    start_lon=0,
    end_lon=None,
):
    end_lat = None if end_lat == "None" else end_lat
    end_lon = None if end_lon == "None" else end_lon
    climatology = torch.load(
        str(artifact_path("stats") / "climatology_stats_net_flux.pt")
    )
    # climatology['surface_mean'] = climatology['surface_mean'][:,cfg.module.surface_variable_indices]
    # climatology['level_mean'] = climatology['level_mean'][:,cfg.module.level_variable_indices]

    # climatology['surface_std'] = climatology['surface_std'][:,cfg.module.surface_variable_indices]

    import math

    n_repeat = math.ceil((num_rollout_steps + 1) / 12)
    climatology_surface = climatology["surface_mean"].repeat(n_repeat, 1, 1, 1, 1)[
        None,
        None,
        1 : num_rollout_steps + 1,
        :,
        :,
        start_lat:end_lat,
        start_lon:end_lon,
    ]
    climatology_level = climatology["level_mean"].repeat(n_repeat, 1, 1, 1, 1)[
        None,
        None,
        1 : num_rollout_steps + 1,
        :,
        :,
        start_lat:end_lat,
        start_lon:end_lon,
    ]
    surface_mask = torch.where(denormalized_predicted_ensemble["surface"].isnan(), 1, 0)
    level_mask = torch.where(denormalized_predicted_ensemble["level"].isnan(), 1, 0)
    # align dims: climatology is 7D [1,1,T,...], mask may be 6D [N,T,...] from callers without batch dim
    while surface_mask.dim() < climatology_surface.dim():
        surface_mask = surface_mask.unsqueeze(0)
    while level_mask.dim() < climatology_level.dim():
        level_mask = level_mask.unsqueeze(0)
    climatology_surface = torch.where(surface_mask == 1, torch.nan, climatology_surface)
    climatology_level = torch.where(level_mask == 1, torch.nan, climatology_level)

    climatology_ensemble = TensorDict(
        {"surface": climatology_surface, "level": climatology_level}
    )
    print(climatology_ensemble)
    climatology_ensemble.batch_size = [
        1,
        climatology_surface.shape[1],
        num_rollout_steps,
    ]
    return climatology_ensemble


def load_predicted_and_target_ensembles(
    cfg,
    experiment,
    num_members,
    num_rollout_steps,
    start_lat=0,
    end_lat=None,
    start_lon=0,
    end_lon=None,
):
    """Load the raw (member-axis-intact) predicted and target ensembles for one
    experiment/year, denormalized and cropped to the requested lat/lon window.

    Factored out of run_metrics() so bootstrap_metrics.py can resample the member
    axis of both ensembles before calling generate_metrics(), without duplicating
    (and risking drift from) this loading logic.

    Returns: (denormalized_predicted_ensemble, denormalized_test_ensemble, n_pred, pred_raw, test_ensemble)
    """
    # only needed for denorm function, sometimes different from predicted (e.g. non spatial std)
    train_spatial_norm = hydra.utils.instantiate(
        cfg.dataloader.dataset,
        domain="all",
        limit_examples=100,
        norm_scheme="spatial_norm",
    )
    train_mean_only_spatial_norm = hydra.utils.instantiate(
        cfg.dataloader.dataset,
        domain="all",
        limit_examples=100,
        norm_scheme="mean_only_spatial_norm",
    )
    target = cfg.inference.target
    year = cfg.inference.year

    # One hindcast start (split + year) per call. Splits are NOT pooled as extra
    # members: each start is scored on its own (npred vs held-out members) and
    # per-start metrics are averaged afterwards (see compare_table3.py).
    _active_splits = [(target, experiment)]

    _pred_raw_parts, _pred_denorm_parts = [], []
    _raw_parts, _denorm_parts = [], []
    for _tgt, _companion in _active_splits:
        _d = load_experiment(cfg, _companion, num_members, num_rollout_steps)
        _pred_raw_parts.append(_d)
        _pred_denorm_parts.append(
            train_mean_only_spatial_norm.denormalize(_d[None], stateless=True)[
                :, :num_members, :num_rollout_steps
            ]
        )
        _tgt_path = (
            f"""{artifact_path("targets")}/{_tgt}_ensemble_{year}_flow_net_flux_1"""
        )
        _ens = torch.load(_tgt_path)[:, : cfg.inference.num_rollout_steps]
        if _ens["surface"].shape[2] > len(
            cfg.module.surface_variable_indices
        ):  # legacy 19-var target files
            _ens["surface"] = _ens["surface"][:, :, cfg.module.surface_variable_indices]
        _raw_parts.append(_ens)
        _denorm_parts.append(
            train_spatial_norm.denormalize(_ens[None], stateless=True)[
                :, :, :num_rollout_steps
            ]
        )

    pred_raw = TensorDict(
        {
            "surface": torch.cat([p["surface"] for p in _pred_raw_parts], dim=0),
            "level": torch.cat([p["level"] for p in _pred_raw_parts], dim=0),
        }
    )
    denormalized_predicted_ensemble = TensorDict(
        {
            "surface": torch.cat([p["surface"] for p in _pred_denorm_parts], dim=1)[
                :, :, :, :, :, start_lat:end_lat, start_lon:end_lon
            ],
            "level": torch.cat([p["level"] for p in _pred_denorm_parts], dim=1)[
                :, :, :, :, :, start_lat:end_lat, start_lon:end_lon
            ],
        }
    )
    n_pred = denormalized_predicted_ensemble["surface"].shape[1]

    test_ensemble = TensorDict(
        {
            "surface": torch.cat([p["surface"] for p in _raw_parts], dim=0),
            "level": torch.cat([p["level"] for p in _raw_parts], dim=0),
        }
    )
    test_ensemble.batch_size = [test_ensemble["surface"].shape[0], num_rollout_steps]

    denormalized_test_ensemble = TensorDict(
        {
            "surface": torch.cat([p["surface"] for p in _denorm_parts], dim=1)[
                :, :, :, :, :, start_lat:end_lat, start_lon:end_lon
            ],
            "level": torch.cat([p["level"] for p in _denorm_parts], dim=1)[
                :, :, :, :, :, start_lat:end_lat, start_lon:end_lon
            ],
        }
    )
    num_test_members = denormalized_test_ensemble["surface"].shape[1]
    denormalized_test_ensemble.batch_size = [1, num_test_members, num_rollout_steps]
    denormalized_predicted_ensemble.batch_size = [1, n_pred, num_rollout_steps]
    pred_raw.batch_size = [n_pred, num_rollout_steps]

    return (
        denormalized_predicted_ensemble,
        denormalized_test_ensemble,
        n_pred,
        pred_raw,
        test_ensemble,
    )


def run_metrics(
    cfg,
    experiment,
    num_members=None,
    num_rollout_steps=None,
    start_lat=0,
    end_lat=None,
    start_lon=0,
    end_lon=None,
):
    """Compute and save all metrics for a given experiment. Can be called directly."""
    if num_members is None:
        num_members = cfg.analysis.num_members
    if num_rollout_steps is None:
        num_rollout_steps = cfg.analysis.num_rollout_steps
    out_dir = experiment + f"_{num_members}_{num_rollout_steps}"
    os.makedirs(f"{cfg.paths.metrics}/{out_dir}", exist_ok=True)
    # target = experiment.split('_')[-2]
    target = cfg.inference.target
    # year = experiment.split('_')[-1]
    year = cfg.inference.year

    (
        denormalized_predicted_ensemble,
        denormalized_test_ensemble,
        n_pred,
        pred_raw,
        test_ensemble,
    ) = load_predicted_and_target_ensembles(
        cfg,
        experiment,
        num_members,
        num_rollout_steps,
        start_lat,
        end_lat,
        start_lon,
        end_lon,
    )
    num_test_members = denormalized_test_ensemble["surface"].shape[1]

    print("--- shape audit ---")
    print(
        "pred members:",
        n_pred,
        "  test members:",
        num_test_members,
        "  held-out target members:",
        num_test_members - n_pred,
    )
    print("denorm_predicted surface:", denormalized_predicted_ensemble["surface"].shape)
    print("denorm_test surface:     ", denormalized_test_ensemble["surface"].shape)
    print("--- nan audit ---")
    print(
        "denorm_predicted surface nans:",
        denormalized_predicted_ensemble["surface"].isnan().sum().item(),
    )
    print(
        "denorm_test surface nans:     ",
        denormalized_test_ensemble["surface"].isnan().sum().item(),
    )
    print("-------------------")

    pred_output_metrics = generate_metrics(
        denormalized_predicted_ensemble[:, :n_pred],
        denormalized_test_ensemble[:, n_pred:, :],
    )
    normalized_pred_output_metrics = generate_metrics(
        pred_raw[None][:, :n_pred], test_ensemble[None][:, n_pred:]
    )
    itself_output_metrics = generate_metrics(
        denormalized_test_ensemble[:, :n_pred], denormalized_test_ensemble[:, n_pred:]
    )

    pred_yearly = _yearly_mean(denormalized_predicted_ensemble[:, :n_pred])
    test_yearly = _yearly_mean(denormalized_test_ensemble[:, n_pred:])
    pred_output_metrics_yearly = generate_metrics(pred_yearly, test_yearly)
    itself_output_metrics_yearly = generate_metrics(
        _yearly_mean(denormalized_test_ensemble[:, :n_pred]), test_yearly
    )

    if not os.path.exists(
        f"{cfg.paths.metrics}/climatology_metrics/climatology_metrics_{target}_{year}_{start_lat}_{end_lat}_{start_lon}_{end_lon}"
    ):
        print(f"computing climatology metrics for {target}_{year}")
        climatology_ensemble = load_climatology(
            cfg,
            num_members,
            num_rollout_steps,
            denormalized_test_ensemble,
            start_lat,
            end_lat,
            start_lon,
            end_lon,
        )

        climatology_metrics = generate_metrics(
            climatology_ensemble, denormalized_test_ensemble
        )
        # make persistence metrics into usable dict
        climatology_metrics.batch_size = [num_rollout_steps]
        torch.save(
            climatology_metrics,
            f"{cfg.paths.metrics}/climatology_metrics/climatology_metrics_{target}_{year}_{start_lat}_{end_lat}_{start_lon}_{end_lon}",
        )

    # print('denormalized_predicted_ensemble',denormalized_predicted_ensemble)
    # print('denormalized_test_ensemble',denormalized_test_ensemble)
    # print('climatology_ensemble',climatology_ensemble)

    torch.save(
        pred_output_metrics,
        f"{cfg.paths.metrics}/{out_dir}/predicted_output_metrics_{start_lat}_{end_lat}_{start_lon}_{end_lon}",
    )
    torch.save(
        normalized_pred_output_metrics,
        f"{cfg.paths.metrics}/{out_dir}/normalized_predicted_output_metrics_{start_lat}_{end_lat}_{start_lon}_{end_lon}",
    )
    torch.save(
        itself_output_metrics,
        f"{cfg.paths.metrics}/{out_dir}/itself_output_metrics_{start_lat}_{end_lat}_{start_lon}_{end_lon}",
    )
    torch.save(
        pred_output_metrics_yearly,
        f"{cfg.paths.metrics}/{out_dir}/predicted_output_metrics_yearly_{start_lat}_{end_lat}_{start_lon}_{end_lon}",
    )
    torch.save(
        itself_output_metrics_yearly,
        f"{cfg.paths.metrics}/{out_dir}/itself_output_metrics_yearly_{start_lat}_{end_lat}_{start_lon}_{end_lon}",
    )

    # Land-only TAS metrics: mask ocean via TOS (index -1) NaN pattern for fair MESMER comparison
    _ocean = ~denormalized_test_ensemble["surface"][
        0, 0, 0, -1, 0
    ].isnan()  # [nlat, nlon] True=ocean

    def _land_mask_tas(td):
        s = td["surface"].clone()
        s[:, :, :, 10, 0] = s[:, :, :, 10, 0].masked_fill(_ocean, float("nan"))
        out = TensorDict({"surface": s, "level": td["level"]})
        out.batch_size = list(td.batch_size)
        return out

    pred_output_metrics_land = generate_metrics(
        _land_mask_tas(denormalized_predicted_ensemble[:, :n_pred]),
        _land_mask_tas(denormalized_test_ensemble[:, n_pred:]),
    )
    itself_output_metrics_land = generate_metrics(
        _land_mask_tas(denormalized_test_ensemble[:, :n_pred]),
        _land_mask_tas(denormalized_test_ensemble[:, n_pred:]),
    )
    torch.save(
        pred_output_metrics_land,
        f"{cfg.paths.metrics}/{out_dir}/predicted_output_metrics_land_{start_lat}_{end_lat}_{start_lon}_{end_lon}",
    )
    torch.save(
        itself_output_metrics_land,
        f"{cfg.paths.metrics}/{out_dir}/itself_output_metrics_land_{start_lat}_{end_lat}_{start_lon}_{end_lon}",
    )

    pred_output_metrics_land_yearly = generate_metrics(
        _land_mask_tas(pred_yearly), _land_mask_tas(test_yearly)
    )
    itself_output_metrics_land_yearly = generate_metrics(
        _land_mask_tas(_yearly_mean(denormalized_test_ensemble[:, :n_pred])),
        _land_mask_tas(test_yearly),
    )
    torch.save(
        pred_output_metrics_land_yearly,
        f"{cfg.paths.metrics}/{out_dir}/predicted_output_metrics_land_yearly_{start_lat}_{end_lat}_{start_lon}_{end_lon}",
    )
    torch.save(
        itself_output_metrics_land_yearly,
        f"{cfg.paths.metrics}/{out_dir}/itself_output_metrics_land_yearly_{start_lat}_{end_lat}_{start_lon}_{end_lon}",
    )

    # Spatial RMSE maps averaged over all timesteps: shape [num_vars, 1, nlat, nlon]
    _pred_mean = denormalized_predicted_ensemble["surface"][0, :n_pred].mean(dim=0)
    _tgt_mean = denormalized_test_ensemble["surface"][0, n_pred:].mean(dim=0)
    spatial_rmse = (((_pred_mean - _tgt_mean) ** 2).mean(dim=0)).sqrt()
    itself_spatial_rmse = (
        (
            (denormalized_test_ensemble["surface"][0, :n_pred].mean(dim=0) - _tgt_mean)
            ** 2
        ).mean(dim=0)
    ).sqrt()
    torch.save(
        spatial_rmse,
        f"{cfg.paths.metrics}/{out_dir}/predicted_spatial_rmse_{start_lat}_{end_lat}_{start_lon}_{end_lon}",
    )
    torch.save(
        itself_spatial_rmse,
        f"{cfg.paths.metrics}/{out_dir}/itself_spatial_rmse_{start_lat}_{end_lat}_{start_lon}_{end_lon}",
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    start_lat = int(cfg.analysis.start_lat)
    end_lat = None if cfg.analysis.end_lat == "None" else int(cfg.analysis.end_lat)
    start_lon = int(cfg.analysis.start_lon)
    end_lon = None if cfg.analysis.end_lon == "None" else int(cfg.analysis.end_lon)
    run_metrics(
        cfg,
        cfg.analysis.experiment,
        num_members=cfg.analysis.num_members,
        num_rollout_steps=cfg.analysis.num_rollout_steps,
        start_lat=start_lat,
        end_lat=end_lat,
        start_lon=start_lon,
        end_lon=end_lon,
    )


if __name__ == "__main__":
    main()
