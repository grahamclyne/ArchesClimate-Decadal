"""Bootstrap confidence intervals for RMSE/ESR over ensemble members and
initialization years, per reviewer_comments.txt Major Comment 8 ("No
uncertainty estimates are given on any of the ablation tables... perhaps by
bootstrapping over members and initialisation years").

Mechanics (see conversation notes / PR description for the full writeup):
  - RMSE is computed from the ensemble MEAN, so you can't bootstrap it by
    resampling already-saved per-year/per-variable numbers after the fact -
    resampling has to happen on the member axis of the pre-aggregation
    ensemble tensors, before generate_metrics() averages over members.
  - generate_metrics() takes target_ensemble (with its own member axis,
    which it averages internally) and input_ensemble (with a member axis,
    which EnsembleMetrics averages internally) - so both axes can be
    resampled just by slicing the member dim of the tensors returned by
    load_predicted_and_target_ensembles() before calling generate_metrics()
    again. No changes to evaluate.py/generate_metrics() were needed.
  - Initialization year isn't an axis inside any one tensor - each year is a
    separate experiment/rollout. So year-resampling means loading several
    years' worth of (predicted, target) tensor pairs up front, then for each
    bootstrap draw, resampling which years' pairs are included (with
    replacement) in addition to resampling members within each.

Usage:
    python bootstrap_metrics.py experiment_years='{0:"flow_..._test_0_None", 10:"flow_..._test_10_None"}'

(passed as a hydra/omegaconf-parseable dict string mapping an arbitrary label
to the experiment folder name under generated_data/, one per available
initialization year - see gen_and_save_metrics.py's `experiment` argument for
the naming convention.)
"""

import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict.tensordict import TensorDict

from ArchesClimate.analysis.evaluate import generate_metrics
from ArchesClimate.analysis.gen_and_save_metrics import (
    load_predicted_and_target_ensembles,
)


def _subset_members(
    tensordict: TensorDict, member_indices, batch_dim: int = 1
) -> TensorDict:
    """Return a copy of tensordict with its member axis (batch_dim) indexed
    by member_indices (may repeat indices, i.e. this is what makes the
    resampling a *bootstrap* rather than a subsample)."""
    idx = torch.as_tensor(member_indices, dtype=torch.long)
    out = TensorDict(
        {
            "surface": tensordict["surface"].index_select(batch_dim, idx),
            "level": tensordict["level"].index_select(batch_dim, idx),
        }
    )
    new_batch_size = list(tensordict.batch_size)
    new_batch_size[batch_dim] = len(member_indices)
    out.batch_size = new_batch_size
    return out


def bootstrap_metrics(
    cfg,
    experiment_years: dict,
    num_members,
    num_rollout_steps,
    n_bootstrap=1000,
    seed=0,
    start_lat=0,
    end_lat=None,
    start_lon=0,
    end_lon=None,
):
    """Compute a bootstrap distribution of generate_metrics() output,
    resampling (with replacement) both which initialization years are
    included and which ensemble members (of both the predicted and the
    target/truth ensembles) are included in each draw.

    Args:
        experiment_years: {year_label: experiment_folder_name} - one entry
            per initialization year that has generated rollouts available.
            year_label is just used for logging; any hashable key works.

    Returns:
        dict with keys 'surface' and 'level', each a stacked tensor of shape
        (n_bootstrap, *metric_shape) - the bootstrap replicates. Percentiles
        of this (e.g. 2.5/97.5 for a 95% CI) are the reviewer-requested
        uncertainty bars for Tables 3-5.
    """
    rng = np.random.default_rng(seed)

    # Load every available year's (predicted, target) tensor pair once up
    # front - resampling later is then just indexing, not re-loading.
    per_year = {}
    for year_label, experiment in experiment_years.items():
        if isinstance(year_label, (tuple, list)):
            cfg.inference.target, cfg.inference.year = year_label[0], int(year_label[1])
        elif isinstance(year_label, int):
            cfg.inference.year = year_label
        pred, target, n_pred, _, _ = load_predicted_and_target_ensembles(
            cfg,
            experiment,
            num_members,
            num_rollout_steps,
            start_lat,
            end_lat,
            start_lon,
            end_lon,
        )
        target = target[
            :, n_pred:
        ]  # score against held-out members only (same as run_metrics)
        n_target = target["surface"].shape[1]
        per_year[year_label] = (pred, target, n_pred, n_target)
        print(
            f"[bootstrap] loaded year={year_label} experiment={experiment}: "
            f"n_pred_members={n_pred}, n_target_members={n_target}"
        )

    year_labels = list(per_year.keys())
    if len(year_labels) < 2:
        print(
            f"[bootstrap] WARNING: only {len(year_labels)} initialization year(s) available "
            "- the year axis of this bootstrap will have essentially no resolution. "
            "Generate more years' rollouts before trusting the resulting CIs."
        )

    surface_reps, level_reps = [], []
    for b in range(n_bootstrap):
        resampled_years = [
            year_labels[i]
            for i in rng.integers(0, len(year_labels), size=len(year_labels))
        ]
        per_draw_surface, per_draw_level = [], []
        for year_label in resampled_years:
            pred, target, n_pred, n_target = per_year[year_label]
            pred_idx = rng.integers(0, n_pred, size=n_pred)
            target_idx = rng.integers(0, n_target, size=n_target)
            pred_sub = _subset_members(pred, pred_idx)
            target_sub = _subset_members(target, target_idx)
            metrics = generate_metrics(pred_sub, target_sub)
            per_draw_surface.append(metrics["surface"])
            per_draw_level.append(metrics["level"])
        # Aggregate across the resampled years the same way the existing
        # pipeline aggregates across the real years for Tables 3-5: average.
        surface_reps.append(torch.stack(per_draw_surface).mean(dim=0))
        level_reps.append(torch.stack(per_draw_level).mean(dim=0))
        if (b + 1) % 100 == 0:
            print(f"[bootstrap] {b + 1}/{n_bootstrap} draws done")

    return {
        "surface": torch.stack(surface_reps),
        "level": torch.stack(level_reps),
    }


def summarize_ci(
    bootstrap_reps: torch.Tensor, point_estimate: torch.Tensor = None, ci=(2.5, 97.5)
):
    """Return (lower, upper) percentile bounds per (rollout_step, variable[, level])
    cell, plus the std, alongside the point estimate if given."""
    lower = torch.quantile(bootstrap_reps, ci[0] / 100, dim=0)
    upper = torch.quantile(bootstrap_reps, ci[1] / 100, dim=0)
    std = bootstrap_reps.std(dim=0)
    out = {"lower": lower, "upper": upper, "std": std}
    if point_estimate is not None:
        out["point_estimate"] = point_estimate
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Example standalone entrypoint. `experiment_years` must be passed as a
    hydra override, e.g.:
        python bootstrap_metrics.py \
            +experiment_years='{0: "flow_ensemble_single_small_0_24_5_1_118_1_test_0_None", \
                                 10: "flow_ensemble_single_small_0_24_5_1_118_1_test_10_None"}' \
            +n_bootstrap=1000
    """
    num_members = cfg.inference.num_members
    num_rollout_steps = cfg.inference.num_rollout_steps
    experiment_years = OmegaConf.to_container(cfg.experiment_years, resolve=True)
    n_bootstrap = cfg.get("n_bootstrap", 1000)

    reps = bootstrap_metrics(
        cfg, experiment_years, num_members, num_rollout_steps, n_bootstrap=n_bootstrap
    )

    out_dir = (
        f"{cfg.paths.metrics}/bootstrap_{cfg.name}_{num_members}_{num_rollout_steps}"
    )
    os.makedirs(out_dir, exist_ok=True)
    torch.save(reps, os.path.join(out_dir, "bootstrap_replicates.pt"))

    surface_ci = summarize_ci(reps["surface"])
    level_ci = summarize_ci(reps["level"])
    torch.save(
        {"surface": surface_ci, "level": level_ci},
        os.path.join(out_dir, "bootstrap_ci.pt"),
    )
    print(f"[bootstrap] wrote {n_bootstrap} replicates and CI summary to {out_dir}")


if __name__ == "__main__":
    main()
