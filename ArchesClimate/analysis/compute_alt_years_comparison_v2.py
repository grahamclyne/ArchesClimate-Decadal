"""Comment #9, corrected: compare RMSE of the original (overlapping-year)
split's deterministic model against the two genuinely held-out splits
(alt_years, gap), all evaluated against the SAME real calendar period.

v1 (compute_alt_years_comparison.py) was wrong in two ways:
  1. It compared each model's rollout against its own "val" split -- but
     "val" means a different real calendar period per filter type (original:
     2010-2016, alt_years: 2000-2010, gap: 2005-2010).
  2. The ground-truth ensemble file loaded for comparison
     ({split}_ensemble_0_flow_net_flux_1) is NOT filter-type aware -- it's a
     single fixed file anchored to the default "dcpp" filter's real years
     (~2010-2020, since each hindcast run extends 10yr past its init year).
     Comparing predictions from a genuinely different real period against
     this fixed truth (by array index, not calendar date) created a
     fake-looking "divergence" that was actually just a growing offset
     between two different real time periods under accelerating real-world
     forcing -- not a model failure.

Fix: evaluate alt_years/gap on their "test" split (test_filter=range(2010,2016)
for both -- identical to the original "dcpp" filter's val_filter range(2010,2016),
same underlying real years/members), and compare ALL THREE against the same
confirmed-correct truth file (val_ensemble_0_flow_net_flux_1, verified against
the near-perfect original-model match).
"""

import math

import hydra
import torch
from hydra import compose, initialize_config_dir

from ArchesClimate.analysis.evaluate import generate_metrics
from ArchesClimate.analysis.gen_and_save_metrics import load_experiment
from ArchesClimate.paths import config_directory
from ArchesClimate.paths import path as artifact_path

CONFIGS_DIR = str(config_directory())
METRICS = str(artifact_path("metrics"))
NUM_MEMBERS = 5
NUM_ROLLOUT_STEPS = 118

EXPERIMENTS = {
    "original (val, real 2010-2016)": {
        "name": "forecast_added_input_state_0",
        "filt": "dcpp",
        "target": "val",
        "inf_cfg": "inference_10",
    },
    "alt_years (test, real 2010-2016)": {
        "name": "forecast_added_input_state_alt_years",
        "filt": "dcpp_alt",
        "target": "test",
        "inf_cfg": "inference_10",
    },
    "gap (test, real 2010-2016)": {
        "name": "forecast_added_input_state_gap",
        "filt": "dcpp_gap",
        "target": "test",
        "inf_cfg": "inference_10",
    },
}

SURFACE_VAR_ORDER = ["tas", "thetaot2000", "tos", "psl", "net_flux", "pr"]
LEVEL_VAR_ORDER = [("hur", 1), ("ta", 1), ("wap", 1), ("zg", 1)]
PLEV_LABELS = {0: "850hPa", 1: "700hPa", 2: "500hPa", 3: "250hPa"}


def get_cfg(name, filt, target, inf_cfg):
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        return compose(
            config_name="config",
            overrides=[
                "module=forecast_added_input_state",
                f"inference={inf_cfg}",
                f"name={name}",
                f"dataloader.dataset.filename_filter_type={filt}",
                f"inference.target={target}",
            ],
        )


# ground truth: SAME file for all three (confirmed real 2010-2020 DCPP data)
_ref_cfg = get_cfg("forecast_added_input_state_0", "dcpp", "val", "inference_10")
train_spatial_norm = hydra.utils.instantiate(
    _ref_cfg.dataloader.dataset,
    domain="all",
    limit_examples=100,
    norm_scheme="spatial_norm",
)
train_mean_only_spatial_norm = hydra.utils.instantiate(
    _ref_cfg.dataloader.dataset,
    domain="all",
    limit_examples=100,
    norm_scheme="mean_only_spatial_norm",
)

_truth_raw = torch.load(
    str(artifact_path("targets") / "val_ensemble_0_flow_net_flux_1")
)
_truth_raw["surface"] = _truth_raw["surface"][
    :, :, _ref_cfg.module.surface_variable_indices
]
_truth_raw = _truth_raw[:, :NUM_ROLLOUT_STEPS]
truth_denorm = train_spatial_norm.denormalize(_truth_raw[None], stateless=True)[
    :, :, :NUM_ROLLOUT_STEPS
]
truth_denorm.batch_size = [1, _truth_raw.shape[0], NUM_ROLLOUT_STEPS]

SURFACE_VARIABLES = _ref_cfg.module.surface_variables
LEVEL_VARIABLES = _ref_cfg.module.level_variables

results = {}
for label, spec in EXPERIMENTS.items():
    cfg = get_cfg(spec["name"], spec["filt"], spec["target"], spec["inf_cfg"])
    experiment = f"{cfg.name}_{cfg.inference.num_inference_steps}_{cfg.inference.num_members}_{cfg.inference.scale_input_noise}_{cfg.inference.num_rollout_steps}_{cfg.inference.cf_guidance}_{cfg.inference.target}_{cfg.inference.year}_{cfg.inference.ckpt_fname}"
    print(label, "->", experiment)
    pred_raw = load_experiment(cfg, experiment, NUM_MEMBERS, NUM_ROLLOUT_STEPS)
    pred_denorm = train_mean_only_spatial_norm.denormalize(
        pred_raw[None], stateless=True
    )[:, :NUM_MEMBERS, :NUM_ROLLOUT_STEPS]
    pred_denorm.batch_size = [1, NUM_MEMBERS, NUM_ROLLOUT_STEPS]

    metrics = generate_metrics(pred_denorm, truth_denorm)
    results[label] = metrics

MSE_IDX = 0


def rmse_per_var(metrics_td, kind, vi, pi=None):
    t = metrics_td[kind]
    mse_series = t[:, MSE_IDX, vi] if kind == "surface" else t[:, MSE_IDX, vi, pi]
    mse_avg = mse_series.nanmean(dim=0).item()
    return mse_avg**0.5 if not math.isnan(mse_avg) and mse_avg >= 0 else float("nan")


rows = []
for vname in SURFACE_VAR_ORDER:
    rows.append((vname, "surface", SURFACE_VARIABLES.index(vname), None))
for vname, plev_idx in LEVEL_VAR_ORDER:
    rows.append(
        (
            f"{vname} {PLEV_LABELS[plev_idx]}",
            "level",
            LEVEL_VARIABLES.index(vname),
            plev_idx,
        )
    )

exp_labels = list(EXPERIMENTS.keys())
col_w = max(len(l) for l in exp_labels) + 2
print(
    "\n=== RMSE (sqrt of MSE), CORRECTED: all three vs. same real 2010-2016 truth ===\n"
)
header = f"{'variable':<20}" + "".join(f"{l:>{col_w}}" for l in exp_labels)
print(header)
print("-" * len(header))
for label, kind, vi, pi in rows:
    vals = [rmse_per_var(results[e], kind, vi, pi) for e in exp_labels]
    print(f"{label:<20}" + "".join(f"{v:>{col_w}.4g}" for v in vals))

torch.save(results, f"{METRICS}/alt_years_comparison_v2_results.pt")
print(f"\nSaved raw metrics to {METRICS}/alt_years_comparison_v2_results.pt")
