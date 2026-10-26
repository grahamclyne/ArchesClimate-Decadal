# ArchesClimate

ArchesClimate is a deep learning climate model emulator trained on IPSL-CM6A-LR
decadal hindcasts (IPSL-DCPP), combining a deterministic forecasting model with
a flow-matching generative model (following ArchesWeatherGen) to emulate
internal climate variability at monthly-to-decadal timescales at a fraction of
the cost of running the physics-based GCM.

## Repository layout

- `dataloaders/` — IPSL-DCPP dataset loading, normalization, and
  train/val/test split logic (`dataloaders/dcpp.py`).
- `model/` — the deterministic forecasting model (`dcpp_forecast.py`) and the
  flow-matching generative model (`dcpp_diffusion.py`), plus the shared
  Lightning module base classes.
- `backbones/` — the underlying network architectures (ArchesWeather-style
  transformer backbone, axial-attention variant, DiT).
- `metrics/` — ensemble evaluation metrics: fair RMSE, spread-skill ratio
  (SSR), ensemble spread ratio (ESR), CRPS, energy score.
- `configs/` — Hydra configs. `module/` defines each trained model variant
  (architecture + training hyperparameters, e.g. `flow_ensemble_single_small`,
  `forecast_added_input_state`); `dataloader/`, `inference/`, `analysis/`, and
  `cluster/` configure the data split, rollout/sampling settings, evaluation
  scripts, and compute environment (Jean Zay account/QOS/module versions)
  respectively.
- `analysis/` — evaluation and figure-generation scripts: rollout generation
  (`long_rollout.py`), metric computation (`gen_and_save_metrics.py`,
  `gen_baseline_metrics.py` for the baselines), bootstrap uncertainty
  estimation (`bootstrap_table3.py`, `bootstrap_table45.py`, and variants),
  and plotting (`plot_metrics.py`, `plot_climatology.py`, `regional_analysis.py`).
- `baselines/` — vendored, in-repo copies of the two baselines ArchesClimate
  is compared against (MESMER-M and DiffESM); see `baselines/README.md`.
- `preprocessing/` — scripts that build the training dataset (memmap
  conversion, forcing time series, climatology) from raw IPSL-DCPP
  NetCDF output.
- `utils/` — dataset archival, delta-stats (normalization) generation, and
  other one-off data-preparation scripts.
- `stats/` — external normalization statistics required for training and inference.
  These files are intentionally excluded from Git. `utils/download_stats_from_hub.py`
  provides the artifact-download entry point; check the selected artifact contains
  the statistics required by your configuration before running a model.

## Setup

Use Python 3.11 or 3.12 in a fresh virtual environment, from the repository root:

```sh
python -m pip install -e '.[train,analysis,data,dev]'
export ARCHESCLIMATE_DATA_ROOT=/absolute/path/to/data
export ARCHESCLIMATE_WORK_ROOT=/absolute/path/to/artifacts
archesclimate-config --cfg job --resolve
```

The base installation (`pip install .`) supports configuration inspection.
Training, analysis and data downloads require the corresponding extras. Baselines
use their separate upstream environments. CUDA drivers and the PyTorch build
must match the target machine; the dependency ranges are not an environment lock.

Training data belongs in `$ARCHESCLIMATE_DATA_ROOT/dcpp_dataset_memmap`,
forcings in `reference_data`, and normalization files in `stats`. Obtain those
external artifacts before training. `archesclimate-stats --directory /path/to/stats`
is an optional Hub download helper; verify the chosen repository and revision
contain the statistics required by the experiment.

The normalization bundle needs `dcpp_stats.pt` and
`dcpp_clim_removed_norm_stats.pt`. Flow recipes additionally need their selected
`dcpp_surface_delta_std` / `dcpp_plev_delta_std` or model-specific
`dcpp_delta_std_<prior-name>` files. Figure metric computation also requires
`climatology_stats_net_flux.pt` in the statistics root. Check these files before
running the corresponding workflow; the download helper does not guarantee
that an arbitrary artifact revision contains the complete bundle.

`ARCHESCLIMATE_<NAME>_ROOT` overrides an individual artifact root. Supported names
are `STATS`, `FORCINGS`, `MODELS`, `ROLLOUTS`, `TARGETS`, `LONG_ROLLOUTS`, `PLOTS`,
`METRICS`, `RESULTS`, `LOGS`, `DIFFESM`, `MESMER`, `RAW` and `ARCHIVE`. Defaults
are defined in `paths.py`; data defaults to `./data`, outputs to `./artifacts`.
Set environment variables before starting Python. `ARCHESCLIMATE_DEVICE=cpu`
selects CPU for bootstrap analysis. Existing checkpoint directories may be moved;
the loader uses their current location and relocates saved prior model paths.

The default cluster is local, with logging disabled. Use `cluster=slurm` and set
`ARCHESCLIMATE_SLURM_ACCOUNT`, then override partition, devices and resources
for your site.

## Training

Training is launched via `submit.py`, which submits a Hydra-configured job to
SLURM through `submitit`:

```bash
archesclimate-submit module=<module_config_name> name=<run_name> [overrides...]
```

`module` selects the architecture/training variant from `configs/module/`
(e.g. `forecast_added_input_state` for the deterministic model,
`flow_ensemble_single_not_spectral` for a generative model conditioned on a
single deterministic prior). `name` sets the run's output directory under
`ARCHESCLIMATE_MODELS_ROOT/<project>/<name>/` and its Weights & Biases run name. Training
resumes automatically from the latest checkpoint if a run with the same name
already has one (`resume: true` by default in `configs/config.yaml`).

## Inference and evaluation

- **Rollout generation**: `analysis/long_rollout.py` (installed command `archesclimate-rollout`), configured via `inference=<inference_config>` and
  `inference.target`/`inference.year` to select which held-out hindcast start
  to roll out from.
- **Metrics**: `analysis/gen_and_save_metrics.py` computes and caches the
  metrics (RMSE, SSR, ESR, CRPS, ...) used throughout the paper's tables and
  figures from a completed rollout; `analysis/gen_baseline_metrics.py` does
  the same for the MESMER-M/DiffESM baselines.
- **Bootstrap uncertainty**: `analysis/bootstrap_table3.py` and
  `analysis/bootstrap_table45.py` (plus the baseline-specific variants,
  `bootstrap_mesmer.py`/`bootstrap_diffesm.py`) resample over ensemble members
  and hindcast starts to give confidence intervals for each reported metric;
  `analyze_table3_bootstrap.py`/`analyze_table45_bootstrap.py` assemble the
  per-model bootstrap outputs into the paper's tables.
- **Figures**: `analysis/regen_psd_rankhist.py` produces the paper's pooled rank
  histogram and temporal spectrum figures; `plot_climatology.py`,
  `regional_analysis.py`, and `plot_50yr_anomaly.py` produce the remaining
  scripted figures. See [analysis/README.md](analysis/README.md) for the exact mapping.

## Baselines

MESMER-M and DiffESM are vendored under `baselines/`, configured to
train/evaluate on the same IPSL-DCPP hindcasts and train/test split as
ArchesClimate. See `baselines/README.md` for details, including which
generated artifacts are excluded from version control and how to regenerate
them.

## Supported scope

This repository provides the IPSL-DCPP paper workflows and their ablations.
See the [workflow inventory](docs/paper_workflows.md) and
[configuration guide](configs/README.md) for the retained entry points.

## Development checks

See [docs/development.md](docs/development.md) for the pinned Ruff setup,
lint/format commands, and regression tests.

## License and external artifacts

Original contributions use the [MIT license](../LICENSE); see the separate
[third-party notices](../THIRD_PARTY_NOTICES.md) for vendored and adapted code.
The [artifact guide](docs/artifacts.md) lists the requirements for training and
reproducing data-driven results. Source release does not include data or weights.
