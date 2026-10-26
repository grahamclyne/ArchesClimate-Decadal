# Baselines

Code for the two baselines ArchesClimate is compared against in the paper
(Section "Baselines"): **MESMER-M** and **DiffESM**. Both were previously
developed and run in separate directories outside this repository; they are
vendored here (source/config only, generated artifacts excluded) so the full
pipeline used to produce the paper's baseline numbers is reviewable and
reproducible from one place.

## Layout

- `mesmer/` — a copy of the [MESMER](https://github.com/MESMER-group/mesmer)
  package, extended with the DCPP-specific scripts used for this paper:
  `mesmer/mesmer/calibrate_mesmer_m_dcpp.py` and
  `mesmer/mesmer/evaluate_mesmer_m_dcpp.py`.
  These calibrate/evaluate MESMER-M on the same IPSL-DCPP hindcasts and
  train/test split used to train ArchesClimate.
- `diffesm/` — a copy of the [DiffESM](https://github.com/JGCRI/diffesm)
  (Bassetti et al. 2024) codebase, configured to train/generate on IPSL-DCPP
  via `configs/data/IPSL/dcpp/data.json` and `configs/paths/default.yaml`.
  See `diffesm/README.md` for the original setup/train/generate/evaluate
  workflow. Note: the vendored `diffesm/CITATION.cff` points to a now-stale
  `sethbassetti/diff-esm` URL — the repo has since moved to `JGCRI/diffesm`.
  Historical native CMIP5 diagnostic recipes are outside the DCPP paper
  workflow and are excluded from the public export.

## What's excluded, and why

Large generated artifacts (trained checkpoints, generated samples, wandb
logs, hydra run directories, slurm logs) were **not** copied here — they're
multi-GB and regenerable, and are `.gitignore`d project-wide. Local ignored symlinks may point to cluster artifacts; those are absent from a
clone or export. Fetch artifacts separately or regenerate them. Install the
core package base (`pip install /path/to/repository`) in each baseline environment
so the DCPP adapters can resolve `ARCHESCLIMATE_DATA_ROOT` and output roots.

## How this connects to ArchesClimate's evaluation

`ArchesClimate/analysis/gen_baseline_metrics.py` (from the repository root) reads the generated
output of both baselines and computes the same metrics (RMSE, ESR, etc.)
used for ArchesClimate in `analysis/gen_and_save_metrics.py`, so results are
directly comparable in `analysis/plot_metrics.py`. It currently expects:

- MESMER-M emulation files under `ARCHESCLIMATE_MESMER_ROOT`
- DiffESM generated samples under an external `generated_data/IPSL/dcpp`
  directory (`ARCHESCLIMATE_DIFFESM_ROOT`, defaulting to the rollout root plus
  `IPSL/dcpp`).

## DCPP adaptation and validation scope

The submitted comparison uses the DiffESM variant with explicit per-frame
month-of-year conditioning. The original unconditioned variant and its cached
comparison results remain for provenance. MESMER-M and DiffESM use statistics
of the target period as conditioning; ArchesClimate uses preceding states and
external forcing inputs. These protocols differ and are documented in the paper.

Install baselines in their separate upstream environments. The core wheel does
not package them. A source export retains their Python/configuration files and
licenses, but excludes institution-specific shell launchers and generated files.
External generated samples are required to rerun the common metric pipeline;
source-only validation does not rerun baseline training.
