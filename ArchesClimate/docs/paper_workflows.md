# Paper workflow inventory

This inventory defines the scope of the pruning pass. It preserves all current
main-text and appendix result workflows, their supporting DCPP code, and the
archived comparison results used by the table assembly scripts.

## Entry points

| Workflow | Entry points | Dependencies retained |
|---|---|---|
| Training and sampling | `submit.py`, `main_hydra.py`, `analysis/long_rollout.py` | DCPP model/encoder, base module, backbones, shared metrics/utilities, DCPP loader and memmap reader |
| Deterministic table ablations | `analysis/bootstrap_table3.py`, `rerun_table3_metrics.py`, `analyze_table3_bootstrap.py` | `table3_*`, deterministic reference configs, `bootstrap_table3_results_11/` |
| Generative tables | `analysis/bootstrap_table45.py`, `bootstrap_diffesm.py`, `bootstrap_mesmer.py`, `analyze_table45_bootstrap.py` | Six corrected flow variants, deterministic priors, both baseline implementations, `bootstrap_table45_results/` |
| Alternative training splits | `analysis/compute_alt_years_comparison_v2.py`, `bootstrap_a2_flow.py` | `forecast_added_input_state`, `flow_ensemble_single_not_spectral`, `dcpp_alt` / `dcpp_gap` loader filters, `bootstrap_a2_flow_results/` |
| Pooled climatology | `analysis/plot_climatology.py` | Corrected model rollouts, all eleven target starts, baseline samples, normalization statistics |
| Pooled rank histograms and temporal spectra | `analysis/regen_psd_rankhist.py` | All eleven starts, six corrected flow models, baseline samples; this is the paper entry point, not the single-start path in `plot_metrics.py` |
| Regional raw/anomaly rollouts | `analysis/regional_analysis.py` | Target and prediction ensembles for test starts 0/10 and validation start 0 |
| Forced/flat 50-year rollout | `analysis/plot_50yr_anomaly.py` | Existing forced/flat 600-month chunk files, real 1970/2010 hindcasts, original flow normalization |
| Supporting diagnostics | `analysis/plot_thetaot2000_conditioning.py`, `plot_data.py`, `plot_metrics.py` | Retained while conditioning/control provenance is reviewed |
| Data and normalization preparation | `preprocessing/preprocess_raw_ipsl_dcpp.py`, `utils/convert_xarray_to_memmap.py`, `generate_delta_stats_clean.py`, target-ensemble helpers | Original DCPP variable order, grids, forcings and statistics |

Paths in this table are relative to `ArchesClimate/`. See
[the analysis guide](../analysis/README.md) for per-figure details and
[the configuration guide](../configs/README.md) for retained model recipes.
The data illustration and training/sampling diagrams are hand-made assets.

## Scope decisions

The cleanup removes 146 configs selecting the superseded `model.forecast`,
`model.diffusion`, or `model.consistency` implementations, together with their
unused support code and unrelated CMIP interpolation/ocean preprocessing
launchers. These are separate implementations from the paper's
`model.dcpp_forecast` and `model.dcpp_diffusion`.

Before pruning, Python imports (including relative imports), Hydra targets,
config defaults, analysis entry points, shell launchers, and available saved
DCPP model configs were checked. All 45 readable saved configs (42
`hydra_config.yaml` files plus three valid `config.yaml` aliases/files) selected
the retained DCPP models, encoder, and backbone. All 38 retained model
configurations were composed and resolved before pruning and compared with the
post-cleanup configurations. Retained scientific Python files and model configs
are unchanged in this pass.

The prior source is recoverable from commit `732c43d`. Generated Hydra output,
notebook autosaves, scheduler/profiler records, installation metadata, old
manuscript archives and LaTeX build files are untracked and ignored while their
local copies remain available. Small bootstrap result summaries remain tracked
because the table assemblers use them. Upstream baseline code, notices and
licenses remain intact; no source data, statistics, checkpoints or generated
rollouts are deleted by this pass.

## Deliberately retained pending further review

- All DCPP model variants: a config absent from the headline plots can still
  document an ablation or an archived checkpoint.
- Reference/alternative bootstrap outputs, including the old-control and 3D
  fallback results: current analysis code still references them.
- Remaining forcing preprocessing and diagnostic tools: their data provenance
  warrants a separate review before removal.
- Baseline upstream source and native-reproduction diagnostics: separate
  environments and artifact provenance still need clean-checkout validation.

## Portability

Artifact locations are configurable through environment variables. Configuration
resources are included in the installed package; the default compute environment
is local. See [development checks](development.md) and
[artifact requirements](artifacts.md). Data and checkpoints remain external.
Configuration composition and code tests do not establish full training or
figure reproduction on a new machine.
