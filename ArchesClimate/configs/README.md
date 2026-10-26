# Configuration scope

The retained model configurations use `model.dcpp_forecast` or
`model.dcpp_diffusion`. The default loader is `dataloader/dcpp.yaml`; the input
is the prepared IPSL-DCPP memmap dataset. Artifact paths use environment variables; compute settings default to local.
Use `cluster=slurm` with your own scheduler account and resources when needed.

Install the package from the repository root. Launch commands from any directory:

```sh
archesclimate-submit module=table3_2d768_cl_c name=<new_deterministic_run>
```

For a generative run, first supply the deterministic-prior directories in
`module.module.load_deterministic_model`, including their saved configs and
checkpoints. Preserve the prior and normalization choices belonging to the
experiment being reproduced. Do not substitute an arbitrary newer checkpoint.

```sh
archesclimate-submit module=flow_ensemble_single_small name=<new_flow_run>
```

These examples are launch templates; they require the data, statistics, prior
artifacts, and cluster setup described in the project README.

| Family | Configuration names/patterns | Role |
|---|---|---|
| Deterministic table ablations | `table3_*` | Explicit architecture, composite-loss, and conditioning variants |
| Deterministic prior/reference recipes | `forecast_added_input_state*` | Original priors, controls, and retained DCPP ablations |
| Generative paper variants | `flow_ensemble_small`, `flow_ensemble_large`, `flow_ensemble`, `flow_ensemble_single_small`, `flow_ensemble_single_not_spectral`, `flow_3d_prior` | Six corrected generative columns |
| Additional DCPP variants | Other retained `flow_ensemble*` and `flow_3d_prior*` | Retained pending finer experiment/checkpoint provenance review |

Alternative train/test splits are selected through
`dataloader.dataset.filename_filter_type=dcpp_alt` or `dcpp_gap`, as used by the
analysis scripts; they do not require the superseded `flow_alt_years` model
configuration. Inference settings remain under `inference/`; plot/evaluation
settings remain under `analysis/`.

## Retained model configurations

- `flow_3d_prior`
- `flow_3d_prior_ensemble`
- `flow_3d_prior_small`
- `flow_3d_prior_small_ensemble`
- `flow_3d_prior_small_state_pred`
- `flow_ensemble`
- `flow_ensemble_alpha_point_0`
- `flow_ensemble_alpha_point_3`
- `flow_ensemble_alpha_point_5`
- `flow_ensemble_alpha_point_7`
- `flow_ensemble_default_delta`
- `flow_ensemble_large`
- `flow_ensemble_large_with_spectral`
- `flow_ensemble_large_with_spectral_and_grad`
- `flow_ensemble_large_with_spectraland_grad`
- `flow_ensemble_real`
- `flow_ensemble_single_not_spectral`
- `flow_ensemble_single_small`
- `flow_ensemble_small`
- `flow_ensemble_small_no_spectral`
- `flow_ensemble_small_with_spectral`
- `forecast_added_input_state`
- `forecast_added_input_state_3d`
- `forecast_added_input_state_small`
- `forecast_added_input_state_small_3d`
- `forecast_added_input_state_small_no_conditioning`
- `forecast_added_input_state_small_no_conditioning_no_year`
- `forecast_added_input_state_small_no_spectral`
- `forecast_added_input_state_small_no_ssi`
- `forecast_added_input_state_small_no_year`
- `forecast_added_input_state_with_spectral`
- `table3_2d192_cl_c`
- `table3_2d192_ncl_c`
- `table3_2d768_cl_c`
- `table3_2d768_cl_nc`
- `table3_2d768_ncl_c`
- `table3_3d192_cl_c`
- `table3_3d768_cl_c`

All 38 retained configurations were compared before and after pruning. The
configuration values and their first-party Hydra targets are preserved. This
scope check does not establish that every retained experiment is a final paper
result; uncertain DCPP variants are kept until their checkpoint lineage is
fully documented.
