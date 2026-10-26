# Data, checkpoints and reproduction requirements

The source export contains no training data, normalization tensors, pretrained
weights, target ensembles or generated rollouts. Paths are configurable through
`ARCHESCLIMATE_DATA_ROOT`, `ARCHESCLIMATE_WORK_ROOT` and the per-artifact overrides
in [the setup guide](../README.md).

| Workflow | Required external artifacts |
|---|---|
| Configuration inspection | None |
| Assemble published tables | Bundled `analysis/bootstrap*results*/*.npz`; NumPy |
| Train a deterministic model | Processed DCPP memmaps, forcing time series, normalization statistics |
| Train a flow model | Deterministic prerequisites, selected deterministic prior checkpoint(s), residual scalers |
| Generate ensembles | Selected trained model and prior checkpoints, initial states, forcings and statistics |
| Regenerate climatology, rank/spectrum and regional plots | Target and generated ensembles, baseline samples where plotted, figure statistics |
| Regenerate 50-year plot | Forced/flat 600-month rollout chunks and reference hindcasts |

## Statistics

The optional `archesclimate-stats` helper defaults to the Hugging Face dataset
`grahamclyne/ArchesClimate-Decadal-stats`. Supply `--revision` to pin a known
artifact revision and `--directory` to choose its destination. An anonymous access
check on 25 October 2026 returned HTTP 401, so this endpoint is not currently
a verified public download. Authorized users may authenticate with the Hub;
public reproduction needs an accessible artifact release. This helper is
not a downloader for the full training dataset or trained models.

Required files include `dcpp_stats.pt`, `dcpp_clim_removed_norm_stats.pt`, the
selected residual scaler(s), and, for figure metrics,
`climatology_stats_net_flux.pt`. Model-specific scalers are not interchangeable.
Verify availability and hashes of the exact chosen bundle before a full run.

## Reproduction scope

The [analysis guide](../analysis/README.md) maps data-driven figures and tables
to scripts. The data illustration and diagrams use offline artwork, so their
imagery is not generated from climate data by the analysis pipeline.

No public, versioned download bundle for all training data, checkpoints and
rollouts is declared here. End-to-end reproduction on a new system therefore
requires arranging access to those artifacts or regenerating them. This is a
remaining artifact-release task, not a claim that the external files are absent
from the authors' storage. The MIT code license does not grant rights to
redistribute third-party climate datasets.
