# Development checks

Run these commands from the repository root in the project's training
environment. `requirements-dev.txt` pins code-quality tooling; it does not
replace the model and analysis dependencies.

```sh
python -m pip install -r requirements-dev.txt
ruff check .
ruff format --check .
python -m unittest discover -s ArchesClimate/tests -v
```

Use `ruff format .` to format first-party Python files. Ruff is pinned to
`0.16.9` in both `requirements-dev.txt` and `pyproject.toml`; its default
correctness rules and import sorting are enabled. Vendored MESMER/DiffESM
source, manuscript directories, generated Hydra outputs, and notebook
autosaves are excluded. The upstream baseline environments retain their own
formatting conventions.

The established `ArchesClimate` package name is intentionally preserved for
checkpoint/import compatibility. Monthly plotting dates intentionally have no
timezone; their two Ruff exceptions are documented at the call sites. The
interactive debugger remains an explicit `debug=True` opt-in in the training
launcher and has a local exception. No project-wide lint ignore list is used.

## Regression coverage

The checked-in unit tests cover the ensemble-mean SSR normalization, metric logging across successive calls and
across model instances, and the rejection of unsupported training-launcher
modes before any data is accessed. Logging previously reused a mutable default
dictionary and could replay stale metric names. Test mode previously referenced
an undefined loader and had no matching model test step. For paper inference,
use `analysis/long_rollout.py` and the analysis scripts instead.

The lint/format cleanup also removes unused, broken legacy NetCDF conversion
and plotting helpers; binds the diffusion scheduler's loop index explicitly;
uses independent constructor defaults; and makes optional-analysis failures
include their exception context. Scalar NaN checks retain their original
handling of infinities and missing values.

For the cleanup validation on Jean Zay, the current table 3 and table 4/5
assembly output and a normalized real DCPP data sample are compared before and
after the edit. Configuration composition, metric/normalization source
comparisons, calendar rollover, scheduler-wrapper behavior, and syntax checks
supplement those regressions. These checks do not replace a full training or
GPU rollout reproduction; packaging and artifact-path tests additionally cover configuration resources,
environment overrides, moved model metadata, and export exclusions. Full GPU
training and figure regeneration still require the external artifacts.

For a lightweight install check, build a wheel, install it into a fresh Python
3.11/3.12 environment, and run `archesclimate-config --cfg job --resolve` from
outside the checkout. The wheel includes configuration YAML and the small
bootstrap summaries; statistics and checkpoint files stay external.

The GitHub Actions workflow validates packaging/configuration on Python 3.11
and 3.12 and runs the full CPU regression suite on Python 3.11. The training
extra includes xarray because the shared metrics import it. Path assertions
compare resolved paths so macOS temporary-directory symlinks are supported.
