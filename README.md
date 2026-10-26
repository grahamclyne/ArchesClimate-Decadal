# ArchesClimate

Probabilistic decadal climate emulation with a deterministic forecast model and
flow matching, trained on IPSL-CM6A-LR DCPP hindcasts.

- [Installation, configuration, training and inference](ArchesClimate/README.md)
- [Paper figure and table reproduction](ArchesClimate/analysis/README.md)
- [Supported experiments](ArchesClimate/configs/README.md)
- [Data and artifact requirements](ArchesClimate/docs/artifacts.md)
- [MESMER-M and DiffESM baselines](ArchesClimate/baselines/README.md)
- [Development checks](ArchesClimate/docs/development.md)

Use Python 3.11 or 3.12. From the repository root:

```sh
python -m pip install -e '.[train,analysis,data,dev]'
archesclimate-config --cfg job --resolve
```

Data, statistics, trained checkpoints and generated rollouts are external.
Configuration inspection and table assembly from the bundled bootstrap summaries
do not require downloading the full training dataset. See the artifact guide
for the boundary between source availability and end-to-end reproduction.

Original ArchesClimate contributions are licensed under [MIT](LICENSE).
[Third-party notices](THIRD_PARTY_NOTICES.md) retain the separate upstream
licenses, including GPL-3.0 for the vendored MESMER baseline.
Citation metadata is provided in [CITATION.cff](CITATION.cff).
