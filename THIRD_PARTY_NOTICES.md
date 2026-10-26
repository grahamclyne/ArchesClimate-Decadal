# Third-party code and attribution

The root MIT license covers original ArchesClimate contributions. It does not
replace third-party licenses or apply to external datasets or checkpoints.

| Component | Source | License / notice |
|---|---|---|
| WeatherLearn utilities and adapted Pangu layers (`ArchesClimate/backbones/weatherlearn_utils/`, `archesweather_layers.py`) | [WeatherLearn](https://github.com/lizhuoq/WeatherLearn) | MIT, Copyright 2023 Zhuoqun Li; [full notice](licenses/WeatherLearn-MIT.txt) |
| Sinusoidal timestep embedding in `backbones/dit.py` | [GLIDE](https://github.com/openai/glide-text2im) | MIT, Copyright 2021 OpenAI; [full notice](licenses/GLIDE-MIT.txt) |
| Ensemble-score implementation referenced in `metrics/ensemble_metrics.py` | [WeatherBench 2](https://github.com/google-research/weatherbench2) | Apache-2.0; [full license](licenses/WeatherBench2-Apache-2.0.txt). ArchesClimate adapts the metrics for climate-ensemble verification. |
| DiffESM source and DCPP adapters | [JGCRI/DiffESM](https://github.com/JGCRI/diffesm) | [Upstream MIT license](ArchesClimate/baselines/diffesm/LICENSE), Copyright 2023 Joint Global Change Research Institute |
| MESMER source and DCPP adapters | [MESMER](https://github.com/MESMER-group/mesmer) | [Upstream GPL-3.0 license](ArchesClimate/baselines/mesmer/LICENSE); retained separately from the core package |

Source comments also credit the Pangu-Weather pseudocode and the ICNR
initialization reference used by the WeatherLearn-derived layers. Those
attributions are retained. No pretrained Pangu-Weather weights are distributed.

Baseline packages use their own environments and retain their upstream notices.
The core wheel excludes baseline packages. Redistribute third-party components
with their respective notices; the root MIT license does not relicense them.
