"""Artifact locations shared by Hydra configs and standalone paper scripts."""

import os
from pathlib import Path


def config_directory():
    """Installed YAML resources; independent of the shell working directory."""
    return Path(__file__).resolve().parent / "configs"


def path(name):
    """Resolve a named artifact root without touching the filesystem.

    Set ARCHESCLIMATE_<NAME>_ROOT before running any command. Data and work
    default to ./data and ./artifacts. Child roots inherit those settings.
    """
    key = "ARCHESCLIMATE_" + name.upper() + "_ROOT"
    if value := os.environ.get(key):
        return Path(value).expanduser().resolve()
    if name in {"data", "work"}:
        return Path("data" if name == "data" else "artifacts").resolve()
    parent, child = {
        "stats": ("data", "stats"),
        "forcings": ("data", "reference_data"),
        "raw": ("data", "batch_raw"),
        "models": ("work", "model_output"),
        "rollouts": ("work", "generated_data"),
        "targets": ("rollouts", ""),
        "long_rollouts": ("rollouts", ""),
        "plots": ("work", "plots"),
        "metrics": ("work", "computed_metrics"),
        "results": ("work", "results"),
        "logs": ("work", "sblogs"),
        "diffesm": ("rollouts", "IPSL/dcpp"),
        "mesmer": ("work", "mesmer/emulations"),
        "archive": ("work", "archive"),
    }[name]
    return path(parent) / child


def reference_results(group):
    """Prefer regenerated results, otherwise use the packaged paper summaries."""
    generated = path("results") / group
    if os.environ.get("ARCHESCLIMATE_RESULTS_ROOT") or generated.exists():
        return generated
    return Path(__file__).resolve().parent / "analysis" / group


def analysis_device():
    """Default to CUDA when available; allow explicit CPU analysis."""
    import torch

    return os.environ.get(
        "ARCHESCLIMATE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
    )


def relocate_model_config(cfg, model_directory):
    """Use a moved checkpoint and move legacy prior paths under the model root."""
    from omegaconf import OmegaConf, open_dict

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(cfg):
        cfg.model_dir = str(Path(model_directory).resolve())
        cfg.exp_dir = cfg.model_dir
    module = cfg.module.module
    priors = module.get("load_deterministic_model")
    if priors:

        def relocated(value):
            value = str(value)
            if "/model_output/" in value:
                return str(path("models") / value.split("/model_output/", 1)[1])
            return value

        module.load_deterministic_model = (
            [relocated(v) for v in priors]
            if not isinstance(priors, str)
            else relocated(priors)
        )
    return cfg
