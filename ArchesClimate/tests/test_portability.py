"""Lightweight tests: no training dependencies or external data needed."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from ArchesClimate.paths import config_directory, path, relocate_model_config


class PortabilityTests(unittest.TestCase):
    def test_environment_roots_and_override(self):
        with patch.dict(
            os.environ,
            {
                "ARCHESCLIMATE_DATA_ROOT": "/tmp/example-data",
                "ARCHESCLIMATE_MODELS_ROOT": "/tmp/models",
            },
            clear=True,
        ):
            self.assertEqual(path("stats"), Path("/tmp/example-data/stats").resolve())
            self.assertEqual(path("models"), Path("/tmp/models").resolve())

    def test_all_experiments_resolve_from_installed_resources(self):
        with (
            patch.dict(
                os.environ, {"ARCHESCLIMATE_DATA_ROOT": "/tmp/data"}, clear=True
            ),
            initialize_config_dir(
                version_base=None, config_dir=str(config_directory())
            ),
        ):
            configs = list((config_directory() / "module").glob("*.yaml"))
            self.assertEqual(len(configs), 38)
            for module in configs:
                cfg = compose(config_name="config", overrides=[f"module={module.stem}"])
                OmegaConf.to_container(cfg, resolve=True)
                self.assertEqual(cfg.dataloader.dataset.stats_path, "/tmp/data/stats")

    def test_moved_model_and_prior_locations(self):
        cfg = OmegaConf.create(
            {
                "model_dir": "/old/model",
                "module": {
                    "module": {
                        "load_deterministic_model": ["/old/model_output/project/prior"]
                    }
                },
            }
        )
        with patch.dict(os.environ, {"ARCHESCLIMATE_MODELS_ROOT": "/tmp/new-models"}):
            moved = relocate_model_config(cfg, "/tmp/moved")
        self.assertEqual(moved.model_dir, str(Path("/tmp/moved").resolve()))
        self.assertEqual(
            moved.module.module.load_deterministic_model[0],
            str(Path("/tmp/new-models/project/prior").resolve()),
        )
        self.assertEqual(cfg.model_dir, "/old/model")


if __name__ == "__main__":
    unittest.main()
