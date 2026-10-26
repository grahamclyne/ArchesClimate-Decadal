"""The training launcher rejects unsupported modes before accessing artifacts."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from ArchesClimate.main_hydra import main
from ArchesClimate.model.base_module import load_module
from ArchesClimate.paths import config_directory


class TrainingEntrypointTests(unittest.TestCase):
    def test_test_mode_points_to_the_supported_inference_pipeline(self):
        with self.assertRaisesRegex(ValueError, "analysis/long_rollout.py"):
            main.__wrapped__(SimpleNamespace(mode="test"))

    def test_offline_training_saves_reload_metadata_and_checkpoint_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            with initialize_config_dir(
                version_base=None, config_dir=str(config_directory())
            ):
                cfg = compose(
                    config_name="config", overrides=["name=offline-test", "log=False"]
                )
            cfg.exp_dir = str(Path(tmp) / "run")
            cfg.model_dir = cfg.exp_dir
            with (
                patch("ArchesClimate.main_hydra.instantiate"),
                patch("ArchesClimate.main_hydra.torch.utils.data.DataLoader"),
                patch("ArchesClimate.main_hydra.L.seed_everything"),
                patch("ArchesClimate.main_hydra.L.Trainer") as trainer,
                patch("ArchesClimate.main_hydra.ModelCheckpoint") as checkpoint,
            ):
                main.__wrapped__(cfg)
            self.assertTrue((Path(cfg.exp_dir) / "hydra_config.yaml").is_file())
            self.assertEqual(
                checkpoint.call_args.kwargs["dirpath"],
                str(Path(cfg.model_dir) / "checkpoints"),
            )
            self.assertIs(trainer.call_args.kwargs["logger"], False)
            trainer.return_value.fit.assert_called_once()

    def test_loader_reads_checkpoint_from_moved_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = OmegaConf.create(
                {
                    "model_dir": "/old/host/model",
                    "module": {"module": {"_target_": "unused.Model"}},
                }
            )
            OmegaConf.save(cfg, Path(tmp) / "hydra_config.yaml")
            with patch("ArchesClimate.model.base_module.instantiate") as instantiate:
                _, loaded = load_module(tmp, device="cpu")
            instantiate.return_value.init_from_ckpt.assert_called_once_with(
                Path(tmp), ckpt_fname=None, missing_warning=False
            )
            self.assertEqual(loaded.model_dir, str(Path(tmp).resolve()))


if __name__ == "__main__":
    unittest.main()
