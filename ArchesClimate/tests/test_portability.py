"""Lightweight tests: no training dependencies or external data needed."""

import importlib.util
import os
import subprocess
import tempfile
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


class ExportTests(unittest.TestCase):
    def test_export_filters_private_files_and_refuses_overwrite(self):
        root = Path(__file__).resolve().parents[2]
        script = root / "scripts/export_public.py"
        if not script.exists():
            self.skipTest("Exporter is a repository tool, not installed package data")
        spec = importlib.util.spec_from_file_location("export_public", script)
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()

            def git(*args):
                return subprocess.check_output(
                    ["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL
                )

            git("init")
            for name in [
                "ArchesClimate/__init__.py",
                "LICENSE",
                "THIRD_PARTY_NOTICES.md",
                "CITATION.cff",
                "licenses/upstream.txt",
                ".github/workflows/validate.yml",
                "ArchesClimate/ac_paper/paper.tex",
                "ArchesClimate/ac_paper/figures/private.pdf",
                "ArchesClimate/ac_paper_old/original.tex",
                "ArchesClimate/ac_paper.zip",
                "ArchesClimate/pcmdi/download_enso_pcmdi.py",
                "ArchesClimate/pmcdi/legacy.py",
                "ArchesClimate/docs/public_release.md",
                "ArchesClimate/configs/cluster/jean_zay_a100.yaml",
                "ArchesClimate/configs/cluster/local.yaml",
                "ArchesClimate/analysis/bootstrap_results/result.npz",
                "ArchesClimate/wandb/run.txt",
                "ArchesClimate/baselines/diffesm/data/native_repro/private.yml",
            ]:
                target = repo / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"fixture")
            git("add", ".")
            tree = git("write-tree").decode().strip()
            out = Path(tmp) / "preview"
            manifest = exporter.export(repo, tree, out)
            self.assertEqual(len(manifest["sha256"]), 8)
            self.assertTrue((out / "LICENSE").is_file())
            self.assertTrue((out / "licenses/upstream.txt").is_file())
            self.assertTrue((out / ".github/workflows/validate.yml").is_file())
            self.assertTrue((out / "ArchesClimate/__init__.py").exists())
            for excluded in [
                "ArchesClimate/ac_paper",
                "ArchesClimate/ac_paper_old",
                "ArchesClimate/ac_paper.zip",
                "ArchesClimate/pcmdi",
                "ArchesClimate/pmcdi",
                "ArchesClimate/docs/public_release.md",
            ]:
                self.assertFalse((out / excluded).exists(), excluded)
            self.assertFalse(
                (out / "ArchesClimate/baselines/diffesm/data/native_repro").exists()
            )
            with self.assertRaises(FileExistsError):
                exporter.export(repo, tree, out)


if __name__ == "__main__":
    unittest.main()
