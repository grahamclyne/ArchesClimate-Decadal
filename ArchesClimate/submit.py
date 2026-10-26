"""Run locally or submit the selected cluster launcher through submitit."""

import hydra
import submitit
from omegaconf import DictConfig, OmegaConf

from ArchesClimate.main_hydra import main as hydra_main
from ArchesClimate.paths import path as artifact_path


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    if not cfg.cluster.launcher:
        return hydra_main(cfg)
    folder = artifact_path("logs") / cfg.name
    folder.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(
        folder=str(folder), cluster="slurm", slurm_python="python"
    )
    executor.update_parameters(
        **OmegaConf.to_container(cfg.cluster.launcher, resolve=True)
    )
    job = executor.submit(hydra_main, cfg)
    print(f"Submitted job {job.job_id}")
    return job


if __name__ == "__main__":
    main()
