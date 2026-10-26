"""Installed entry points; config inspection does not import training libraries."""

import hydra
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="config")
def config_main(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))


def train_main():
    from ArchesClimate.main_hydra import main

    main()


def submit_main():
    from ArchesClimate.submit import main

    main()


def rollout_main():
    from ArchesClimate.analysis.long_rollout import main

    main()


def stats_main():
    from ArchesClimate.utils.download_stats_from_hub import main

    main()
