import os

import hydra
import torch
from omegaconf import DictConfig
from tensordict.tensordict import TensorDict

from ArchesClimate.paths import path as artifact_path


def generate_dataset(cfg, dataset, target, year):
    ipsl_ensemble = []
    file_name = f"""{artifact_path("targets")}/{target}_ensemble_{year}_{cfg.name}"""
    if not os.path.exists(file_name):
        for k in range(0 + year, 10 + year):
            batch_timeseries = []
            for j in range(120):
                batch = dataset.__getitem__((k * 120) + j)
                print(k, j, batch["timestamp"])

                batch_timeseries.append(TensorDict(batch["state"]))
            batch_timeseries.append(TensorDict(batch["next_state"]))
            ipsl_ensemble.append(torch.stack(batch_timeseries))
        test_ensemble = torch.stack(ipsl_ensemble)
        torch.save(test_ensemble, file_name)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # generate test data
    test = hydra.utils.instantiate(cfg.dataloader.dataset, domain="test")
    # should be [1969,1979,1879,1999,2009]
    generate_dataset(cfg, test, "test", 0)
    generate_dataset(cfg, test, "test", 10)
    generate_dataset(cfg, test, "test", 20)
    generate_dataset(cfg, test, "test", 30)

    val = hydra.utils.instantiate(cfg.dataloader.dataset, domain="val")
    generate_dataset(cfg, val, "val", 0)
    generate_dataset(cfg, val, "val", 10)
    generate_dataset(cfg, val, "val", 20)


main()
