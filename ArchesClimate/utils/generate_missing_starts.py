"""Target ensembles for the hindcast starts not covered by generate_test_val_ensembles.py
(test 2009 = year 40; val 2013-2015 = years 30/40/50). The final start of each split has
only 118 usable steps per member (dataset length 5998 / 7198), so it is built with 118 steps."""

import os

import hydra
import torch
from omegaconf import DictConfig
from tensordict.tensordict import TensorDict

from ArchesClimate.paths import path as artifact_path

OUTDIR = str(artifact_path("targets"))


def build(cfg, dataset, target, year, nsteps):
    name = f"{OUTDIR}/{target}_ensemble_{year}_{cfg.name}"
    if os.path.exists(name):
        print("exists", name)
        return
    members = []
    for k in range(year, year + 10):
        ts = []
        for j in range(nsteps):
            batch = dataset.__getitem__(k * 120 + j)
            ts.append(TensorDict(batch["state"]))
        ts.append(TensorDict(batch["next_state"]))
        members.append(torch.stack(ts))
    torch.save(torch.stack(members), name)
    print("saved", name, len(members), nsteps + 1)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    test = hydra.utils.instantiate(cfg.dataloader.dataset, domain="test")
    build(cfg, test, "test", 40, 118)
    val = hydra.utils.instantiate(cfg.dataloader.dataset, domain="val")
    build(cfg, val, "val", 30, 120)
    build(cfg, val, "val", 40, 120)
    build(cfg, val, "val", 50, 118)


main()
