import hydra
import torch
from omegaconf import DictConfig
from tensordict.tensordict import TensorDict


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # need this to generate climatology
    print(cfg)
    train = hydra.utils.instantiate(cfg.dataloader.dataset, domain="train")
    # generate climatology ensemble

    # 80-90 for 1968
    # 160-170 for 1978
    # 250-260 for 1988

    ipsl_ensemble = []
    for k in range(250, 260):
        batch_timeseries = []
        for j in range(120):
            batch = train.__getitem__((k * 120) + j)
            print(k, j, batch["timestamp"])

            batch_timeseries.append(TensorDict(batch["state"]))
        batch_timeseries.append(TensorDict(batch["next_state"]))
        ipsl_ensemble.append(torch.stack(batch_timeseries))
    climatology_ensemble = torch.stack(ipsl_ensemble)
    torch.save(climatology_ensemble, "persistence_ensemble_1988_full_vars")


main()
