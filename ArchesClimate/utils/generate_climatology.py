import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from tensordict.tensordict import TensorDict


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    train = hydra.utils.instantiate(cfg.dataloader.dataset, domain="train")

    states = [[] for _ in range(12)]
    for count in range(4000):
        print(count)
        sample_idx = torch.randint(len(train), size=(1,)).item()
        batch = train.__getitem__(sample_idx, normalize=False)
        times = pd.to_datetime(batch["timestamp"].cpu().numpy(), unit="s").tz_localize(
            None
        )
        print(times)
        if (times.year > 2000) or (times.year < 1900):
            print(times.year)
            continue
        month = torch.tensor(times.month) - 1 % 12
        states[month].append(batch["state"].squeeze())

    states_stacked = [torch.stack(state) for state in states]
    level_mean = torch.stack([x["level"].mean(0) for x in states_stacked])
    level_std = torch.stack([x["level"].std(axis=(0, -1, -2)) for x in states_stacked])
    surface_mean = torch.stack([x["surface"].mean(0) for x in states_stacked])
    surface_std = torch.stack(
        [x["surface"].std(axis=(0, -1, -2)) for x in states_stacked]
    )
    lev_mean = torch.stack([x["lev"].mean(0) for x in states_stacked])
    lev_std = torch.stack([x["lev"].std(axis=(0, -1, -2)) for x in states_stacked])

    output = TensorDict(
        level_mean=level_mean,
        level_std=level_std,
        surface_mean=surface_mean,
        surface_std=surface_std,
        lev_mean=lev_mean,
        lev_std=lev_std,
    )
    print(output)

    torch.save(output, "climatology_stats_filled_in_1900_2000_full_ocean.pt")


main()
