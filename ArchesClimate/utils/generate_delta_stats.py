import hydra
import torch
from omegaconf import DictConfig
from tensordict.tensordict import TensorDict

from ArchesClimate.model.base_module import load_module
from ArchesClimate.paths import path as artifact_path


def nanvar(tensor, dim=None, keepdim=False):
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).apply(lambda x: x.square())
    output = TensorDict(
        {
            "surface": output["surface"].nanmean(dim=0, keepdim=keepdim),
            "level": output["level"].nanmean(dim=0, keepdim=keepdim),
            "lev": output["lev"].nanmean(dim=0, keepdim=keepdim),
        }
    )
    return output


def nanstd(tensor, dim=None, keepdim=False):
    output = nanvar(tensor, dim=dim, keepdim=keepdim)
    output = output.apply(lambda x: x.sqrt())
    return output


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    train = hydra.utils.instantiate(cfg.dataloader.dataset, domain="train")
    print(cfg)
    pl_module, _config = load_module(
        cfg.model_dir,
        cfg=cfg,
        # path=cfg.model_dir,
        avg_with_modules=[
            str(
                artifact_path("models")
                / "dcpp-deterministic/forecast_added_input_state_1"
            ),
            str(
                artifact_path("models")
                / "dcpp-deterministic/forecast_added_input_state_2"
            ),
            str(
                artifact_path("models")
                / "dcpp-deterministic/forecast_added_input_state_0"
            ),
            str(
                artifact_path("models")
                / "dcpp-deterministic/forecast_added_input_state_3"
            ),
        ],
    )

    # deltas = []
    # for count in range(2000):
    #     print(count)
    #     sample_idx = torch.randint(len(train), size=(1,)).item()
    #     batch = train[sample_idx]
    #     batch = {k: v.to('cuda') for k, v in batch.items()}
    #     batch = {k: v[None] for k, v in batch.items()}
    #     del batch['next_state']['non_spatial_forcings']
    #     del batch['next_state']['spatial_forcings']
    #     f_x = pl_module.forward(batch)
    #     delta = (batch['next_state'] - f_x).squeeze()
    #     deltas.append(delta.detach().cpu())
    # torch.save(nanstd(torch.stack(deltas),dim=0),'cmip_delta_std_new_oro')

    batch = train[0]
    # del batch["next_state"]["non_spatial_forcings"]
    # del batch["next_state"]["spatial_forcings"]
    keys = batch["next_state"].keys()
    n = 0
    mean = {k: 0 for k in keys}
    M2 = {k: 0 for k in keys}

    with torch.no_grad():
        for i in range(2000):
            print(i)
            sample_idx = torch.randint(len(train), size=(1,)).item()
            batch = train[sample_idx]
            batch = {k: v.to("cuda") for k, v in batch.items()}
            batch = {k: v[None] for k, v in batch.items()}
            # del batch["next_state"]["non_spatial_forcings"]
            # del batch["next_state"]["spatial_forcings"]

            f_x = pl_module.forward(batch)
            delta = {k: (batch["next_state"][k] - f_x[k]).squeeze().cpu() for k in keys}

            n += 1
            for k in keys:
                delta_diff = delta[k] - mean[k]
                mean[k] += delta_diff / n
                M2[k] += delta_diff * (delta[k] - mean[k])

    std = {k: torch.sqrt(M2[k] / (n - 1)) for k in keys}
    torch.save(std, f"dcpp_delta_std_{cfg.name}")
    # np.save('depth_delta_std',np.nanstd(np.stack(depth_deltas),axis=0))
    # torch.save(nanstd(torch.stack(plev_deltas),dim=0),'cmip_plev_delta_std')


if __name__ == "__main__":
    main()
