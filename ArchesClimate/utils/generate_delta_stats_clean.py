from ArchesClimate.paths import analysis_device

"""Generate the delta (residual) std stats file used by DCPPDiffusion's
state_scaler when state_normalization='pred': std of (next_state - det_model
prediction) over random training examples, for ONE specific deterministic
backbone (no averaging across unrelated models, unlike the older
generate_delta_stats.py which hardcoded a 4-model ensemble average).

Must be regenerated whenever the deterministic backbone it conditions on
changes (e.g. after the indexed_year_forcings fix + backbone retrain), since
the residual statistics are backbone-specific.

Usage: overrides must set `name` and `model_dir` to the deterministic
backbone being characterized (and dataloader.dataset.filename_filter_type to
match), e.g.:
  python generate_delta_stats_clean.py name=forecast_added_input_state_0 \
      model_dir=/path/to/models/forecast_added_input_state_0
"""

import os

import hydra
import torch
from omegaconf import DictConfig

from ArchesClimate.model.base_module import load_module
from ArchesClimate.paths import path as artifact_path

STATS_DIR = str(artifact_path("stats"))
N_SAMPLES = 2000


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    train = hydra.utils.instantiate(cfg.dataloader.dataset, domain="train")
    pl_module, _ = load_module(
        cfg.model_dir, cfg=cfg, ckpt_fname=cfg.inference.ckpt_fname
    )
    pl_module = pl_module.to(analysis_device()).eval()

    batch0 = train[0]
    keys = batch0["next_state"].keys()
    n = 0
    mean = {k: 0 for k in keys}
    M2 = {k: 0 for k in keys}

    with torch.no_grad():
        for i in range(N_SAMPLES):
            if i % 200 == 0:
                print(i)
            sample_idx = torch.randint(len(train), size=(1,)).item()
            batch = train[sample_idx]
            batch = {k: v.to(analysis_device()) for k, v in batch.items()}
            batch = {k: v[None] for k, v in batch.items()}

            f_x = pl_module.forward(batch)
            delta = {k: (batch["next_state"][k] - f_x[k]).squeeze().cpu() for k in keys}

            n += 1
            for k in keys:
                delta_diff = delta[k] - mean[k]
                mean[k] += delta_diff / n
                M2[k] += delta_diff * (delta[k] - mean[k])

    std = {k: torch.sqrt(M2[k] / (n - 1)) for k in keys}
    out_path = os.path.join(STATS_DIR, f"dcpp_delta_std_{cfg.name}")
    torch.save(std, out_path)
    print("saved", out_path)


if __name__ == "__main__":
    main()
