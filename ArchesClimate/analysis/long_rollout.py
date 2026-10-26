import hydra
import lightning as L
from omegaconf import DictConfig

from ArchesClimate.analysis.gen_and_save_metrics import run_metrics
from ArchesClimate.model.base_module import load_module


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed)
    dataset = hydra.utils.instantiate(
        cfg.dataloader.dataset, domain=cfg.inference.target
    )

    pl_module, _config = load_module(
        cfg.model_dir,
        cfg=cfg,
        ckpt_fname=cfg.inference.ckpt_fname,
    )

    index = 120 * cfg.inference.year

    experiment = f"{cfg.name}_{cfg.inference.num_inference_steps}_{cfg.inference.num_members}_{cfg.inference.scale_input_noise}_{cfg.inference.num_rollout_steps}_{cfg.inference.cf_guidance}_{cfg.inference.target}_{cfg.inference.year}_{cfg.inference.ckpt_fname}"
    out_dir = f"{cfg.paths.rollouts}/{experiment}/"
    pl_module.generate_rollouts(
        index,
        dataset,
        f"{cfg.inference.target}_{cfg.inference.year}",
        out_dir,
        cfg.inference.start_member,
        cfg.inference.end_member,
        cfg.inference.flat_forcings,
        cfg.inference.num_rollout_steps,
    )

    run_metrics(
        cfg,
        experiment,
        num_members=cfg.inference.num_members,
        num_rollout_steps=cfg.inference.num_rollout_steps,
    )


if __name__ == "__main__":
    main()
