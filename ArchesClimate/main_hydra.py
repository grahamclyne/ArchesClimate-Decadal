import os
import warnings
from pathlib import Path

import hydra
import lightning as L
import torch
import torch.multiprocessing
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf

# torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
# torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata])
# torch.serialization.add_safe_globals([typing.Any])
# torch.serialization.add_safe_globals([list])
# torch.serialization.add_safe_globals([collections.defaultdict])
# torch.serialization.add_safe_globals([dict])
# torch.serialization.add_safe_globals([int])
# torch.serialization.add_safe_globals([omegaconf.nodes.AnyNode])
# import tracemalloc

# class CPUMemoryMonitor(L.pytorch.Callback):
#     def on_train_start(self, *args, **kwargs):
#         tracemalloc.start()

#     def on_train_epoch_end(self, trainer, pl_module):
#         snapshot = tracemalloc.take_snapshot()
#         top_stats = snapshot.statistics('lineno')
#         print("[Top 3 CPU memory lines:]")
#         for stat in top_stats[:3]:
#             print(stat)
# import tracemalloc

# class CPUMemoryMonitor(L.pytorch.Callback):
#     def on_train_start(self, *args, **kwargs):
#         tracemalloc.start()


#     def on_train_epoch_end(self, trainer, pl_module):
#         snapshot = tracemalloc.take_snapshot()
#         top_stats = snapshot.statistics('lineno')
#         print("[Top 3 CPU memory lines:]")
#         for stat in top_stats[:3]:
#             print(stat)
# Custom resolver
def calculate_length(surface_variables, level_variables):
    # three pressure levels, three times added: prev, noise, actual
    return (len(surface_variables) + len(level_variables) * 3) * 3


def get_random_code():
    import random
    import string

    # generate random code that alternates letters and numbers
    l = random.choices(string.ascii_lowercase, k=3)
    n = random.choices(string.digits, k=3)
    return "".join([f"{a}{b}" for a, b in zip(l, n)])


# def collate_fn(lst):
#     # print(lst)
#     keys = list(lst[0]) if not hasattr(lst[0], "keys") else list(lst[0].keys())

#     return {k: torch.stack([x[k] for x in lst]) for k in keys}
#     # print(lst)
#     keys = list(lst[0]) if not hasattr(lst[0], "keys") else list(lst[0].keys())

#     return {k: torch.stack([x[k] for x in lst]) for k in keys}


from tensordict import TensorDict, stack

# def collate_fn(lst):
#     # Much faster C-level implementation for TensorDicts
#     return stack(lst)


def collate_fn(lst):
    # Convert each dict in the list to a TensorDict
    # batch_size=[] is required for individual samples
    te_list = [TensorDict(d, batch_size=[]) if isinstance(d, dict) else d for d in lst]
    return stack(te_list)


class CheckpointEveryNSteps(L.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        dirpath="./",
        save_step_frequency=50000,
        prefix="checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename
        self.dirpath = dirpath

    def on_train_batch_end(self, trainer: L.Trainer, *args, **kwargs):
        """Check if we should save a checkpoint after every train batch"""
        if not hasattr(self, "trainer"):
            self.trainer = trainer

        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            self.save()

    # def save(self, *args, trainer=None, **kwargs):
    #     if trainer is None and not hasattr(self, "trainer"):
    #         print("No trainer !")
    #         return
    #     if trainer is None:
    #         trainer = self.trainer

    #     global_step = trainer.global_step
    #     if self.use_modelcheckpoint_filename:
    #         filename = trainer.checkpoint_callback.filename
    #     else:
    #         filename = f"{self.prefix}_{global_step=}.ckpt"
    #     ckpt_path = Path(self.dirpath) / "checkpoints"
    #     ckpt_path.mkdir(exist_ok=True, parents=True)
    #     trainer.save_checkpoint(ckpt_path / filename)
    def save(self, *args, trainer=None, **kwargs):
        if trainer is None and not hasattr(self, "trainer"):
            print("No trainer !")
            return
        if trainer is None:
            trainer = self.trainer

        global_step = trainer.global_step

        # Try to get ModelCheckpoint callback
        mc_callbacks = [
            cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)
        ]
        ckpt_cb = mc_callbacks[0] if mc_callbacks else None

        if self.use_modelcheckpoint_filename and ckpt_cb is not None:
            # Use ModelCheckpoint's filename template
            filename = (
                ckpt_cb.filename
                or f"epoch={trainer.current_epoch}-step={global_step}.ckpt"
            )
        else:
            # Fallback to custom naming
            filename = f"{self.prefix}_step={global_step}.ckpt"

        ckpt_dir = Path(self.dirpath) / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True, parents=True)

        ckpt_path = ckpt_dir / filename
        trainer.save_checkpoint(str(ckpt_path))

        print(f"Checkpoint saved at {ckpt_path}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig, cli_overrides=None):

    if cfg.mode != "train":
        raise ValueError(
            "Only mode=train is supported here. Use analysis/long_rollout.py "
            "and the analysis evaluation scripts for paper inference."
        )
    OmegaConf.register_new_resolver("calculate_length", calculate_length)
    # torch.multiprocessing.set_sharing_strategy('file_system')

    warnings.simplefilter(action="ignore", category=FutureWarning)
    print("Working dir", os.getcwd())

    main_node = int(os.environ.get("SLURM_PROCID", "0")) == 0
    print("is main node", main_node)

    # init some variables
    logger = None
    ckpt_path = None

    # first, check if exp exists
    if Path(cfg.exp_dir).exists():
        print("Experiment already exists. Trying to resume it.")
        hydra_cfg_file = Path(cfg.exp_dir) / "hydra_config.yaml"
        exp_cfg = OmegaConf.load(
            hydra_cfg_file
            if hydra_cfg_file.exists()
            else Path(cfg.exp_dir) / "config.yaml"
        )
        if cfg.resume:
            # we just copy cluster info
            cfg.module = exp_cfg.module
            # cfg.dataloader = exp_cfg.dataloader
            # # then we update we cli overrides
            # cli_overrides = cli_overrides #or HydraConfig.get().overrides.task
            # cli_overrides = [x for x in cli_overrides if x.startswith("+")]

            # cli_cfg = compose(overrides=cli_overrides)
            # OmegaConf.set_struct(cfg, False)  # to merge
            # cfg = OmegaConf.merge(cfg, cli_cfg)

        else:
            # check that new config and old config match
            if OmegaConf.to_yaml(cfg.module, resolve=True) != OmegaConf.to_yaml(
                exp_cfg.module
            ):
                print("Module config mismatch. Exiting")
                print("Old config", OmegaConf.to_yaml(exp_cfg.module))
                print("New config", OmegaConf.to_yaml(cfg.module))

            if OmegaConf.to_yaml(cfg.dataloader, resolve=True) != OmegaConf.to_yaml(
                exp_cfg.dataloader
            ):
                print("Dataloader config mismatch. Exiting.")
                print("Old config", OmegaConf.to_yaml(exp_cfg.dataloader))
                print("New config", OmegaConf.to_yaml(cfg.dataloader))
                return

        # trying to find checkpoints
        ckpt_dir = Path(cfg.model_dir).joinpath("checkpoints")
        ckpt_dir = Path(cfg.model_dir).joinpath("checkpoints")
        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.iterdir(), key=os.path.getmtime)
            if len(ckpts):
                print("Found checkpoints", ckpts)
                ckpt_path = ckpts[-1]
                print(ckpt_path)

    if cfg.log:
        os.environ["WANDB_DISABLE_SERVICE"] = "False"
        print("wandb mode", cfg.cluster.wandb_mode)
        print(
            "wandb service", os.environ.get("WANDB_DISABLE_SERVICE", "variable unset")
        )
        run_id = (
            cfg.name + "-" + get_random_code()
            if cfg.cluster.use_custom_requeue
            else cfg.name
        )
        logger = L.pytorch.loggers.WandbLogger(
            project=cfg.project,
            name=cfg.name,
            id=run_id,
            save_dir=cfg.cluster.wandb_dir,
            offline=(cfg.cluster.wandb_mode != "online"),
            resume="allow",
        )

    if cfg.log and main_node:
        hparams = OmegaConf.to_container(cfg, resolve=True)
        logger.log_hyperparams(hparams)

    if main_node and not Path(cfg.exp_dir).exists():
        print("registering exp on main node")
        Path(cfg.exp_dir).mkdir(parents=True)
        config_path = Path(cfg.exp_dir) / "hydra_config.yaml"
        with open(config_path, "w") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
        if cfg.log:
            logger.experiment.save(
                str(config_path), base_path=str(Path(cfg.exp_dir)), policy="now"
            )

    if cfg.log and main_node:
        logger.experiment.log_code(
            root=str(Path(__file__).parent),
            include_fn=lambda path: (
                (path.endswith((".py", ".yaml"))) and os.path.getsize(path) < 1_000_000
            ),
            exclude_fn=lambda path, root: any(
                part in path
                for part in ["sblogs", "outputs", "__pycache__", ".git", "wandb"]
            ),
        )

    if cfg.mode == "train":
        valset = instantiate(cfg.dataloader.dataset, domain="val")
        trainset = instantiate(
            cfg.dataloader.dataset
        )  # will automatically pickup cfg split

        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=1,  # to make sample size
            num_workers=cfg.cluster.cpus,
            shuffle=False,
            collate_fn=collate_fn,
        )  # to viz shuffle samples

        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=cfg.batch_size,
            num_workers=cfg.cluster.cpus,
            # num_workers=0,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
            prefetch_factor=3 if cfg.cluster.cpus else None,  # <--- INCREASE THIS
            # prefetch_factor=4, # <--- The key to fixing troughs
            # pin_memory=True,
            # persistent_workers=True
        )

    pl_module = instantiate(cfg.module.module, cfg.module)

    # if hasattr(cfg, "load_ckpt"):
    #     # load weights w/o resuming run

    # 4. Fit
    # pl_module = torch.compile(pl_module)
    # checkpointer = CheckpointEveryNSteps(
    #     dirpath=cfg.exp_dir, save_step_frequency=cfg.save_step_frequency
    # )
    # checkpointer = ModelCheckpoint(
    #     save_on_train_epoch_end=True,
    #     # save_top_k=3,
    #     filename="epoch-{epoch:02d}",
    #     save_last=False,
    #     save_top_k=-1,
    # )
    # checkpointer = CheckpointEveryNSteps(
    #     dirpath=cfg.exp_dir, save_step_frequency=cfg.save_step_frequency
    # )
    checkpointer = ModelCheckpoint(
        dirpath=str(Path(cfg.model_dir) / "checkpoints"),
        save_on_train_epoch_end=True,
        # save_top_k=3,
        filename="epoch-{epoch:02d}",
        save_last=False,
        save_top_k=-1,
    )
    # signal.signal(signal.SIGTERM, signal.SIG_DFL)

    # if cfg.resume and main_node:
    #     print("setting up custom slurm requeuing")

    #     def handler(*args, **kwargs):
    #         print("GCO: SIGTERM signal received. Requeueing job on main node.")
    #         if not hasattr(checkpointer, "is_handled"):
    #             checkpointer.is_handled = True
    #             # checkpointer.save()
    #             # checkpointer.save()
    #             from ArchesClimate.submit import main as geoarches_submit

    #             geoarches_submit(cfg)
    #         exit()

    # signal.signal(signal.SIGTERM, handler)
    # signal.signal(signal.SIGCONT, handler)

    L.seed_everything(cfg.seed)
    trainer = L.Trainer(
        devices=cfg.cluster.gpus if torch.cuda.is_available() else 1,
        accelerator="auto",
        # strategy='auto'
        # if torch.cuda.is_available()
        # else "auto",
        num_nodes=cfg.cluster.launcher.get("nodes", 1),
        precision=cfg.cluster.precision,
        log_every_n_steps=cfg.log_freq,
        profiler=getattr(cfg, "profiler", None),
        gradient_clip_val=1,
        strategy=DDPStrategy(find_unused_parameters=True)
        if cfg.cluster.gpus > 1 or cfg.cluster.launcher.get("nodes", 1) > 1
        else "auto",
        # max_steps=cfg.max_steps,
        max_epochs=cfg.get("max_epochs", None),
        # limit_train_batches=0.01,
        callbacks=[
            TQDMProgressBar(refresh_rate=100 if cfg.mode == "train" else 1),
            checkpointer,
        ],
        logger=logger if cfg.log else False,
        plugins=[],
        limit_val_batches=cfg.limit_val_batches,  # max 5 samples
        limit_test_batches=cfg.limit_val_batches,  # max 5 samples
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        reload_dataloaders_every_n_epochs=1,
    )

    if cfg.debug:
        breakpoint()  # noqa: T100 -- explicit interactive opt-in with debug=True
    if cfg.mode == "train":
        trainer.fit(pl_module, train_loader, val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()


# python main_hydra.py log=False limit_examples=100 debug=True
