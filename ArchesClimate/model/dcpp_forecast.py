import os

import diffusers
import pandas as pd
import torch
import torch.utils.checkpoint as gradient_checkpoint
from hydra.utils import instantiate
from tensordict.tensordict import TensorDict
from torch import nn

from ArchesClimate.backbones.dit import TimestepEmbedder
from ArchesClimate.metrics.ensemble_metrics import EnsembleMetrics
from ArchesClimate.model.base_module import BaseLightningModule
from ArchesClimate.model.dcpp_diffusion import apply_iszero
from ArchesClimate.model.spectral_loss import OptimizedSpectralLoss
from ArchesClimate.paths import path as artifact_path

geoarches_stats_path = artifact_path("stats")


def apply_zero_to_nan(td):
    return TensorDict(
        {
            key: torch.where(
                value == 0,
                torch.tensor(float("nan"), dtype=value.dtype, device=value.device),
                value,
            )
            for key, value in td.items()
        },
        batch_size=td.batch_size,
    )


class ForecastModule(BaseLightningModule):
    def __init__(
        self,
        cfg=None,  # instead of backbone
        name="forecast",
        dataset=None,
        pow=2,  # 2 is standard mse
        loss_delta_normalization=True,
        lr=1e-4,
        betas=(0.9, 0.98),
        weight_decay=1e-5,
        num_warmup_steps=1000,
        num_training_steps=300000,
        num_cycles=0.5,
        add_input_state=False,
        save_test_outputs=False,
        use_weatherbench_lat_coeffs=True,
        lead_time_hours=24,
        rollout_iterations=1,
        test_filename_suffix="",
        include_forcings=True,
        forcing_embed=True,
        time_forcing=True,
        month_embedding=True,
        year_embedding=True,
        flat_forcings=False,
        separate_forcings=False,
        lat_dimension=143,
        ghg_only=False,
        proper_year_forcings=False,
        include_orography=False,
        include_co2=True,
        include_ssi=True,
        atmos_forcing_dim=5,  # number of leading forcing dims that are CO2/GHG
        # NOTE: previously accepted here only via **kwargs and silently discarded --
        # loss() always applied the full composite (spectral+gradient) loss
        # regardless of these flags, so every "NCL" config (including the
        # original paper's forecast_added_input_state_small_no_spectral*) was
        # actually training with composite loss on the whole time. Now real:
        # default True preserves that historical always-on behavior for every
        # config that doesn't explicitly set these to False.
        use_spectral_loss=True,
        use_grad_loss=True,
        **kwargs,
    ):
        """should create self.encoder and self.decoder in subclasses"""
        super().__init__()
        # self.save_hyperparameters()
        self.__dict__.update(locals())
        self.backbone = instantiate(cfg.backbone)  # necessary to put it on device
        self.embedder = instantiate(cfg.embedder)
        self.forcing_embedder = TimestepEmbedder(512)
        self.atmos_embedder = TimestepEmbedder(512)
        self.solar_embedder = TimestepEmbedder(512)
        self.spectral_loss = OptimizedSpectralLoss()
        # define coeffs for loss

        # Define initial weights
        self.loss_weights = {
            "mse": 1.0,
            "spectral": 0,
            "gradient": 0,
        }

        # nn.ModuleList (not a plain list) so Lightning's automatic device
        # placement moves this metric to each DDP rank's own device; a plain
        # list + hardcoded .to('cuda') pins it to cuda:0 for every rank,
        # crashing non-zero ranks with "different devices in metric
        # calculation" as soon as validation_step runs on cuda:1+.
        self.val_metrics = nn.ModuleList(
            [EnsembleMetrics(data_shape=(1, len(cfg.surface_variables)))]
        )  # only one timestep
        lat_coeffs_equi = torch.tensor(
            [
                torch.cos(x)
                for x in torch.arange(
                    -torch.pi / 2 + 1e-2, torch.pi / 2, torch.pi / lat_dimension
                )
            ]
        )
        lat_coeffs_equi = lat_coeffs_equi / lat_coeffs_equi.mean()
        self.loss_coeffs = TensorDict(
            surface=lat_coeffs_equi[None, None, None, :, None],
            level=lat_coeffs_equi[None, None, None, :, None],
        )
        # if loss_delta_normalization:
        #     # assumes include vertical wind component

        #     surface_stats = torch.load(
        #         geoarches_stats_path / "dcpp_surface_delta_std", weights_only=True
        #     )
        #     level_stats = torch.load(
        #         geoarches_stats_path / "dcpp_plev_delta_std", weights_only=True
        #     )
        #     # # mul by first to remove norm, div by second to apply fake delta normalization
        #     self.loss_delta_scaler = TensorDict(
        #         level=level_stats,
        #         surface=surface_stats
        #     )

        #     self.loss_coeffs = self.loss_coeffs * self.loss_delta_scaler.pow(self.pow)

    def forward(self, batch, *args, **kwargs):
        x = self.embedder.encode(batch["state"], batch.get("prev_state", None))
        x = self.backbone(x, *args, **kwargs)
        out = self.embedder.decode(x)  # we get tdict
        # if self.add_input_state:
        #     out += batch["state"]

        return out

    def loss(self, pred, gt, multistep=False, **kwargs):
        loss_coeffs = self.loss_coeffs.to(self.device)

        mask = TensorDict(
            {k: (apply_iszero(v) if "state" in k else v) for k, v in gt.items()}
        )

        pred = TensorDict(
            {k: (v * mask[k] if "state" in k else v) for k, v in pred.items()}
        )

        # pred = TensorDict({k: (apply_zero_to_nan(v) if "state" in k else v) for k, v in pred.items()})
        # gt = TensorDict({k: (apply_zero_to_nan(v) if "state" in k else v) for k, v in gt.items()})

        if multistep:  # means we have to compute multistep loss
            # discount for multistep loss
            lead_iter = next(iter(gt.values())).shape[1]
            future_coeffs = (
                torch.tensor([1 / (1 + i) ** 2 for i in range(lead_iter)])
                .to(self.device)
                .reshape(-1, 1, 1, 1, 1)
            )
            loss_coeffs.apply(lambda x: x * future_coeffs)

        # Gradient loss for preserving spatial patterns
        grad_x_pred = pred.apply(lambda x: x[..., :, 1:] - x[..., :, :-1])
        grad_x_target = gt.apply(lambda x: x[..., :, 1:] - x[..., :, :-1])
        grad_y_pred = pred.apply(lambda x: x[..., 1:, :] - x[..., :-1, :])
        grad_y_target = gt.apply(lambda x: x[..., 1:, :] - x[..., :-1, :])

        grad_loss = sum(
            (grad_x_pred - grad_x_target).abs().pow(self.pow).mean().values()
        ) + sum((grad_y_pred - grad_y_target).abs().pow(self.pow).mean().values())

        weighted_error = (pred - gt).abs().pow(self.pow).mul(loss_coeffs)

        loss = sum(weighted_error.mean().values()) * self.loss_weights["mse"]
        grad_loss = grad_loss * self.loss_weights["gradient"]
        spectral_surface_loss = (
            self.spectral_loss(pred["surface"], gt["surface"])
            * self.loss_weights["spectral"]
        )
        spectral_level_loss = (
            self.spectral_loss(pred["level"], gt["level"])
            * self.loss_weights["spectral"]
        )
        total_loss = loss + grad_loss + spectral_surface_loss + spectral_level_loss
        if not torch.isfinite(total_loss):
            print(
                f"[NaN/Inf] mse={loss.item():.4f} grad={grad_loss.item():.4f} "
                f"spec_sfc={spectral_surface_loss.item():.4f} spec_lvl={spectral_level_loss.item():.4f}"
            )
            print(
                f"[NaN/Inf] pred surface: min={pred['surface'].min().item():.3f} max={pred['surface'].max().item():.3f} "
                f"nans={pred['surface'].isnan().sum().item()}"
            )
            print(
                f"[NaN/Inf] pred level:   min={pred['level'].min().item():.3f} max={pred['level'].max().item():.3f} "
                f"nans={pred['level'].isnan().sum().item()}"
            )
            print(
                f"[NaN/Inf] gt surface nans={gt['surface'].isnan().sum().item()} "
                f"gt level nans={gt['level'].isnan().sum().item()}"
            )
        return total_loss

    def training_step(self, batch, batch_nb):

        step = self.global_step
        warmup_steps = 10000
        max_warmup_steps = 30000
        if step > warmup_steps:
            if self.use_spectral_loss:
                self.loss_weights["spectral"] = 0.1 * min(
                    (step - warmup_steps) / max_warmup_steps, 1.0
                )
            if self.use_grad_loss:
                self.loss_weights["gradient"] = 1.0 * min(
                    (step - warmup_steps) / max_warmup_steps, 1.0
                )

        if "future_states" not in batch:
            # standard prediction
            pred = self.forward(batch)
            loss = self.loss(pred, batch["next_state"])
            self.mylog(loss=loss)

        else:
            # multistep prediction
            lead_iter = batch["future_states"].shape[1]
            pred_future_states = self.forward_multistep(
                batch,
                iters=lead_iter,
                test_dataset=self.trainer.train_dataloader.dataset,
            )
            loss = self.loss(pred_future_states, batch["future_states"], multistep=True)

            self.mylog(lead_iter=lead_iter)
            self.mylog(loss=loss)

        return loss

    def on_validation_epoch_start(self):
        for metric in self.val_metrics:
            metric.reset()

    def forward_multistep(
        self,
        batch,
        iters=None,
        return_format="tensordict",
        test_dataset=None,
        is_sampling=False,
        use_avg=True,
        out_dir=None,
        index=None,
        ensemble_member_index=None,
    ):
        # sequential_date_indices = list(filter_unique_third_tuple_values(test_dataset.id2pt))
        if use_avg and self.avg_modules is not None:
            out = self.forward_multistep(
                batch,
                iters=iters,
                use_avg=False,
                test_dataset=test_dataset,
                is_sampling=is_sampling,
            )
            for m in self.avg_modules:
                out = out + m.forward_multistep(
                    batch,
                    iters=iters,
                    use_avg=False,
                    test_dataset=test_dataset,
                    is_sampling=is_sampling,
                )
            return out / (1 + len(self.avg_modules))
        preds_future = []
        loop_batch = {k: v for k, v in batch.items()}
        if is_sampling:  # append first state as sanity check for sampling
            preds_future.append(loop_batch["state"])

        if self.flat_forcings:
            next_timestamp = torch.stack(
                [
                    torch.tensor(
                        test_dataset.next_timestamp_map[x.item()],
                        device=loop_batch["state"].device,
                    )
                    for x in loop_batch["timestamp"]
                ]
            )
            times = pd.to_datetime(next_timestamp.cpu(), unit="s").tz_localize(None)
            first_year = torch.tensor(times.year)

        for i in range(iters):
            with torch.set_grad_enabled(not is_sampling):
                if torch.is_grad_enabled():
                    pred = gradient_checkpoint.checkpoint(
                        self.forward, loop_batch, use_reentrant=False
                    )
                else:
                    pred = self.forward(loop_batch)
                preds_future.append(pred)
                next_timestamp = torch.stack(
                    [
                        torch.tensor(
                            test_dataset.next_timestamp_map[x.item()],
                            device=loop_batch["state"].device,
                        )
                        for x in loop_batch["timestamp"]
                    ]
                )
                times = pd.to_datetime(next_timestamp.cpu(), unit="s").tz_localize(None)
                if self.proper_year_forcings:
                    current_year = torch.tensor(times.year)  # -1961 to zero index
                    current_solar_year = torch.tensor(times.year) - 1961
                else:
                    current_year = torch.tensor(times.year) - 1961  # to zero index
                    current_solar_year = torch.tensor(times.year) - 1961
                current_month = torch.tensor(times.month) - 1 % 12
                if self.ghg_only:
                    current_year += 1961  # different scheme for cmip
                    next_forcings = torch.tensor(
                        test_dataset.atmos_forcings[current_year]
                    ).to(loop_batch["state"].device)[None]
                elif self.flat_forcings:
                    next_forcings = torch.concatenate(
                        [
                            test_dataset.atmos_forcings[first_year, :],
                            test_dataset.solar_forcings[
                                (first_year * 12) + current_month, :
                            ],
                        ],
                        dim=1,
                    ).to(loop_batch["state"].device)
                else:
                    next_forcings = torch.concatenate(
                        [
                            test_dataset.atmos_forcings[current_year, :],
                            test_dataset.solar_forcings[
                                (current_solar_year * 12) + current_month, :
                            ],
                        ],
                        dim=1,
                    ).to(loop_batch["state"].device)
                loop_batch["state"]["surface"] = loop_batch["state"]["surface"][
                    :, :17
                ]  # remove orography before putting as prev_state
                loop_batch = {
                    "prev_state": loop_batch["state"],
                    "state": pred,
                    "timestamp": next_timestamp,
                    "forcings": next_forcings,
                }
                if ((len(preds_future) % 120) == 0) and i > 0:
                    out = torch.stack(preds_future, dim=1)
                    torch.save(
                        out["surface"],
                        f"{out_dir}/rollout_{index}_surface_{i}_{ensemble_member_index}.pt",
                    )
                    torch.save(
                        out["level"],
                        f"{out_dir}/rollout_{index}_level_{i}_{ensemble_member_index}.pt",
                    )
                    preds_future = []
        if return_format == "list":
            return preds_future
        preds_future = torch.stack(preds_future, dim=1)
        return preds_future
        # MSE remains fixed or can be adjusted similarly

    def validation_step(self, batch, batch_nb):

        pred = self.forward(batch)
        loss = self.loss(pred, batch["next_state"])
        self.mylog(loss=loss)

        for metric in self.val_metrics:
            metric.update(
                batch["next_state"]["surface"][:, :, 0], pred["surface"][None, :, :, 0]
            )
        return loss

    def generate_rollouts(
        self,
        batch_index,
        test_dataset,
        target_name,
        out_dir,
        start_member,
        end_member,
        flat_forcings,
        rollout_length,
    ):
        os.makedirs(out_dir, exist_ok=True)

        for ensemble_member in range(start_member, end_member):
            # b_index = (batch_index + (ensemble_member*120)) % 1200 #get different IC for each ensemble member
            b_index = batch_index + (ensemble_member * 120)
            batch = test_dataset.__getitem__(b_index)
            batch = {k: batch[k].unsqueeze(0) for k in batch}
            batch = {k: batch[k].to(self.device) for k in batch}
            rollout = self.forward_multistep(
                batch,
                iters=rollout_length,
                test_dataset=test_dataset,
                is_sampling=True,
                out_dir=out_dir,
                index=batch_index,
                ensemble_member_index=ensemble_member,
            )
            torch.save(
                rollout["surface"], f"{out_dir}/rollout_{ensemble_member}_surface.pt"
            )
            torch.save(
                rollout["level"], f"{out_dir}/rollout_{ensemble_member}_level.pt"
            )

    def on_validation_epoch_end(self):
        for metric in self.val_metrics:
            outputs = metric.compute()
            for k, v in outputs.items():
                outputs[k] = v.mean()
            self.mylog(**outputs, mode="val_")
            metric.reset()

    def configure_optimizers(self):
        decay_params = {
            k: True
            for k, v in self.named_parameters()
            if "weight" in k and "norm" not in k
        }
        opt = torch.optim.AdamW(
            [
                {
                    "params": [
                        v for k, v in self.named_parameters() if k in decay_params
                    ]
                },
                {
                    "params": [
                        v for k, v in self.named_parameters() if k not in decay_params
                    ],
                    "weight_decay": 0,
                },
            ],
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        sched = diffusers.optimization.get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
            num_cycles=self.num_cycles,
        )
        sched = {
            "scheduler": sched,
            "interval": "step",  # or 'epoch'
            "frequency": 1,
        }
        return [opt], [sched]


class ForecastModuleWithCond(ForecastModule):
    """
    module that can take additional information:
    - month and hour
    - previous state
    - pred state (e.g. prediction of other weather model)
    """

    def __init__(
        self,
        *args,
        cond_dim=32,
        use_prev=False,
        use_avg=False,
        avg_with_modules=None,
        **kwargs,
    ):
        from ArchesClimate.backbones import dit

        super().__init__(*args, **kwargs)
        # cond_dim should be given as arg to the backbone
        self.cond_dim = cond_dim
        self.month_embedder = dit.TimestepEmbedder(cond_dim)
        self.hour_embedder = dit.TimestepEmbedder(cond_dim)
        self.use_prev = use_prev
        self.use_avg = use_avg
        self.avg_modules = None

        if avg_with_modules:
            from ArchesClimate.model.base_module import load_module

            self.avg_modules = nn.ModuleList(
                [load_module(m, return_config=False) for m in avg_with_modules]
            )
            self.strict_loading = False

    def forward(self, batch, use_avg=True):
        device = batch["state"].device
        times = pd.to_datetime(batch["timestamp"].cpu().numpy(), unit="s").tz_localize(
            None
        )
        cond_emb = 0
        if self.month_embedding:
            month = (torch.tensor(times.month) - 1 % 12).to(device)
            month_emb = self.month_embedder(month)
            cond_emb = cond_emb + month_emb
        if self.year_embedding:
            if self.proper_year_forcings:
                year = (torch.tensor(times.year)).to(device)
            else:
                year = (torch.tensor(times.year) - 1961).to(device)

            year_emb = self.hour_embedder(year)
            cond_emb = cond_emb + year_emb
        if self.include_forcings:
            bound = len(batch["forcings"][0])
            for forcing_index in range(bound):
                is_atmos = forcing_index < self.atmos_forcing_dim
                if is_atmos and not self.include_co2:
                    continue
                if not is_atmos and not self.include_ssi:
                    continue
                if self.ghg_only:
                    cond_emb += self.atmos_embedder(batch["forcings"][:, forcing_index])
                elif self.separate_forcings:
                    if is_atmos:
                        cond_emb += self.atmos_embedder(
                            batch["forcings"][:, forcing_index]
                        )
                    else:
                        cond_emb += self.solar_embedder(
                            batch["forcings"][:, forcing_index]
                        )
                else:
                    if self.forcing_embed:
                        cond_emb += self.forcing_embedder(
                            batch["forcings"][:, forcing_index]
                        )
                    else:
                        cond_emb += self.month_embedder(
                            batch["forcings"][:, forcing_index]
                        )

        if isinstance(cond_emb, int):
            # No conditioning source enabled at all (month/year/forcings all
            # off) -- this is the "NC" (no conditional layer norm) ablation:
            # cond_emb stays the literal 0 it was initialized to, which would
            # crash the backbone's nn.Linear adaLN_modulation (expects a
            # tensor). Feed a constant zero tensor instead -- adaLN_modulation
            # then produces the SAME fixed shift/scale for every sample
            # regardless of input, i.e. an ordinary (non-conditional) layer
            # norm, matching the paper's "NC" definition.
            cond_emb = torch.zeros(len(times), self.cond_dim, device=device)

        return super().forward(batch, cond_emb)
