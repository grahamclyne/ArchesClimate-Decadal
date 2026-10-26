import copy
import os
from copy import deepcopy

import lovely_tensors as lt
import pandas as pd
import torch
from diffusers.schedulers import (
    FlowMatchEulerDiscreteScheduler,
)
from hydra.utils import instantiate
from tensordict.tensordict import TensorDict
from tqdm import tqdm

from ArchesClimate.backbones.dit import TimestepEmbedder
from ArchesClimate.metrics.ensemble_metrics import EnsembleMetrics
from ArchesClimate.model.base_module import BaseLightningModule
from ArchesClimate.model.spectral_loss import OptimizedSpectralLoss
from ArchesClimate.paths import path as artifact_path
from ArchesClimate.utils.tensordict_utils import tensordict_apply, tensordict_cat
from ArchesClimate.utils.util_functions import (
    apply_iszero,
)

lt.monkey_patch()

geoarches_stats_path = artifact_path("stats")


class DCPPDiffusion(BaseLightningModule):
    def __init__(
        self,
        cfg=None,
        name="diffusion",
        add_input_state="default",
        cond_dim=256,
        num_train_timesteps=1000,
        scheduler="ddpm",
        prediction_type="v_prediction",
        beta_schedule="squaredcos_cap_v2",
        beta_start=0.0001,
        beta_end=0.012,
        snr_gamma=None,
        flow_matching=False,
        conditional=False,
        uncond_proba=0.0,
        ckpt_path=None,
        ft_lr=1e-5,
        snr_reweighting=False,  # this is useful for sample prediction.
        selected_vars=False,
        num_inference_steps=50,
        cf_guidance=1,
        num_members=1,
        debug=False,
        load_deterministic_model=False,
        loss_delta_normalization=False,
        pow=2,
        lr=1e-4,
        betas=(0.9, 0.98),
        weight_decay=1e-5,
        num_warmup_steps=1000,
        num_training_steps=300000,
        num_cycles=0.5,
        num_rollout_steps=10,
        num_batch_examples=1,
        dataset=None,
        month_conditioning=False,
        year_conditioning=False,
        timestamp_conditioning=False,
        s_churn=0,
        scale_input_noise=1,
        boundary_padding=False,
        prior="normal",
        load_prev=False,
        total_boundary_calc=False,
        save_v_fields=False,
        learn_residual=False,
        det_model_alpha=1.0,
        state_normalization=False,
        sd3_timestep_sampling=True,
        separate_forcing_embedder=False,
        forcing_embed=True,
        proper_year_forcings=False,
        per_variable_noise=False,
        use_spectral_loss=False,
        use_grad_loss=False,
        include_ssi=True,
        atmos_forcing_dim=5,
        state_scaler_source="pred",
    ):
        kwargs = locals().copy()
        # del kwargs['self']  # remove 'self' from the arguments
        super().__init__()
        self.__dict__.update(kwargs)

        # latitude weighting
        lat_coeffs_equi = torch.tensor(
            [
                torch.cos(x)
                for x in torch.arange(
                    -torch.pi / 2 + 1e-2, torch.pi / 2, torch.pi / 143
                )
            ]
        )
        lat_coeffs_equi = lat_coeffs_equi / lat_coeffs_equi.mean()
        self.loss_coeffs = TensorDict(
            surface=lat_coeffs_equi[None, None, None, :, None],
            level=lat_coeffs_equi[None, None, None, :, None],
        )
        self.loss_weights = {
            "mse": 1.0,
            "spectral": 0.0,
            "gradient": 0.0,
        }

        self.backbone = instantiate(cfg.backbone)  # necessary to put it on device
        self.embedder = instantiate(cfg.embedder)

        self.det_model = None
        if load_deterministic_model:
            # TODO: confirm that it works
            from ArchesClimate.model.base_module import AvgModule, load_module

            if isinstance(load_deterministic_model, str):
                self.det_model, _ = load_module(
                    load_deterministic_model, cfg.ckpt_fname
                )
            else:
                # load averaging model
                self.det_model = AvgModule(load_deterministic_model)

        self.month_embedder = TimestepEmbedder(cond_dim)
        self.hour_embedder = TimestepEmbedder(cond_dim)
        self.timestep_embedder = TimestepEmbedder(cond_dim)
        self.atmos_embedder = TimestepEmbedder(cond_dim)
        self.solar_embedder = TimestepEmbedder(cond_dim)

        # nn.ModuleList (not a plain list) so Lightning's automatic device
        # placement moves this metric to each DDP rank's own device; a plain
        # list + hardcoded .to('cuda') pins it to cuda:0 for every rank,
        # crashing non-zero ranks with "different devices in metric
        # calculation" as soon as validation_step runs on cuda:1+ (same bug
        # already fixed in model/dcpp_forecast.py's self.val_metrics).
        self.val_metrics = torch.nn.ModuleList(
            [EnsembleMetrics(data_shape=(1, len(cfg.surface_variables)))]
        )  # only one timestep
        if ckpt_path is not None:
            print(next(self.backbone.named_parameters()))
            print("init from ckpt", ckpt_path)
            self.init_from_ckpt(ckpt_path)
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps
        )

        if state_normalization == "pred":
            print(f"loading scalers (source={state_scaler_source})")
            if state_scaler_source == "clima":
                delta_path_surface = geoarches_stats_path.joinpath(
                    "dcpp_surface_delta_std"
                )
                delta_path_level = geoarches_stats_path.joinpath("dcpp_plev_delta_std")
                scaler_surface = (
                    torch.load(delta_path_surface)
                    .nanmean(axis=(-1, -2), keepdims=True)
                    .unsqueeze(1)
                    .unsqueeze(0)
                )
                scaler_plev = (
                    torch.load(delta_path_level)
                    .nanmean(axis=(-1, -2), keepdims=True)
                    .unsqueeze(0)
                )
            else:  # 'pred'
                # Delta stats must match the actual deterministic backbone this
                # flow model conditions on (residual std differs per backbone) --
                # derive the filename from load_deterministic_model rather than
                # hardcoding a single backbone's stats for every flow model.
                det_model_name = os.path.basename(
                    load_deterministic_model
                    if isinstance(load_deterministic_model, str)
                    else load_deterministic_model[0]
                )
                delta_path = geoarches_stats_path.joinpath(
                    f"dcpp_delta_std_{det_model_name}"
                )
                scaler_surface = (
                    torch.load(delta_path)["surface"]
                    .nanmean(axis=(-1, -2), keepdims=True)
                    .unsqueeze(1)
                    .unsqueeze(0)
                )
                scaler_plev = (
                    torch.load(delta_path)["level"]
                    .nanmean(axis=(-1, -2), keepdims=True)
                    .unsqueeze(0)
                )
            scaler = TensorDict({"surface": scaler_surface, "level": scaler_plev})
            self.state_scaler = scaler

        # elif((loss_delta_normalization == "pred" or state_normalization == "pred") and ghg_only):
        #     scaler_surface = torch.load(f'{geoarches_stats_path}/cmip_delta_std')['surface'].nanmean(axis=(-1,-2),keepdims=True).unsqueeze(0)
        #     scaler_plev = torch.load(f'{geoarches_stats_path}/cmip_delta_std')['level'].nanmean(axis=(-1,-2),keepdims=True).unsqueeze(0)
        #     scaler = TensorDict({'surface':scaler_surface,'level':scaler_plev})
        #     self.state_scaler = scaler

        # if loss_delta_normalization:
        #     self.loss_coeffs = tensordict_apply(
        #         torch.mul, self.loss_coeffs, (pangu_scaler / scaler).pow(self.pow)
        #     )

        self.inference_scheduler = deepcopy(self.noise_scheduler)

        if use_spectral_loss:
            self.spectral_criterion = OptimizedSpectralLoss()

        self.test_outputs = []
        self.validation_samples = {}

    def forward(
        self, batch, noisy_next_state, timesteps, use_condition=True, is_sampling=False
    ):
        device = batch["state"].device

        if type(use_condition) is bool:
            sel = torch.tensor(int(use_condition)).to(device)
        else:
            sel = use_condition.to(device)[:, None, None, None, None]

        input_state = noisy_next_state
        conditional_keys = self.conditional.split("+")  # all the things we condition on

        if "det" in conditional_keys:
            assert "pred_state" in batch
            pred_state = batch["pred_state"]

        if "prev" in conditional_keys:
            if self.load_prev > 1:
                for i in range(2):
                    input_state = tensordict_cat(
                        [
                            batch["prev_state"][:, i].apply(lambda x: x * sel),
                            input_state,
                        ],
                        dim=1,
                    )
            else:
                input_state = tensordict_cat(
                    [batch["prev_state"].apply(lambda x: x * sel), input_state], dim=1
                )

        if "det" in conditional_keys:
            input_state = tensordict_cat(
                [pred_state.apply(lambda x: x * sel), input_state], dim=1
            )

        timestep_emb = self.timestep_embedder(timesteps.reshape(-1))

        cond_emb = timestep_emb
        times = pd.to_datetime(batch["timestamp"].cpu().numpy(), unit="s").tz_localize(
            None
        )
        if self.month_conditioning:
            month = (torch.tensor(times.month) - 1 % 12).to(device)
            month_emb = self.month_embedder(month)
            cond_emb = cond_emb + month_emb
        if self.year_conditioning:
            year = (torch.tensor(times.year) - 1961).to(device)
            year_emb = self.hour_embedder(year)
            cond_emb = cond_emb + year_emb
        bound = len(batch["forcings"][0])
        if self.separate_forcing_embedder:
            for forcing_index in range(bound):
                is_atmos = forcing_index < self.atmos_forcing_dim
                if not is_atmos and not self.include_ssi:
                    continue
                if forcing_index < self.atmos_forcing_dim:
                    cond_emb += self.atmos_embedder(batch["forcings"][:, forcing_index])
                else:
                    cond_emb += self.solar_embedder(batch["forcings"][:, forcing_index])
        else:
            for forcing_index in range(bound):
                is_atmos = forcing_index < self.atmos_forcing_dim
                if not is_atmos and not self.include_ssi:
                    continue
                cond_emb += self.timestep_embedder(batch["forcings"][:, forcing_index])
        x = self.embedder.encode(batch["state"], input_state)
        x = self.backbone(x, cond_emb)
        out = self.embedder.decode(x)

        batch["state"]["surface"] = batch["state"]["surface"][:, :17]
        if self.add_input_state == "default":
            out += batch["state"]
        return out

    def validation_step(self, batch, batch_nb):
        val_num_members = 5
        samples = [
            self.sample(batch, seed=j, disable_tqdm=True)["surface"]
            for j in tqdm(range(val_num_members))
        ]
        for metric in self.val_metrics:
            for sample in samples:
                metric.update(
                    batch["next_state"]["surface"][:, :, 0],
                    torch.stack(samples)[None, :, :, 0],
                )
        self.validation_samples[batch_nb] = samples

    def on_validation_epoch_end(self):
        for metric in self.val_metrics:
            scores = metric.compute()
            for k, v in scores.items():
                scores[k] = v.mean()
            self.log_dict(scores, sync_dist=True)  # dont put on_epoch = True here
            metric.reset()
        self.validation_samples.clear()

    def get_scheduled_weights(self, global_step):
        # 0-5k: pure MSE
        # 5k-10k: fade in gradient loss (target 10.0)
        # 20k-25k: fade in spectral loss (target 35.0)
        grad_start, grad_end = 5000, 10000
        spec_start, spec_end = 10000, 15000
        target_grad, target_spec = 10.0, 35.0

        if global_step < grad_start:
            grad_weight = 0.0
        elif global_step < grad_end:
            grad_weight = (
                target_grad * (global_step - grad_start) / (grad_end - grad_start)
            )
        else:
            grad_weight = target_grad

        if global_step < spec_start:
            spec_weight = 0.0
        elif global_step < spec_end:
            spec_weight = (
                target_spec * (global_step - spec_start) / (spec_end - spec_start)
            )
        else:
            spec_weight = target_spec

        return {"mse": 1.0, "gradient": grad_weight, "spectral": spec_weight}

    def loss(
        self,
        pred,
        gt,
        implied_x1=None,
        gt_x1=None,
        timestep_weighting=None,
        multistep=False,
        **kwargs,
    ):
        loss_coeffs = self.loss_coeffs.to(self.device)

        mask = TensorDict(
            {k: (apply_iszero(v) if "state" in k else v) for k, v in gt.items()}
        )
        pred = TensorDict(
            {k: (v * mask[k] if "state" in k else v) for k, v in pred.items()}
        )

        weighted_error = (pred - gt).abs().pow(self.pow).mul(loss_coeffs)
        loss = sum(weighted_error.mean().values())

        if (self.use_spectral_loss or self.use_grad_loss) and implied_x1 is not None:
            w = (
                timestep_weighting.view(-1, 1, 1, 1, 1).to(self.device)
                if timestep_weighting is not None
                else 1.0
            )

            if self.use_spectral_loss and self.loss_weights["spectral"] > 0:
                spec_loss = sum(
                    (self.spectral_criterion(implied_x1[k], gt_x1[k]) * w).mean()
                    for k in ["surface", "level"]
                )
                loss = loss + self.loss_weights["spectral"] * spec_loss

            if self.use_grad_loss and self.loss_weights["gradient"] > 0:
                grad_loss = 0
                for grad_fn in [
                    lambda x: x[..., :, 1:] - x[..., :, :-1],
                    lambda x: x[..., 1:, :] - x[..., :-1, :],
                ]:
                    diff = (
                        (implied_x1.apply(grad_fn) - gt_x1.apply(grad_fn))
                        .abs()
                        .pow(self.pow)
                    )
                    grad_loss += sum(diff.apply(lambda x: x * w).mean().values())
                loss = loss + self.loss_weights["gradient"] * grad_loss

        return loss

    def sample_rollout(
        self,
        batch,
        index,
        test_dataset,
        *args,
        lead_time_months=1,
        disable_tqdm=False,
        debug=False,
        return_format="tensordict",
        seed,
        out_dir,
        ensemble_member_index,
        flat_forcings=False,
        **kwargs,
    ):
        torch.set_grad_enabled(False)
        preds_future = []
        loop_batch = {k: v for k, v in batch.items()}
        surface_mask = torch.where(batch["state"]["surface"][:, :17] == 0, 1.0, 0.0).to(
            loop_batch["state"]["surface"].device
        )
        level_mask = torch.where(batch["state"]["level"][:, :17] == 0, 1.0, 0.0).to(
            loop_batch["state"]["surface"].device
        )
        # put in t0 for sanity check that we can use to compare against test/persistence, i.e. do we start from the same spot
        first_step = deepcopy(batch)
        first_step["state"]["surface"] = first_step["state"]["surface"][:, :17]
        preds_future.append(first_step["state"])

        # Captured once, before the loop, so flat_forcings holds the GHG/atmos
        # level fixed at whatever year the rollout actually starts from
        # (matches dcpp_forecast.py's forward_multistep) instead of a
        # hardcoded 1970 -- e.g. a 1960-start rollout now correctly holds at
        # 1960 forcing levels, not an unrelated fixed year.
        first_timestamp = test_dataset.next_timestamp_map[
            loop_batch["timestamp"].item()
        ]
        first_times = pd.to_datetime(first_timestamp, unit="s").tz_localize(None)
        if not self.proper_year_forcings:
            first_year = torch.tensor(first_times.year) - 1961
        else:
            first_year = torch.tensor(first_times.year)
        first_solar_year = torch.tensor(first_times.year) - 1961

        for i in tqdm(range(lead_time_months), disable=disable_tqdm):
            loop_batch = {k: v.to(self.device) for k, v in loop_batch.items()}

            sample = self.sample(
                loop_batch,
                *args,
                **kwargs,
                disable_tqdm=True,
                seed=seed * 100000 + i,
                cf_guidance=int(self.cf_guidance),
            )
            sample.batch_size = [1]
            if int(self.load_prev) > 1:
                loop_batch["prev_state"] = torch.stack(
                    [loop_batch["state"], loop_batch["prev_state"][:, 0]], dim=1
                )
            else:
                loop_batch["prev_state"] = loop_batch["state"].copy()

            loop_batch["state"] = TensorDict(sample)
            loop_batch["state"].batch_size = [1]

            next_timestamp = test_dataset.next_timestamp_map[
                loop_batch["timestamp"].item()
            ]
            times = pd.to_datetime(next_timestamp, unit="s").tz_localize(None)
            if not self.proper_year_forcings:
                current_year = torch.tensor(times.year) - 1961  # to zero index
            else:
                current_year = torch.tensor(times.year)  # -1961 to zero index

            current_month = torch.tensor(times.month) - 1 % 12
            current_solar_year = torch.tensor(times.year) - 1961  # -1961 to zero index

            if flat_forcings:
                next_forcings = torch.concatenate(
                    [
                        test_dataset.atmos_forcings[first_year, :],
                        test_dataset.solar_forcings[
                            (first_solar_year * 12) + current_month, :
                        ],
                    ]
                ).to(loop_batch["state"].device)
            else:
                next_forcings = torch.concatenate(
                    [
                        test_dataset.atmos_forcings[current_year, :],
                        test_dataset.solar_forcings[
                            (current_solar_year * 12) + current_month, :
                        ],
                    ]
                ).to(loop_batch["state"].device)
            loop_batch["forcings"] = next_forcings[None]
            loop_batch["timestamp"] = torch.tensor(next_timestamp).to(
                loop_batch["state"].device
            )[None]
            loop_batch["state"]["surface"] = torch.where(
                surface_mask == 1.0, 0, loop_batch["state"]["surface"]
            )
            loop_batch["prev_state"]["surface"] = loop_batch["prev_state"]["surface"][
                :, :17
            ]
            loop_batch["prev_state"]["surface"] = torch.where(
                surface_mask == 1.0, 0, loop_batch["prev_state"]["surface"]
            )
            loop_batch["state"]["level"] = torch.where(
                level_mask == 1.0, 0, loop_batch["state"]["level"]
            )
            loop_batch["prev_state"]["level"] = torch.where(
                level_mask == 1.0, 0, loop_batch["prev_state"]["level"]
            )
            preds_future.append(loop_batch["state"])
            if self.learn_residual == "pred":
                del loop_batch["pred_state"]

            # reset for longer rollouts
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

    def training_step(self, batch, batch_nb):
        device, bs = batch["state"].device, batch["state"].shape[0]

        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=device
        ).long()

        if self.use_spectral_loss or self.use_grad_loss:
            self.loss_weights = self.get_scheduled_weights(self.global_step)

        if self.prior == "student":
            from torch.distributions.studentT import StudentT

            noise = TensorDict(
                {
                    k: StudentT(3, loc=0.0, scale=1.0).rsample(v.shape).to(device)
                    for k, v in batch["next_state"].items()
                }
            )
        else:
            noise = torch.randn_like(batch["next_state"])
        next_state = batch["next_state"]

        if self.learn_residual == "default":
            next_state = batch["next_state"] - batch["state"]

        elif self.learn_residual == "pred":
            if "pred_state" not in batch:
                with torch.no_grad():
                    batch["pred_state"] = self.det_model(batch).detach()
            pred_state = (
                self.det_model_alpha * batch["pred_state"]
                + (1 - self.det_model_alpha) * batch["state"]
            )
            next_state = batch["next_state"] - pred_state
        if self.state_normalization:
            next_state = tensordict_apply(
                torch.div, next_state, self.state_scaler.to(self.device)
            )

        if self.scheduler == "flow":
            # weighting scheme: logit normal
            u = torch.normal(mean=0, std=1, size=(bs,), device="cpu").sigmoid()

            indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
            if self.sd3_timestep_sampling:
                u = torch.normal(mean=0, std=1, size=(bs,), device="cpu").sigmoid()
                indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
            else:
                indices = timesteps.cpu()
            timesteps = self.noise_scheduler.timesteps[indices].to(device)

            schedule_timesteps = self.noise_scheduler.timesteps.to(device)
            sigmas = self.noise_scheduler.sigmas.to(
                device=device, dtype=next(iter(batch["state"].values())).dtype
            )

            step_indices = [
                (schedule_timesteps == t).nonzero().item() for t in timesteps
            ]
            sigma = sigmas[step_indices].flatten()[
                :, None, None, None, None
            ]  # shape (bs,)

            noisy_next_state = noise.apply(lambda x: x * sigma) + next_state.apply(
                lambda x: x * (1.0 - sigma)
            )
            target_state = noise - next_state
            if self.prediction_type == "sample":
                target_state = next_state

        else:
            noisy_next_state = tensordict_apply(
                self.noise_scheduler.add_noise, next_state, noise, timesteps
            ).to(device)

            # Get the target for loss depending on the prediction type
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target_state = noise

            elif self.noise_scheduler.config.prediction_type == "sample":
                target_state = next_state

            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target_state = tensordict_apply(
                    self.noise_scheduler.get_velocity,
                    next_state,
                    noise,
                    timesteps,
                )

            else:
                raise ValueError(
                    f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                )

        # create uncond
        use_condition = torch.rand((bs,), device=device) > self.uncond_proba

        pred = self.forward(
            batch,
            noisy_next_state,
            timesteps,
            use_condition.float(),
        )

        # implied_x1: reconstruct the target state from the velocity prediction
        # weighted by (1-sigma)^2 so spec/grad losses are stronger near t=0
        implied_x1 = None
        timestep_weighting = None
        if (self.use_spectral_loss or self.use_grad_loss) and self.scheduler == "flow":
            sigma_1d = sigma[:, 0, 0, 0, 0]
            implied_x1 = noisy_next_state - pred.apply(
                lambda x: x * sigma_1d[:, None, None, None, None]
            )
            timestep_weighting = (1.0 - sigma_1d).pow(2)

        loss = self.loss(
            pred,
            target_state,
            implied_x1=implied_x1,
            gt_x1=next_state,
            timestep_weighting=timestep_weighting,
        )

        self.mylog(loss=loss)

        return loss

    def sample(
        self,
        batch,
        seed=None,
        num_steps=None,
        cf_guidance=None,
        disable_tqdm=False,
        rescale=1.0,
        **kwargs,
    ):
        """
        kwargs args are fed to scheduler_step
        """

        if cf_guidance is None:
            cf_guidance = int(self.cf_guidance)

        scheduler = self.inference_scheduler
        num_steps = num_steps or self.num_inference_steps
        scheduler.set_timesteps(num_steps)
        if seed is not None:
            torch.manual_seed(seed)

        # get args to scheduler
        if self.scheduler == "flow":
            scheduler_kwargs = {"s_churn": self.s_churn}

        else:
            # TODO: put some churning here
            scheduler_kwargs = {}  # dict(eta=self.cfg.inference.eta)

        # scheduler_kwargs.update(kwargs)
        if self.learn_residual == "pred" and "pred_state" not in batch:
            with torch.no_grad():
                batch["pred_state"] = self.det_model(batch).detach()
        if self.prior == "student":
            from torch.distributions.studentT import StudentT

            noisy_state = TensorDict(
                {
                    k: StudentT(3, loc=0.0, scale=1.0)
                    .rsample(v.shape)
                    .to(batch["state"].device)
                    for k, v in batch["state"].items()
                }
            )
        else:
            noisy_state = batch["state"].apply(torch.randn_like)  # amazing it works

        scale_input_noise = self.scale_input_noise
        noisy_state["surface"] = noisy_state["surface"][:, :17]
        if self.per_variable_noise:
            print("per_variable_noise")
            surface_scales = (
                torch.tensor(
                    [
                        0.0389,
                        0.0164,
                        0.0351,
                        0.0150,
                        -0.1449,
                        0.0702,
                        0.0719,
                        0.0365,
                        0.0231,
                        0.0221,
                        0.0225,
                        0.0170,
                        0.0188,
                        0.0271,
                        0.0314,
                        0.0397,
                        0.0202,
                    ]
                ).to(batch["state"]["surface"].device)
                * 2
            ) + 1
            level_scales = (
                torch.tensor(
                    [
                        [0.0555, 0.0458, 0.0364, 0.0470],
                        [0.0193, 0.0144, -0.0004, -0.0083],
                        [0.0264, 0.0331, 0.0300, 0.0300],
                        [0.0628, 0.0681, 0.0688, 0.0560],
                        [0.0225, 0.0198, 0.0231, 0.0274],
                        [0.0750, 0.0946, 0.1087, 0.0878],
                        [0.0671, 0.0579, 0.0439, 0.0312],
                    ]
                ).to(batch["state"]["surface"].device)
                + 1
            )
            scale_input_noise = TensorDict(
                surface=surface_scales[None, ..., None, None, None],
                level=level_scales[None, ..., None, None],
            )

        if scale_input_noise is not None:
            noisy_state = noisy_state * scale_input_noise
        loop_batch = {
            k: v.clone().detach() for k, v in batch.items() if "next" not in k
        }  # ensures no data leakage
        v_fields = []
        with torch.no_grad():
            for t in tqdm(scheduler.timesteps, disable=disable_tqdm):
                pred = self.forward(
                    deepcopy(loop_batch),
                    noisy_state,
                    timesteps=torch.tensor([t]).to(self.device),
                    use_condition=(cf_guidance > 0),
                    is_sampling=True,
                )
                if cf_guidance > 1:
                    uncond_pred = self.forward(
                        loop_batch,
                        noisy_state,
                        timesteps=torch.tensor([t]).to(self.device),
                        use_condition=False,
                        is_sampling=True,
                    )
                    # compute epsilon from uncond_pred
                    epsilon = pred - uncond_pred
                    # pred = pred + cf_guidance * epsilon
                    pred = epsilon.apply(lambda x: cf_guidance * x) + pred
                if self.save_v_fields:
                    v_fields.append(pred)
                step_index = getattr(scheduler, "_step_index", None)

                def scheduler_step(*args, _step_index=step_index, **kwargs):
                    out = scheduler.step(*args, **kwargs)

                    if hasattr(scheduler, "_step_index"):
                        scheduler._step_index = _step_index
                    return out.prev_sample

                pred_stacked = torch.concatenate(
                    [
                        pred["surface"].squeeze(2),
                        pred["level"].reshape(
                            -1, len(self.cfg.level_variables) * 4, 143, 144
                        ),
                    ],
                    dim=1,
                )
                noisy_state_stacked = torch.concatenate(
                    [
                        noisy_state["surface"].squeeze(2),
                        noisy_state["level"].reshape(
                            -1, len(self.cfg.level_variables) * 4, 143, 144
                        ),
                    ],
                    dim=1,
                )
                out = scheduler_step(
                    pred_stacked, t, noisy_state_stacked, **scheduler_kwargs
                )
                noisy_state["level"] = out[
                    :, len(self.cfg.surface_variables) :
                ].reshape(-1, len(self.cfg.level_variables), 4, 143, 144)
                noisy_state["surface"] = out[
                    :, : len(self.cfg.surface_variables)
                ].unsqueeze(2)

                # at the end
                if step_index is not None:
                    scheduler._step_index = step_index + 1

        if self.save_v_fields:
            torch.save(torch.stack(v_fields), f"v_fields_{seed}.pt")

        final_state = noisy_state.detach()
        if self.state_normalization:
            final_state = tensordict_apply(
                torch.mul, final_state, self.state_scaler.to(self.device)
            )
        if self.learn_residual == "default":
            final_state = batch["state"] + final_state
        elif self.learn_residual == "pred":
            pred_state = (
                self.det_model_alpha * batch["pred_state"]
                + (1 - self.det_model_alpha) * batch["state"]
            )
            final_state = pred_state + final_state

        return final_state

    def generate_rollouts(
        self,
        batch_index,
        test_dataset,
        target_name,
        out_dir,
        start_member,
        end_member,
        flat_forcings,
        num_rollout_steps,
    ):
        os.makedirs(out_dir, exist_ok=True)
        for ensemble_member in range(start_member, end_member):
            b_index = batch_index + (
                ensemble_member * 120
            )  # get different IC for each ensemble member
            batch = test_dataset.__getitem__(b_index)
            batch = {k: batch[k].unsqueeze(0) for k in batch}
            batch = {k: batch[k].to(self.device) for k in batch}

            rollout = self.sample_rollout(
                copy.deepcopy(batch),
                lead_time_months=self.num_rollout_steps,
                seed=(batch_index * 10) + ensemble_member,
                test_dataset=test_dataset,
                index=b_index,
                out_dir=out_dir,
                ensemble_member_index=ensemble_member,
                flat_forcings=flat_forcings,
            )
            torch.save(
                rollout["surface"], f"{out_dir}/rollout_{ensemble_member}_surface.pt"
            )
            torch.save(
                rollout["level"], f"{out_dir}/rollout_{ensemble_member}_level.pt"
            )
