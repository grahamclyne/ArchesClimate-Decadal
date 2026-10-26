from pathlib import Path

"""Same 50-year (1970-start) rollout data as plot_50yr_paper_style.py, but
plotted as ANOMALIES from each series' own baseline rather than absolute
values -- sidesteps the climatological bias discussion already in the paper
and isolates whether the FORCED RESPONSE (warming trend) is captured
correctly, independent of any absolute offset.

Baseline convention: model series (forced, flat) are anomalies from the
MODEL's own first-decade (1971-1980) mean. Real series (1970-init, 2010-init)
are anomalies from the REAL 1970-init decade's own mean -- since both real
series share one consistent physical dataset/scale, this lets the real
2010-init anomaly show the true observed warming relative to the 1970s,
directly comparable to the model's own captured warming relative to its own
1970s baseline.
"""

import matplotlib

from ArchesClimate.paths import config_directory
from ArchesClimate.paths import path as artifact_path

matplotlib.use("Agg")
import hydra
import matplotlib.pyplot as plt
import torch
from hydra import compose, initialize_config_dir
from tensordict.tensordict import TensorDict

CONFIGS_DIR = str(config_directory())
WORK = str(artifact_path("long_rollouts"))
NUM_MEMBERS = 10
NUM_ROLLOUT_STEPS = 600
CHUNK_BOUNDARIES = [118, 238, 358, 478, 598]
REAL_HINDCAST_MONTHS = 120
MODEL_BATCH_INDEX_OFFSET = 10800


def get_cfg(overrides):
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        return compose(config_name="config", overrides=overrides)


_ref_cfg = get_cfg(
    [
        "module=flow_ensemble_single_not_spectral",
        "name=flow_ensemble_single_not_spectral_1",
    ]
)
TOS_IDX = _ref_cfg.module.surface_variables.index("tos")
RAW_TOS_IDX = list(_ref_cfg.dataloader.dataset.surface_variables).index("tos")
train_mean_only_spatial_norm = hydra.utils.instantiate(
    _ref_cfg.dataloader.dataset,
    domain="all",
    limit_examples=100,
    norm_scheme="mean_only_spatial_norm",
)


def yearly_mean(monthly):
    T = monthly.shape[-1]
    n_years = T // 12
    return (
        monthly[..., : n_years * 12]
        .reshape(*monthly.shape[:-1], n_years, 12)
        .mean(dim=-1)
    )


def load_model_rollout_tos(exp_dir, batch_index_offset=0):
    members = []
    for m in range(NUM_MEMBERS):
        b_index = batch_index_offset + m * 120
        chunks = []
        for i in CHUNK_BOUNDARIES:
            surf = torch.load(
                f"{WORK}/{exp_dir}/rollout_{b_index}_surface_{i}_{m}.pt",
                map_location="cpu",
            )
            lev = torch.load(
                f"{WORK}/{exp_dir}/rollout_{b_index}_level_{i}_{m}.pt",
                map_location="cpu",
            )
            chunk_td = TensorDict({"surface": surf, "level": lev})
            chunk_td.batch_size = list(surf.shape[:2])
            chunk_td = train_mean_only_spatial_norm.denormalize(
                chunk_td[None], stateless=True
            )
            chunks.append(
                chunk_td["surface"][0, 0, :, TOS_IDX, 0].nanmean(dim=(-1, -2))
            )
        members.append(torch.cat(chunks))
    return torch.stack(members)


def load_real_hindcast_tos(domain, num_members=10, start_b_index=0):
    ds = hydra.utils.instantiate(_ref_cfg.dataloader.dataset, domain=domain)
    members = []
    for m in range(num_members):
        b_index = start_b_index + m * 120
        surfs, levs = [], []
        for i in range(REAL_HINDCAST_MONTHS):
            item = ds.__getitem__(b_index + i, normalize=True)
            surfs.append(item["state"]["surface"])
            levs.append(item["state"]["level"])
        surf = torch.stack(surfs)
        lev = torch.stack(levs)
        td = TensorDict({"surface": surf, "level": lev})
        td.batch_size = [surf.shape[0]]
        td = train_mean_only_spatial_norm.denormalize(td[None], stateless=True)
        members.append(td["surface"][0, :, RAW_TOS_IDX, 0].nanmean(dim=(-1, -2)))
    return torch.stack(members)


model_forced = load_model_rollout_tos(
    "flow_ensemble_single_not_spectral_1_1970start_25_10_1_600_1_train_90_None",
    batch_index_offset=MODEL_BATCH_INDEX_OFFSET,
)
model_flat = load_model_rollout_tos(
    "flow_ensemble_single_not_spectral_1_1970start_flat_25_10_1_600_1_train_90_None",
    batch_index_offset=MODEL_BATCH_INDEX_OFFSET,
)
model_forced_yearly = yearly_mean(model_forced)
model_flat_yearly = yearly_mean(model_flat)

real_1970 = load_real_hindcast_tos("train", start_b_index=MODEL_BATCH_INDEX_OFFSET)
real_2010 = load_real_hindcast_tos("val")
real_1970_yearly = yearly_mean(real_1970)
real_2010_yearly = yearly_mean(real_2010)

# --- baselines ---
model_baseline = model_forced_yearly[
    :, :10
].mean()  # model's own first-decade mean (forced run)
model_flat_baseline = model_flat_yearly[
    :, :10
].mean()  # flat run's own first-decade mean
real_baseline = real_1970_yearly.mean()  # real 1970s decade's own mean

print("model_baseline (forced, 1971-1980):", model_baseline.item())
print("model_flat_baseline (1971-1980):", model_flat_baseline.item())
print("real_baseline (1970s):", real_baseline.item())

model_forced_anom = model_forced_yearly - model_baseline
model_flat_anom = model_flat_yearly - model_flat_baseline
real_1970_anom = real_1970_yearly - real_baseline
real_2010_anom = (
    real_2010_yearly - real_baseline
)  # SAME real baseline -- shows true observed warming

years_model = list(range(1971, 1971 + model_forced_anom.shape[1]))
years_real_1970 = list(range(1971, 1971 + real_1970_anom.shape[1]))
years_real_2010 = list(range(2011, 2011 + real_2010_anom.shape[1]))

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    years_model,
    model_forced_anom.mean(dim=0).numpy(),
    color="tab:blue",
    label="ArchesClimate (forced) anomaly",
    linewidth=2,
)
ax.fill_between(
    years_model,
    (model_forced_anom.mean(0) - model_forced_anom.std(0)).numpy(),
    (model_forced_anom.mean(0) + model_forced_anom.std(0)).numpy(),
    color="tab:blue",
    alpha=0.15,
)

ax.plot(
    years_model,
    model_flat_anom.mean(dim=0).numpy(),
    color="tab:orange",
    label="ArchesClimate (flat forcing) anomaly",
    linewidth=2,
    linestyle="--",
)
ax.fill_between(
    years_model,
    (model_flat_anom.mean(0) - model_flat_anom.std(0)).numpy(),
    (model_flat_anom.mean(0) + model_flat_anom.std(0)).numpy(),
    color="tab:orange",
    alpha=0.15,
)

ax.plot(
    years_real_1970,
    real_1970_anom.mean(dim=0).numpy(),
    color="black",
    label="IPSL-DCPP real (1970 init) anomaly",
    linewidth=2,
)
ax.fill_between(
    years_real_1970,
    (real_1970_anom.mean(0) - real_1970_anom.std(0)).numpy(),
    (real_1970_anom.mean(0) + real_1970_anom.std(0)).numpy(),
    color="black",
    alpha=0.12,
)

ax.plot(
    years_real_2010,
    real_2010_anom.mean(dim=0).numpy(),
    color="dimgray",
    label="IPSL-DCPP real (2010 init) anomaly",
    linewidth=2,
    linestyle=":",
)
ax.fill_between(
    years_real_2010,
    (real_2010_anom.mean(0) - real_2010_anom.std(0)).numpy(),
    (real_2010_anom.mean(0) + real_2010_anom.std(0)).numpy(),
    color="dimgray",
    alpha=0.12,
)

ax.axhline(0, color="gray", linewidth=0.5)
ax.set_xlabel("Year")
ax.set_ylabel("tos anomaly (K, relative to own 1970s baseline)")
ax.legend()
fig.tight_layout()
out_path = str(artifact_path("plots") / "fifty_year_anomaly.png")
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
paper_path = str(artifact_path("plots") / "50_year_rollout.pdf")
fig.savefig(paper_path)
print(f"Saved to {paper_path}")

print()
print(
    "final-decade (2011-2020) anomaly, model forced:",
    model_forced_anom[:, -10:].mean().item(),
)
print("final-decade (2011-2020) anomaly, real 2010-init:", real_2010_anom.mean().item())
print(
    "final-decade (2011-2020) anomaly, model flat:",
    model_flat_anom[:, -10:].mean().item(),
)
