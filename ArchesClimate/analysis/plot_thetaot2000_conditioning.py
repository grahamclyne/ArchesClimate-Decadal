import warnings

from ArchesClimate.paths import config_directory
from ArchesClimate.paths import path as artifact_path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
from hydra import compose, initialize_config_dir

from ArchesClimate.analysis.gen_and_save_metrics import (
    load_predicted_and_target_ensembles,
)

CD = str(config_directory())
SURF = [
    "clt",
    "net_flux",
    "hurs",
    "huss",
    "pr",
    "ps",
    "psl",
    "sfcWind",
    "tasmax",
    "tasmin",
    "tas",
    "evspsbl",
    "thetaot2000",
    "thetaot300",
    "thetaot700",
    "t20d",
    "tos",
]
THETAOT2000_IDX = SURF.index("thetaot2000")

STARTS = [("test", 0, 1969), ("test", 40, 2009), ("val", 0, 2010)]
MODELS = {
    "2D-768-CL-C (conditioned)": ("table3_2d768_cl_c", "table3_2d768_cl_c"),
    "2D-768-CL-NC (no conditioning)": ("table3_2d768_cl_nc", "table3_2d768_cl_nc"),
}


def load_one(module, name, tgt, yr):
    with initialize_config_dir(version_base=None, config_dir=CD):
        cfg = compose(
            config_name="config",
            overrides=[
                f"module={module}",
                f"name={name}",
                "inference=inference_5",
                "inference.num_inference_steps=10",
                f"inference.target={tgt}",
                f"inference.year={yr}",
            ],
        )
    exp = f"{name}_10_5_1_118_1_{tgt}_{yr}_None"
    pe, te, n, _, _ = load_predicted_and_target_ensembles(cfg, exp, 5, 118)
    return pe, te[:, n:]


fig, axes = plt.subplots(1, len(STARTS), figsize=(6 * len(STARTS), 4.5), sharey=False)
if len(STARTS) == 1:
    axes = [axes]

_palette = {
    "2D-768-CL-C (conditioned)": "#d62728",
    "2D-768-CL-NC (no conditioning)": "#1f77b4",
}

for ax, (tgt, yr, cal_yr) in zip(axes, STARTS):
    tgt_series = None
    for label, (module, name) in MODELS.items():
        pe, te = load_one(module, name, tgt, yr)
        print(
            f"{label} pe surface shape: {pe['surface'].shape}, te surface shape: {te['surface'].shape}",
            flush=True,
        )
        # load_predicted_and_target_ensembles wraps both in an extra leading
        # batch dim (size 1) via [None] internally -- index it out (dim 0)
        # before reaching [members, T, var, lev, lat, lon].
        pred_thetaot = pe["surface"][
            0, :, :, THETAOT2000_IDX, 0
        ]  # [members, T, lat, lon]
        pred_spatial_mean = pred_thetaot.flatten(start_dim=2).nanmean(
            dim=-1
        )  # [members, T]
        x = np.arange(pred_spatial_mean.shape[1])
        for m in range(pred_spatial_mean.shape[0]):
            ax.plot(
                x,
                pred_spatial_mean[m].numpy(),
                color=_palette[label],
                linewidth=0.4,
                alpha=0.35,
            )
        ax.plot(
            x,
            pred_spatial_mean.mean(0).numpy(),
            color=_palette[label],
            linewidth=2.2,
            label=label,
        )

        if tgt_series is None:
            tgt_thetaot = te["surface"][0, :, :, THETAOT2000_IDX, 0]
            tgt_spatial_mean = tgt_thetaot.flatten(start_dim=2).nanmean(dim=-1)
            tgt_series = tgt_spatial_mean

    x = np.arange(tgt_series.shape[1])
    for m in range(tgt_series.shape[0]):
        ax.plot(
            x,
            tgt_series[m].numpy(),
            color="black",
            linewidth=0.4,
            alpha=0.35,
            linestyle=":",
        )
    ax.plot(
        x,
        tgt_series.mean(0).numpy(),
        color="black",
        linewidth=2.2,
        label="IPSL (target)",
        linestyle=":",
    )

    ax.set_title(f"{tgt} start {yr} (cal. {cal_yr})")
    ax.set_xlabel("Month of rollout")
    if ax is axes[0]:
        ax.set_ylabel("thetaot2000 (spatial mean)")

axes[0].legend(loc="upper left", fontsize=9, frameon=False)
plt.tight_layout()
out_path = str(artifact_path("plots") / "thetaot2000_conditioning_compare.pdf")
import os

os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved to {out_path}")
