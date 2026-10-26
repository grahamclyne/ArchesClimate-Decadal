"""Member + hindcast-start bootstrap of Table 4/5 (flow ablations) rows for ONE model.
Stores time-mean [mse, frmse, spskr(ESR)] per row so the RMSE definition can be chosen afterwards.
Usage: python bootstrap_table45.py NAME MODULE CKPT N_BOOT [extra hydra overrides...]
NAME=IPSL scores the IPSL ensemble against itself (no rollouts needed).
The same seed => identical resample indices for every model (paired comparisons)."""

import sys
import time
import warnings

from ArchesClimate.paths import analysis_device, config_directory
from ArchesClimate.paths import path as artifact_path

warnings.filterwarnings("ignore")
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from tensordict.tensordict import TensorDict

import ArchesClimate.analysis.evaluate as ev
from ArchesClimate.analysis.gen_and_save_metrics import (
    _yearly_mean,
    load_predicted_and_target_ensembles,
)

DEVICE = analysis_device()
_O = ev.EnsembleMetrics
ev.EnsembleMetrics = lambda *a, **k: _O(*a, **k).to(DEVICE)
CD = str(config_directory())
OUT = str(artifact_path("results") / "bootstrap_table45_results")
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
LEV = ["hur", "hus", "ta", "ua", "va", "wap", "zg"]
STARTS = [
    ("test", 0),
    ("test", 10),
    ("test", 20),
    ("test", 30),
    ("test", 40),
    ("val", 0),
    ("val", 10),
    ("val", 20),
    ("val", 30),
    ("val", 40),
    ("val", 50),
]
ROWS = [
    ("tas", "s", "tas", None, "all"),
    ("tas (land)", "s", "tas", None, "land"),
    ("tas (yearly)", "s", "tas", None, "yearly"),
    ("tas (land, yearly)", "s", "tas", None, "land_yearly"),
    ("thetaot2000", "s", "thetaot2000", None, "all"),
    ("tos", "s", "tos", None, "all"),
    ("psl", "s", "psl", None, "all"),
    ("net_flux", "s", "net_flux", None, "all"),
    ("pr", "s", "pr", None, "all"),
    ("hur 700", "l", "hur", 1, "all"),
    ("ta 700", "l", "ta", 1, "all"),
    ("wap 700", "l", "wap", 1, "all"),
    ("zg 700", "l", "zg", 1, "all"),
]
METRICS = [0, 1, 3]  # mse, frmse, spskr (ESR)


def cuda(td):
    o = TensorDict(
        {"surface": td["surface"].to(DEVICE), "level": td["level"].to(DEVICE)}
    )
    o.batch_size = td.batch_size
    return o


def sub(td, idx):
    i = torch.as_tensor(idx, dtype=torch.long, device=DEVICE)
    o = TensorDict(
        {
            "surface": td["surface"].index_select(1, i),
            "level": td["level"].index_select(1, i),
        }
    )
    o.batch_size = [1, len(idx), td.batch_size[2]]
    return o


def land(td, ocean):
    s = td["surface"].clone()
    s[:, :, :, 10, 0] = s[:, :, :, 10, 0].masked_fill(ocean, float("nan"))
    o = TensorDict({"surface": s, "level": td["level"]})
    o.batch_size = list(td.batch_size)
    return o


def rows_from(pred, targ, ocean):
    m = ev.generate_metrics(pred, targ)
    ml = ev.generate_metrics(land(pred, ocean), land(targ, ocean))
    my = ev.generate_metrics(_yearly_mean(pred), _yearly_mean(targ))
    mly = ev.generate_metrics(
        land(_yearly_mean(pred), ocean), land(_yearly_mean(targ), ocean)
    )
    res = np.zeros((len(METRICS), len(ROWS)))
    for r, (_, k, v, pi, mode) in enumerate(ROWS):
        mm = {"all": m, "land": ml, "yearly": my, "land_yearly": mly}[mode]
        t = mm["surface"] if k == "s" else mm["level"]
        if "yearly" in mode:
            t = t[-2:]
        for a, mi in enumerate(METRICS):
            x = (
                t.nanmean(dim=0)[mi, SURF.index(v)]
                if k == "s"
                else t.nanmean(dim=0)[mi, LEV.index(v), pi]
            )
            res[a, r] = x.item()
    return res


def main():
    name, module, ck, nboot = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    extra = sys.argv[5:]
    itself = name == "IPSL"
    data = []
    for tgt, yr in STARTS:
        with initialize_config_dir(version_base=None, config_dir=CD):
            cfg = compose(
                config_name="config",
                overrides=[
                    f"module={module}",
                    f"name={name}",
                    "inference=inference_25",
                    "inference.num_inference_steps=24",
                    f"inference.target={tgt}",
                    f"inference.year={yr}",
                ]
                + extra,
            )
        exp = f"{'flow_ensemble_single_small_cl' if itself else name}_24_5_1_118_1_{tgt}_{yr}_{ck}"
        pe, te, n, _, _ = load_predicted_and_target_ensembles(cfg, exp, 5, 118)
        if itself:
            pe = te[:, :n]
        ocean = (~te["surface"][0, 0, 0, -1, 0].isnan()).to(DEVICE)
        data.append((cuda(pe), cuda(te[:, n:]), n, ocean))
        print("loaded", tgt, yr, flush=True)
    pt = np.array([rows_from(p, t, o) for p, t, n, o in data])  # [starts, metric, row]
    print(
        "point estimate (mean over starts) [mse, frmse, esr]:",
        pt.mean(0).round(4).tolist(),
        flush=True,
    )
    rng = np.random.default_rng(0)
    reps = np.zeros((nboot, len(METRICS), len(ROWS)))
    t0 = time.time()
    for b in range(nboot):
        starts = rng.integers(0, len(data), size=len(data))
        acc = []
        for s in starts:
            p, t, n, o = data[s]
            acc.append(
                rows_from(
                    sub(
                        p,
                        rng.integers(
                            0, p["surface"].shape[1], size=p["surface"].shape[1]
                        ),
                    ),
                    sub(
                        t,
                        rng.integers(
                            0, t["surface"].shape[1], size=t["surface"].shape[1]
                        ),
                    ),
                    o,
                )
            )
        reps[b] = np.mean(acc, axis=0)
        if b % 20 == 0:
            print("draw", b, f"{time.time() - t0:.0f}s", flush=True)
    import os

    os.makedirs(OUT, exist_ok=True)
    np.savez(
        f"{OUT}/{name}.npz",
        ssr_normalization="ensemble_mean_v1",
        ssr_pred_members=5,
        ssr_reference_members=5,
        per_start=pt,
        reps=reps,
        rows=[r[0] for r in ROWS],
        metrics=["mse", "frmse", "esr"],
        starts=[f"{a}_{b}" for a, b in STARTS],
    )
    print("saved", f"{OUT}/{name}.npz")


main()
