"""Member + hindcast-start bootstrap of DiffESM's tas rows (RMSE + SSR), for
either the 'old' (no month conditioning, matches the paper's current baseline)
or 'new_monthcond' (new per-frame month-of-year conditioning) checkpoint --
lets us directly compare whether month conditioning improved DiffESM, and
gives DiffESM the same RMSE/SSR + bootstrap CI treatment as every other row.
Same real-target-loading trick as bootstrap_mesmer.py.
Usage: python bootstrap_diffesm.py SAVE_NAME N_BOOT   (SAVE_NAME: old | new_monthcond)"""

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
from ArchesClimate.analysis.gen_baseline_metrics import load_diffesm

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
    ("test", 0, 1969),
    ("test", 10, 1979),
    ("test", 20, 1989),
    ("test", 30, 1999),
    ("test", 40, 2009),
    ("val", 0, 2010),
    ("val", 10, 2011),
    ("val", 20, 2012),
    ("val", 30, 2013),
    ("val", 40, 2014),
    ("val", 50, 2015),
]
ROWS = [
    ("tas", "s", "tas", None, "all"),
    ("tas (land)", "s", "tas", None, "land"),
    ("tas (yearly)", "s", "tas", None, "yearly"),
    ("tas (land, yearly)", "s", "tas", None, "land_yearly"),
]
METRICS = [0, 1, 3]  # mse, frmse, spskr (SSR)
NUM_ROLLOUT_STEPS = 118
COMPANION_MODULE, COMPANION_NAME, COMPANION_CK = (
    "flow_ensemble_single_small",
    "flow_ensemble_single_small_cl",
    "epoch-epoch=19.ckpt",
)


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
    save_name, nboot = sys.argv[1], int(sys.argv[2])
    data = []
    for tgt, yr, diffesm_yr in STARTS:
        with initialize_config_dir(version_base=None, config_dir=CD):
            cfg = compose(
                config_name="config",
                overrides=[
                    f"module={COMPANION_MODULE}",
                    f"name={COMPANION_NAME}",
                    "inference=inference_25",
                    "inference.num_inference_steps=24",
                    f"inference.target={tgt}",
                    f"inference.year={yr}",
                ],
            )
        exp = f"{COMPANION_NAME}_24_5_1_118_1_{tgt}_{yr}_{COMPANION_CK}"
        _, te, n, _, _ = load_predicted_and_target_ensembles(
            cfg, exp, 5, NUM_ROLLOUT_STEPS
        )
        num_surface_vars = len(cfg.module.surface_variables)
        num_level_vars = len(cfg.module.level_variables)
        num_plevs = len(cfg.module.pressure_levels)
        # DiffESM's raw data starts at real calendar January, but `te` (like every
        # ArchesClimate rollout) starts at real February -- forward_multistep's
        # is_sampling branch saves t=0 as the echoed 2nd input state (Jan+Feb are
        # the 2-month init pair; the first genuine prediction, March, is t=1), and
        # utils/generate_test_val_ensembles.py uses the identical dataset indexing
        # for the target files, so target/pred agree at t=0=February. See the
        # matching `sm = start_month - 1` correction in plot_climatology.py for the
        # same DiffESM/MESMER-M calendar issue in the climatology figure. Request 1
        # extra month and drop it (real January) so pe's t=0 is February, matching
        # te's convention, before scoring -- without this, every monthly RMSE/SSR
        # is computed against a 1-month-misaligned target.
        MONTH_OFFSET = 1
        pe = load_diffesm(
            diffesm_yr,
            NUM_ROLLOUT_STEPS + MONTH_OFFSET,
            num_surface_vars,
            num_level_vars,
            num_plevs,
            None,
            None,
            0,
            None,
            save_name=save_name,
        )
        pe = TensorDict(
            {
                "surface": pe["surface"][:, :, MONTH_OFFSET:],
                "level": pe["level"][:, :, MONTH_OFFSET:],
            }
        )
        # DiffESM's later val starts (2011-2015) have progressively less real
        # target data available for that held-out realization and so generate
        # correspondingly fewer months -- use the actual overlap, not a fixed 118.
        T = pe["surface"].shape[2]
        pe.batch_size = [1, 5, T]
        te = te[:, :, :T]
        print("loaded", tgt, yr, "diffesm_year", diffesm_yr, "T=", T, flush=True)
        ocean = (~te["surface"][0, 0, 0, -1, 0].isnan()).to(DEVICE)
        data.append((cuda(pe), cuda(te[:, n:]), n, ocean))
    pt = np.array([rows_from(p, t, o) for p, t, n, o in data])  # [starts, metric, row]
    print(
        "point estimate (mean over starts) [mse, frmse, ssr]:",
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
    outname = f"diffesm_{save_name}"
    np.savez(
        f"{OUT}/{outname}.npz",
        ssr_normalization="ensemble_mean_v1",
        ssr_pred_members=5,
        ssr_reference_members=5,
        per_start=pt,
        reps=reps,
        rows=[r[0] for r in ROWS],
        metrics=["mse", "frmse", "ssr"],
        starts=[f"{a}_{b}" for a, b, _ in STARTS],
    )
    print("saved", f"{OUT}/{outname}.npz")


main()
