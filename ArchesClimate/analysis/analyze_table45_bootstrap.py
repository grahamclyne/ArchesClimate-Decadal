import os

from ArchesClimate.analysis.ssr_normalization import load_paper_bootstrap
from ArchesClimate.paths import reference_results

D = str(reference_results("bootstrap_table45_results"))
COLS = [
    ("2D-E192-CL", "flow_ensemble_small_cl"),
    ("2D-E768-CL", "flow_ensemble_large_cl"),
    ("2D-E768-NCL", "flow_ensemble_ncl_v2"),
    ("2D-S768-CL", "flow_ensemble_single_small_cl"),
    ("2D-S768-NCL", "flow_ensemble_single_not_spectral_v2"),
    (
        "3D-S768-CL",
        "flow_3d_prior_cl",
    ),  # corrected retrain -- falls back to _v2_1 (mislabeled NCL prior) with a warning if not yet ready
    ("diffesm", "diffesm_new_monthcond"),
    ("MESMER-M", "MESMER-M"),
    ("IPSL", "IPSL"),
]

R = {}
missing = []
for n, f in COLS:
    path = f"{D}/{f}.npz"
    if n == "3D-S768-CL" and not os.path.exists(path):
        path = f"{D}/flow_3d_prior_v2_1.npz"
        if os.path.exists(path):
            print(
                "WARNING: using flow_3d_prior_v2_1.npz (known-mislabeled NCL prior) as a placeholder for 3D-S768-CL -- replace once flow_3d_prior_cl is bootstrapped"
            )
    if os.path.exists(path):
        R[n] = load_paper_bootstrap(path)
    else:
        missing.append(n)
if missing:
    print(f"NOTE: skipping columns with no bootstrap data yet: {missing}\n")
COLS = [(n, f) for n, f in COLS if n in R]

rows = list(R["IPSL"]["rows"])
metrics = list(
    R["IPSL"]["metrics"]
)  # ['mse','frmse','ssr'] (or 'esr' label, same underlying spskr computation)


def fmt(v):
    return f"{v:.3g}" if abs(v) < 1e-3 or abs(v) >= 100 else f"{v:.3f}"


def table(metric_name, label):
    mi = metrics.index(metric_name)
    print(
        f"## {label}, mean over hindcast starts available per model, [point +/- 1.96 SD] bootstrap"
    )
    print("| row | " + " | ".join(n for n, _ in COLS) + " |")
    print("|" + "---|" * (len(COLS) + 1))
    for i, r in enumerate(rows):
        cells = []
        for n, _ in COLS:
            if i >= R[n]["per_start"].shape[2]:
                cells.append("--")
                continue
            pt = R[n]["per_start"][:, mi, i].mean()
            sd = R[n]["reps"][:, mi, i].std()
            lo, hi = pt - 1.96 * sd, pt + 1.96 * sd
            cells.append(f"{fmt(pt)} [{fmt(lo)}, {fmt(hi)}]")
        print(f"| {r} | " + " | ".join(cells) + " |")
    print()


def pairs_table(metric_name, label):
    mi = metrics.index(metric_name)
    print(
        f"## Paired differences, {label} (A - B, same resamples); * = 95% CI excludes 0 (negative = A better)"
    )
    pairs = [
        ("2D-S768-CL", "2D-S768-NCL", "composite loss @ single-det, embed 768"),
        ("2D-E192-CL", "2D-E768-CL", "embed size @ ensemble-det, CL"),
        ("2D-E768-CL", "2D-E768-NCL", "composite loss @ ensemble-det, embed 768"),
        ("2D-S768-CL", "2D-E768-CL", "single vs ensemble deterministic (CL)"),
        ("3D-S768-CL", "2D-S768-CL", "axial vs 2D (single-det, CL)"),
    ]
    for a, b, desc in pairs:
        if a not in R or b not in R:
            continue
        out = []
        for i, r in enumerate(rows):
            d = R[a]["reps"][:, mi, i] - R[b]["reps"][:, mi, i]
            m = R[a]["per_start"][:, mi, i].mean() - R[b]["per_start"][:, mi, i].mean()
            lo, hi = m - 1.96 * d.std(), m + 1.96 * d.std()
            rel = 100 * m / R[b]["per_start"][:, mi, i].mean()
            out.append(f"{r} {rel:+.0f}%{'*' if lo > 0 or hi < 0 else ''}")
        print(f"\n{desc}  [{a} - {b}]\n  " + "; ".join(out))
    print()


table("frmse", "RMSE (fair RMSE, unbiased for ensemble size)")
table(
    "esr" if "esr" in metrics else "ssr",
    "SSR (spread-skill ratio; unit reference for exchangeable ensembles; accounts for both ensemble sizes)",
)
pairs_table("frmse", "RMSE")
pairs_table("esr" if "esr" in metrics else "ssr", "SSR")
