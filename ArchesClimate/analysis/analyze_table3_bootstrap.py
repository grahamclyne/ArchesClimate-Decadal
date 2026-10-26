from ArchesClimate.analysis.ssr_normalization import load_paper_bootstrap
from ArchesClimate.paths import reference_results

D = str(reference_results("bootstrap_table3_results_11"))
COLS = [
    ("2D-768-CL-NC", "table3_2d768_cl_nc"),
    ("3D-192-CL-C", "table3_3d192_cl_c"),
    ("2D-192-CL-C", "table3_2d192_cl_c"),
    ("2D-768-NCL-C", "table3_2d768_ncl_c"),
    ("2D-192-NCL-C", "table3_2d192_ncl_c"),
    ("2D-768-CL-C", "table3_2d768_cl_c"),
    ("old 2D-768-NCL-C ep19", "ctrl_old_small2_ep19"),
    ("IPSL", "IPSL"),
]
R = {n: load_paper_bootstrap(f"{D}/{f}.npz") for n, f in COLS}
rows = list(R["IPSL"]["rows"])
metrics = list(R["IPSL"]["metrics"])  # ['mse','frmse','ssr']


def fmt(v):
    return f"{v:.3g}" if abs(v) < 1e-3 or abs(v) >= 100 else f"{v:.3f}"


def table(metric_name, label):
    mi = metrics.index(metric_name)
    print(
        f"## {label}, mean over 11 hindcast starts, [point +/- 1.96 SD] from 150 member+start bootstrap draws (percentile CI is biased upward by member resampling)"
    )
    print("| row | " + " | ".join(n for n, _ in COLS) + " |")
    print("|" + "---|" * (len(COLS) + 1))
    for i, r in enumerate(rows):
        cells = []
        for n, _ in COLS:
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
        ("2D-768-CL-C", "2D-768-NCL-C", "composite loss @768"),
        ("2D-192-CL-C", "2D-192-NCL-C", "composite loss @192"),
        ("2D-768-NCL-C", "2D-192-NCL-C", "768 vs 192 (no CL)"),
        ("2D-768-CL-C", "2D-192-CL-C", "768 vs 192 (CL)"),
        ("3D-192-CL-C", "2D-192-CL-C", "axial vs 2D @192"),
        ("2D-768-CL-NC", "2D-768-CL-C", "no conditioning vs conditioning"),
        (
            "2D-768-NCL-C",
            "old 2D-768-NCL-C ep19",
            "retrained (fixed) vs old checkpoint",
        ),
    ]
    for a, b, desc in pairs:
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
    "ssr",
    "SSR (spread-skill ratio; unit reference for exchangeable ensembles; accounts for both ensemble sizes)",
)
pairs_table("frmse", "RMSE")
pairs_table("ssr", "SSR")
