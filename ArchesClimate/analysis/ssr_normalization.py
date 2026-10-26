"""Read current and legacy five-versus-five paper bootstrap results."""

import numpy as np

NORMALIZATION = "ensemble_mean_v1"


def load_paper_bootstrap(path):
    """Convert legacy SSR arrays in memory; never rescale a marked archive twice.

    The paper bootstrap producers all use five predicted and five reference
    members. This loader is deliberately specific to those archives.
    """
    with np.load(path) as archive:
        data = {key: archive[key] for key in archive.files}
    version = str(data.get("ssr_normalization", "legacy_single_realization"))
    if version == NORMALIZATION:
        return data
    if version != "legacy_single_realization":
        raise ValueError(f"Unknown SSR normalization: {version}")
    for index, name in enumerate(data["metrics"]):
        if name in ("ssr", "esr", "spskr"):
            for key in ("per_start", "reps"):
                data[key][:, index, :] /= np.sqrt(3)
    data["ssr_normalization"] = np.array(NORMALIZATION)
    data["ssr_pred_members"] = np.array(5)
    data["ssr_reference_members"] = np.array(5)
    return data
