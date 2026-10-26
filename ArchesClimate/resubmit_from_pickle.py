"""Resubmit a submitit job from its original `<jobid>_submitted.pkl`.

Both forecast_added_input_state_alt_years (1751980) and
forecast_added_input_state_gap (1752159) failed at dataloader instantiation
(FileNotFoundError on reference_data/cmip_ghg_forcings_ssp245.pt, now fixed by
preprocessing/build_ssp245_ghg_forcings.py) before any training/checkpointing
happened. Rather than reverse-engineering the original `python submit.py
module=... name=...` CLI overrides, this reuses the exact pickled DictConfig
that was already submitted once, so the resubmit is byte-identical to the
original job apart from the now-present forcings file.

Run under `module load pytorch-gpu/py3/2.1.1` with PYTHONPATH pointing at the
archesclimate_aeis_submission parent dir (the same recipe
analysis/run_long_rollout.slurm uses successfully) -- NOT pytorch-gpu/py3/2.3.0,
which pulls in a torch/torchvision pair that conflicts with what's on
~/.local/lib/python3.11/site-packages.
"""

import sys

import submitit

from ArchesClimate.main_hydra import main as hydra_main
from ArchesClimate.paths import path as artifact_path

pkl_path = sys.argv[1]

delayed = submitit.core.utils.DelayedSubmission.load(pkl_path)

cfg = delayed.args[0]
print(f"resubmitting name={cfg.name}")

aex = submitit.AutoExecutor(
    folder=str(artifact_path("logs") / cfg.name),
    cluster="slurm",
)
aex.update_parameters(**cfg.cluster.launcher)
job = aex.submit(hydra_main, cfg)
print(f"submitted job {job.job_id}")
