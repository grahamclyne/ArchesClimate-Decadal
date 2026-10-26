from hydra import compose, initialize_config_dir

from ArchesClimate.analysis.gen_and_save_metrics import run_metrics
from ArchesClimate.paths import config_directory

CD = str(config_directory())
# (experiment name, module config, extra overrides)
MODELS = [
    (f"table3_{k}", f"table3_{k}", [])
    for k in ["2d768_cl_c", "2d192_ncl_c", "2d192_cl_c", "3d192_cl_c", "2d768_cl_nc"]
]
MODELS += [
    ("forecast_added_input_state_0", "forecast_added_input_state", []),
    (
        "ctrl_old_small2",
        "forecast_added_input_state_small",
        ["module.module.proper_year_forcings=False"],
    ),
]
for n, mod, extra in MODELS:
    for split in ["test", "val"]:
        with initialize_config_dir(version_base=None, config_dir=CD):
            cfg = compose(
                config_name="config",
                overrides=[
                    f"module={mod}",
                    f"name={n}",
                    "inference=inference_5",
                    "inference.num_inference_steps=10",
                    f"inference.target={split}",
                    "inference.year=0",
                ]
                + extra,
            )
        run_metrics(
            cfg,
            f"{n}_10_5_1_118_1_{split}_0_None",
            num_members=5,
            num_rollout_steps=118,
        )
