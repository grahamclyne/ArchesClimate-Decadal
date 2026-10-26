import glob
import os

import numpy as np
import torch
import xarray as xr
from tensordict.tensordict import TensorDict

from ArchesClimate.paths import path as artifact_path

scratch = str(artifact_path("data"))
files = glob.glob(f"{scratch}/dcpp_dataset/*")

variables = {
    "surface": [
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
    ],
    "level": ["hur", "hus", "ta", "ua", "va", "wap", "zg"],
}

out_dir = "dcpp_dataset_memmap"
for file in files:
    print("processing file:", file)
    if os.path.exists(
        f"{scratch}/{out_dir}/{file.split('/')[-1].replace('.nc', '.memmap')}"
    ):
        print(
            f"{file.split('/')[-1].replace('.nc', '.memmap')} memmap already exists, skipping"
        )
        continue

    ds = xr.open_dataset(file, engine="netcdf4")
    # ds = ds.sel(plev=[100000.,  92500.,  85000.,  70000.,  60000.,  50000.,  40000.,  30000., 25000.,  20000.,  15000.,  10000.,   7000.,   5000.,   3000.,   2000.,1000.])
    # ds = ds.sel(lev=[0.50576,3.8562799,8.092519,13.991038,22.757616,35.740204,53.850636,77.61116,108.03028, 147.40625,])
    np_arrays = {}
    for key, variable in variables.items():
        if variable:  # non-empty
            np_arrays[key] = ds[list(variable)].to_array().to_numpy()
        else:  # empty list -> create an empty array
            np_arrays[key] = np.empty((0,))

    tdict = TensorDict(
        {key: torch.from_numpy(np_array).float() for key, np_array in np_arrays.items()}
    )
    tdict["time"] = ds.time.values
    tdict.memmap(f"{scratch}/{out_dir}/{file.split('/')[-1].replace('.nc', '.memmap')}")
