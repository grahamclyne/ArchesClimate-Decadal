import os
import random
from typing import Any

import numpy as np
import torch
import xarray as xr
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset


class ClimateDataset(Dataset):
    def __init__(
        self,
        seq_len: int,
        realizations: list[str],
        esm: str,
        data_dir: str,
        scenario: str,
        vars: list[str],
    ):
        self.seq_len = seq_len
        self.realizations = realizations

        self.data_dir = os.path.join(data_dir, esm, scenario)

        # Necessary to convert vars into a Python list
        self.vars = OmegaConf.to_object(vars) if not isinstance(vars, list) else vars

        self.xr_data: xr.Dataset
        self.tensor_data: torch.Tensor

        # Normalization stats (mean, std) per variable, computed once from the
        # first loaded realization and reused for all subsequent ones.
        self.norm_stats: dict[str, tuple[float, float]] = {}

        self.load_data(self.realizations[0])

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _normalize(self, da: xr.DataArray) -> xr.DataArray:
        mean, std = self.norm_stats[da.name]
        return (da - mean) / std

    def _denorm(self, da: xr.DataArray) -> xr.DataArray:
        mean, std = self.norm_stats[da.name]
        return da * std + mean

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def estimate_num_batches(self, batch_size: int) -> int:
        return len(self) * len(self.realizations) // batch_size

    def load_data(self, realization: str):
        realization_dir = os.path.join(self.data_dir, realization, "*.nc")
        dataset = xr.open_mfdataset(realization_dir, combine="nested", concat_dim="time").sortby("time")
        # DCPP hindcasts overlap in calendar time; deduplicate, keeping first occurrence
        _, unique_idx = np.unique(dataset.time.values, return_index=True)
        dataset = dataset.isel(time=np.sort(unique_idx))[self.vars]

        # Mask fill values and non-finite entries (e.g. 1e20 sentinels in NetCDF)
        # before computing statistics so they don't corrupt mean/std.
        dataset = xr.Dataset({v: dataset[v].where(np.isfinite(dataset[v])) for v in self.vars})

        # Compute z-score stats from the first realization and reuse for all others
        if not self.norm_stats:
            loaded = dataset.load()
            self.norm_stats = {
                var: (float(loaded[var].mean()), float(loaded[var].std()))
                for var in self.vars
            }

        # NaN remaining after masking becomes 0 (the normalized mean) in tensor space
        self.xr_data = dataset.map(self._normalize).fillna(0.0)
        self.tensor_data = self.convert_xarray_to_tensor(self.xr_data)
        self.start_indices = list(range(len(self.xr_data.time) - self.seq_len + 1))
        # 0-indexed (0-11) calendar month per timestep, for month-of-year
        # conditioning -- see UNetModel3D's month_cond.
        self.months = torch.tensor(
            self.xr_data.time.dt.month.values - 1, dtype=torch.long
        )

    # ------------------------------------------------------------------
    # Tensor / xarray conversion
    # ------------------------------------------------------------------

    def convert_xarray_to_tensor(self, ds: xr.Dataset) -> torch.Tensor:
        stacked_ds = ds.to_stacked_array(
            new_dim="var", sample_dims=["time", "lon", "lat"]
        ).transpose("var", "time", "lat", "lon")
        return torch.tensor(stacked_ds.to_numpy(), dtype=torch.float32)

    def convert_tensor_to_xarray(
        self, tensor: torch.Tensor, coords: xr.DataArray = None
    ) -> xr.Dataset:
        assert len(tensor.shape) == 4, "Tensor must have shape (var, time, lat, lon)"

        np_data = tensor.cpu().numpy()
        data_vars = {
            var_name: (["time", "lat", "lon"], np_data[i])
            for i, var_name in enumerate(self.xr_data.data_vars.keys())
        }
        ds = xr.Dataset(
            data_vars,
            coords={
                "time": np.arange(np_data.shape[1]),
                "lat": np.linspace(-90, 90, np_data.shape[2]),
                "lon": np.linspace(0, 360, np_data.shape[3]),
            },
        ).map(self._denorm)

        if coords is not None:
            ds = ds.assign_coords(coords)
        return ds

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx: int):
        start = self.start_indices[idx]
        return {
            "data": self.tensor_data[:, start : start + self.seq_len],
            "months": self.months[start : start + self.seq_len],
        }


class ClimateDataLoader:
    def __init__(
        self,
        dataset: ClimateDataset,
        accelerator: Accelerator,
        batch_size: int,
        **dataloader_kwargs: dict[str, Any],
    ):
        self.dataset = dataset
        self.accelerator = accelerator
        self.batch_size = batch_size
        self.dataloader_kwargs = dataloader_kwargs

    def __len__(self):
        return self.dataset.estimate_num_batches(self.batch_size)

    def generate(self) -> torch.Tensor:
        random.shuffle(self.dataset.realizations)

        for realization in self.dataset.realizations:
            self.dataset.load_data(realization)

            dl = self.accelerator.prepare(
                DataLoader(
                    self.dataset, batch_size=self.batch_size, **self.dataloader_kwargs
                )
            )

            yield from dl
