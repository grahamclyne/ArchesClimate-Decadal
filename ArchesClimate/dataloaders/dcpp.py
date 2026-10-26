import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tensordict.tensordict import TensorDict

from ArchesClimate.paths import path as artifact_path
from ArchesClimate.utils.tensordict_utils import (
    apply_nan_to_num,
    get_non_nan_mask,
    replace_inf_and_large_values,
)

from .netcdf import XarrayDataset


class DCPPForecast(XarrayDataset):
    """
    Load DCPP data for the forecast task.

    Loads previous timestep and multiple future timesteps if configured.
    Also handles normalization.
    """

    def __init__(
        self,
        path="/path/to/data/",
        forcings_path="data/",
        domain="train",
        filename_filter=None,
        lead_time_months=1,
        multistep=1,
        load_prev=1,  # how many previous states to load
        load_clim=False,
        norm_scheme="spatial_norm",
        limit_examples: int = 0,
        mask_value=0,
        variables=None,
        surface_variables=None,
        level_variables=None,
        surface_variable_indices=None,
        level_variable_indices=None,
        pressure_levels=None,
        filename_filter_type="dcpp",  # choose train/test split with filename filter: "dcpp", "dcpp_alt", or "dcpp_gap"
        indexed_year_forcings=False,  # toggle index for forcings between zero-indexed and and year-indexed
        stats_path=None,
        file_pressure_levels=None,  # all pressure levels stored in the memmap in order; defaults to CMIP6 standard 17-level set
    ):
        """
        Args:
            path: Single filepath or directory holding files.
            domain: Specify data split for the filename filters (eg. train, val, test, testz0012..)
            engine: xarray dataset backend.
            filename_filter: To filter files within data directory based on filename.
            lead_time_months: Time difference between current state and previous and future states.
            multistep: How many future states to load. By default, loads one (current time + lead_time_months).
            load_prev: Whether to load state at previous timestamp (current time - lead_time_months).
            load_clim: Whether to load climatology.
            limit_examples: Return set number of examples in dataset
            mask_value: what value to use as mask for nan values in dataset
            dimension_indexers: dict, dimension indexers for the dataset.
                Default is set to pressure levels, latitude, longitude, and time.

        """
        if surface_variable_indices is None:
            surface_variable_indices = []
        if level_variable_indices is None:
            level_variable_indices = []
        if pressure_levels is None:
            pressure_levels = [85000, 70000, 50000, 25000]
        self.__dict__.update(
            locals()
        )  # concise way to update self with input arguments
        self.timedelta = 1
        if self.filename_filter_type == "dcpp_alt":
            train_filter = range(1960, 2000)
            val_filter = range(2000, 2010)
            test_filter = range(2010, 2016)
            filename_filters = {
                "all": (lambda _: True),
                "train": lambda x: any(
                    substring in x for substring in [f"{x!s}_" for x in train_filter]
                ),
                "test": lambda x: any(
                    substring in x for substring in [f"{x!s}_" for x in test_filter]
                ),
                "val": lambda x: any(
                    substring in x for substring in [f"{x!s}_" for x in val_filter]
                ),
                "empty": lambda x: False,
            }
        elif self.filename_filter_type == "dcpp_gap":
            # Like "dcpp_alt", but with a genuine multi-year buffer between
            # the end of training and the start of val/test, so there is
            # zero calendar-year adjacency between train and held-out years
            # at all (reviewer_comments.txt, Major Comment 9: distinguishes
            # forced-signal/calendar-year leakage from the internal-
            # variability decorrelation argument in Section 2d). The
            # buffer (1995-2004, unused by any split) is chosen well beyond
            # the ~2-5 year internal-variability decorrelation timescale
            # cited from Swingedouw et al., to test whether skill holds up
            # with no adjacent-year overlap whatsoever. val/test windows
            # match "dcpp_alt" exactly so only the train cutoff differs,
            # keeping the comparison controlled.
            train_filter = range(1960, 1995)
            val_filter = range(2005, 2010)
            test_filter = range(2010, 2016)
            filename_filters = {
                "all": (lambda _: True),
                "train": lambda x: any(
                    substring in x for substring in [f"{x!s}_" for x in train_filter]
                ),
                "test": lambda x: any(
                    substring in x for substring in [f"{x!s}_" for x in test_filter]
                ),
                "val": lambda x: any(
                    substring in x for substring in [f"{x!s}_" for x in val_filter]
                ),
                "empty": lambda x: False,
            }
        elif self.filename_filter_type == "dcpp":
            train_filter = [
                x for i, x in enumerate(range(1960, 2010)) if (i + 1) % 10 != 0
            ]
            val_filter = range(2010, 2016)
            test_filter = [1969, 1979, 1989, 1999, 2009]
            filename_filters = {
                "all": (lambda _: True),
                "train": lambda x: any(
                    substring in x for substring in [f"{x!s}_" for x in train_filter]
                ),
                "val": lambda x: any(
                    substring in x for substring in [f"{x!s}_" for x in val_filter]
                ),
                "test": lambda x: any(
                    substring in x for substring in [f"{x!s}_" for x in test_filter]
                ),
                "empty": lambda x: False,
            }
        if filename_filter is None:
            filename_filter = filename_filters[domain]
        if variables is None:
            variables = {"surface": surface_variables, "level": level_variables}
        stats_file_path = "dcpp_stats.pt"
        dimension_indexers = {"plev": ("plev", pressure_levels)}

        # Compute plev_indices from pressure_levels + file_pressure_levels.
        # Falls back to CMIP6 standard 17 levels if file_pressure_levels not set.
        # For new memmaps that store a 'plev' key this is overridden by auto-detection in the parent.
        _file_plevs = file_pressure_levels or [
            100000,
            92500,
            85000,
            70000,
            60000,
            50000,
            40000,
            30000,
            25000,
            20000,
            15000,
            10000,
            7000,
            5000,
            3000,
            2000,
            1000,
        ]
        plev_indices = [list(_file_plevs).index(p) for p in pressure_levels]

        super().__init__(
            path,
            filename_filter=filename_filter,
            variables=variables,
            limit_examples=limit_examples,
            dimension_indexers=dimension_indexers,
            timestamp_key=lambda x: (x[0], x[1]),
            plev_indices=plev_indices,
        )

        geoarches_stats_path = (
            Path(stats_path) if stats_path is not None else artifact_path("stats")
        )
        norm_file_path = geoarches_stats_path / stats_file_path

        spatial_norm_stats = torch.load(norm_file_path)

        clim_removed_file_path = (
            geoarches_stats_path / "dcpp_clim_removed_norm_stats.pt"
        )
        clim_removed_norm_stats = torch.load(clim_removed_file_path)

        if self.norm_scheme is None:
            self.data_mean = TensorDict(
                surface=torch.tensor(0),
                level=torch.tensor(0),
            )
            self.data_std = TensorDict(
                surface=torch.tensor(1),
                level=torch.tensor(1),
            )
        # both mean and std_dev have spatial dimensions
        elif self.norm_scheme == "spatial_norm":
            self.data_mean = TensorDict(
                surface=torch.stack(
                    [
                        spatial_norm_stats["surface_mean"][i]
                        for i in surface_variable_indices
                    ]
                ),
                level=torch.stack(
                    [
                        spatial_norm_stats["level_mean"][i]
                        for i in level_variable_indices
                    ]
                ),
            )
            self.data_std = TensorDict(
                surface=torch.stack(
                    [
                        spatial_norm_stats["surface_std"][i]
                        for i in surface_variable_indices
                    ]
                ),
                level=torch.stack(
                    [spatial_norm_stats["level_std"][i] for i in level_variable_indices]
                ),
            )
        # only mean is spatial, std_dev is averaged over space
        elif self.norm_scheme == "mean_only_spatial_norm":
            self.data_mean = TensorDict(
                surface=torch.stack(
                    [
                        spatial_norm_stats["surface_mean"][i]
                        for i in surface_variable_indices
                    ]
                ),
                level=torch.stack(
                    [
                        spatial_norm_stats["level_mean"][i]
                        for i in level_variable_indices
                    ]
                ),
            )

            self.data_std = TensorDict(
                surface=torch.stack(
                    [
                        spatial_norm_stats["surface_std"][i]
                        for i in surface_variable_indices
                    ]
                ).nanmean(axis=(-1, -2), keepdim=True),
                level=torch.stack(
                    [spatial_norm_stats["level_std"][i] for i in level_variable_indices]
                ).nanmean(axis=(-1, -2), keepdim=True),
            )
        # statistics for predicting the anomaly, both mean and std_dev have spatial dimensions
        elif self.norm_scheme == "clim_removed":
            self.data_mean = TensorDict(
                surface=clim_removed_norm_stats["surface_mean"],
                level=clim_removed_norm_stats["level_mean"],
            )
            self.data_mean.batch_size = [
                12
            ]  # this is so the tensordict can be indexed by one value for both surface/level
            self.data_std = TensorDict(
                surface=clim_removed_norm_stats["surface_std"],
                level=clim_removed_norm_stats["level_std"],
            )
            self.data_std.batch_size = [12]

        self.surface_variables = surface_variables
        self.level_variables = [
            a + " " + str(p // 100) for a in level_variables for p in pressure_levels
        ]
        self.atmos_forcings = torch.load(f"{forcings_path}/cmip_ghg_forcings_ssp245.pt")
        self.solar_forcings = torch.tensor(
            np.load(f"{forcings_path}/solar_forcings_normed.npy")
        )
        times_seconds = [v[2].item() // 10**9 for k, v in self.id2pt.items()]
        self.next_timestamp_map = {
            k: v for k, v in list(itertools.pairwise(times_seconds))
        }

        # override netcdf functionality
        self.timestamps = sorted(
            self.timestamps, key=lambda x: (x[0], x[1])
        )  # sort by timestamp

    def __len__(self):
        # Take into account previous and/or future timestamps loaded for one example.
        offset = self.multistep + self.load_prev
        return super().__len__() - offset * self.lead_time_months // self.timedelta

    def __getitem__(self, i, normalize=True):
        out = {}
        # Shift index forward if need to load previous timestamp.
        i = i + self.load_prev * self.lead_time_months // self.timedelta

        out = {}
        #  load current state
        out["state"] = super().__getitem__(i)
        out["timestamp"] = torch.tensor(
            self.id2pt[i][2].item() // 10**9,
            dtype=torch.int64,
        )  # time in seconds
        times = pd.to_datetime(out["timestamp"].cpu().numpy(), unit="s").tz_localize(
            None
        )
        current_month = torch.tensor(times.month) - 1 % 12
        if self.indexed_year_forcings:
            current_year = torch.tensor(times.year)
            current_solar_year = torch.tensor(times.year) - 1961
        else:
            current_year = torch.tensor(times.year) - 1961
            current_solar_year = torch.tensor(times.year) - 1961  # -1961 to zero index
        out["forcings"] = torch.concatenate(
            [
                self.atmos_forcings[current_year, :],
                self.solar_forcings[(current_solar_year * 12) + current_month, :],
            ]
        )

        t = self.lead_time_months  # multistep
        out["next_state"] = super().__getitem__(i + t // self.timedelta)
        # Load multiple future timestamps if specified.
        if self.multistep > 1:
            future_states = []
            for k in range(1, self.multistep + 1):
                future_states.append(super().__getitem__(i + k * t // self.timedelta))
            out["future_states"] = torch.stack(future_states, dim=0)

        if self.load_prev:
            if self.load_prev > 1:
                prev_states = []
                for k in range(self.load_prev):
                    prev_states.append(
                        super().__getitem__(
                            i - (self.lead_time_months * k + 1) // self.timedelta
                        )
                    )
                out["prev_state"] = torch.stack(prev_states, dim=0)
            else:
                out["prev_state"] = super().__getitem__(
                    i - self.lead_time_months // self.timedelta
                )
                prev_timestamp = (
                    self.id2pt[i - self.lead_time_months // self.timedelta][2].item()
                    // 10**9
                )
                times = pd.to_datetime(prev_timestamp, unit="s").tz_localize(None)

        if normalize:
            out = self.normalize(out, month=current_month)

        # need to replace nans with mask_value
        mask = {k: (get_non_nan_mask(v) if "state" in k else v) for k, v in out.items()}

        out = {k: apply_nan_to_num(v) if "state" in k else v for k, v in out.items()}
        out = {k: (v * mask[k] if "state" in k else v) for k, v in out.items()}
        return out

    def normalize(self, batch, stateless=False, month=None):
        device = next(iter(batch.values())).device
        means = self.data_mean.to(device)
        stds = self.data_std.to(device)

        if self.norm_scheme == "clim_removed":
            return {
                k: ((v - means[month]) / stds[month] if "state" in k else v)
                for k, v in batch.items()
            }
        elif stateless:
            return (batch - means) / stds
        else:
            dict_out = {
                k: ((v - means) / stds if "state" in k else v) for k, v in batch.items()
            }
            dict_out = {
                k: replace_inf_and_large_values(v, 1e35, 0) if "state" in k else v
                for k, v in dict_out.items()
            }
            return dict_out

    def denormalize(self, batch, stateless=False, month=None):
        if stateless:
            device = batch.device
        else:
            device = next(iter(batch.values())).device
        means = self.data_mean.to(device)
        stds = self.data_std.to(device)
        if self.norm_scheme == "clim_removed" and stateless:
            return batch * stds[month] + means[month]
        elif self.norm_scheme == "clim_removed":
            return {
                k: (v * stds[month] + means[month] if "state" in k else v)
                for k, v in batch.items()
            }
        elif stateless:
            return (batch * stds) + means
        else:
            return {
                k: ((v * stds) + means if "state" in k else v) for k, v in batch.items()
            }
