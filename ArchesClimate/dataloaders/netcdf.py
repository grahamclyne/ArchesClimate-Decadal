from collections.abc import Callable
from pathlib import Path

import torch
from tensordict.tensordict import TensorDict
from tqdm import tqdm

# Appropriate xarray engine for a given file extension
engine_mapping = {
    ".nc": "netcdf4",
    ".nc4": "netcdf4",
    ".h5": "h5netcdf",
    ".hdf5": "h5netcdf",
    ".grib": "cfgrib",
    ".zarr": "zarr",
}


default_dimension_indexers = {
    "latitude": ("latitude", slice(None)),
    "longitude": ("longitude", slice(None)),
    "level": ("level", slice(None)),
    "time": ("time", slice(None)),
}


class XarrayDataset(torch.utils.data.Dataset):
    """
    Read TensorDict memmap files and iterate over their time coordinates.

    The class name is retained for compatibility with existing dataset classes.
    Samples are read directly from memmaps by __getitem__.
    """

    def __init__(
        self,
        path: str,
        variables: dict[str, list[str]],
        dimension_indexers: dict[str, list] | None = None,
        filename_filter: Callable = lambda _: True,  # condition to keep file in dataset
        return_timestamp: bool = False,
        warning_on_nan: bool = False,
        limit_examples: int | None = None,
        timestamp_key: Callable = lambda x: x[-1],
        interpolate_nans: bool = False,
        plev_indices: list[int]
        | None = None,  # explicit pressure-level indices into the memmap
        max_cache_files: int = 1,  # LRU bound; keep <=N mmap'd files alive per worker
    ):
        """
        Args:
        path: Single filepath or directory holding xarray files.
        variables: Dict holding xarray data variable lists mapped by their keys to be processed into tensordict.
            e.g. {surface: [data_var1, datavar2, ...], level: [...]}
            The memmap entries are selected by the dataset subclass.
        dimension_indexers: Dict of dimensions to select in xarray using Dataset.sel(dimension_indexers). Also provides
            the dimension names to treat the xarray dataset as tensordict.
            defaults to:
                dimension_indexers = {
                    'latitude': ('latitude', None),
                    'longitude': ('longitude', None),
                    'level': ('level', None),
                    'time': ('time', None)
                }
            If not provided, defaults to selecting all data in all dimensions.
            First value is the dimension name in xarray, second value is the indexer
            If None is used as the indexer, all coordinates in that dimension are used.
            To select a range, use a tuple (start, end, step) as the indexer.
        filename_filter: To filter files within `path` based on filename.
        return_timestamp: Whether to return timestamp in __getitem__() along with tensordict.
        warning_on_nan: Whether to log warning if nan data found.
        limit_examples: Return set number of examples in dataset
        interpolate_nans: Whether to fill NaN values in the data with the mean of the
                            data across latitude and longitude dimensions. Defaults to True.
        """
        self.filename_filter = filename_filter
        self.variables = variables

        self.dimension_indexers = dimension_indexers or default_dimension_indexers
        self.return_timestamp = return_timestamp
        self.warning_on_nan = False
        self.interpolate_nans = interpolate_nans
        self.max_cache_files = max_cache_files

        # Workaround to avoid calling ds.sel() after ds.transponse() to avoid OOM.
        self.already_ran_index_selection = False

        if not Path(path).exists():
            raise ValueError("Path does not exist:", path)

        if Path(path).is_file() and "." in path.split("/")[-1]:
            print("Single file detected. Loading single file ", path)
            self.files = [path]
        else:
            files = list(Path(path).glob("*"))
            if len(files) == 0:
                raise ValueError("No files found under path:", path)

            self.files = sorted(
                [str(x) for x in files if filename_filter(x.name)],
                key=lambda x: x.replace("_6h", "_06h").replace("_0h", "_00h"),
            )
            if len(self.files) == 0:
                raise ValueError("filename_filter filtered all files under path:", path)

        # engine = engine_mapping[file_extension]
        # self.xr_options = dict(engine=engine, cache=True)

        self.timestamps = []
        # for fid, f in tqdm(enumerate(self.files)):
        #     # Open Zarr store
        #     zr = zarr.open(f, mode='r')

        #     # Get time values from the Zarr array
        #     time_values = zr['time'][:]
        #     file_stamps = [(fid, i, t) for (i, t) in enumerate(time_values)]
        #     self.timestamps.extend(file_stamps)

        #     if (
        #         limit_examples and len(self.timestamps) > limit_examples
        #     ):  # get fraction of full dataset
        #         print("Limiting number of examples loaded to", limit_examples)
        #         self.timestamps = self.timestamps[:limit_examples]
        #         break

        # for fid, f in tqdm(enumerate(self.files)):
        #     with xr.open_dataset(f, **self.xr_options) as obs:
        #         file_stamps = [(fid, i, t) for (i, t) in enumerate(obs.time.to_numpy())]
        #         self.timestamps.extend(file_stamps)
        #     if (
        #         limit_examples and len(self.timestamps) > limit_examples
        #     ):  # get fraction of full dataset
        #         print("Limiting number of examples loaded to", limit_examples)
        #         self.timestamps = self.timestamps[:limit_examples]
        #         break
        self.plev_indices = plev_indices  # may be overridden below by auto-detection

        for fid, f in tqdm(enumerate(self.files)):
            obs = TensorDict.load_memmap(f)
            file_stamps = [(fid, i, t) for (i, t) in enumerate(obs["time"])]
            self.timestamps.extend(file_stamps)

            # Auto-detect plev indices from the first file if the memmap stores a 'plev' key
            if fid == 0 and self.plev_indices is None and self.dimension_indexers:
                plev_key = next((k for k in ("plev", "lev") if k in obs), None)
                if plev_key is not None and "plev" in self.dimension_indexers:
                    _, wanted = self.dimension_indexers["plev"]
                    file_plevs = obs[plev_key].tolist()
                    self.plev_indices = [file_plevs.index(p) for p in wanted]
                    print(
                        f"Auto-detected plev_indices {self.plev_indices} from '{plev_key}' key"
                    )

            if (
                limit_examples and len(self.timestamps) > limit_examples
            ):  # get fraction of full dataset
                print("Limiting number of examples loaded to", limit_examples)
                self.timestamps = self.timestamps[:limit_examples]
                break

        self.timestamps = sorted(
            self.timestamps, key=timestamp_key
        )  # sort by timestamp
        self.id2pt = dict(enumerate(self.timestamps))

        self.cached_xrdataset = None
        self.cached_fileid = None
        # Open with consolidated metadata (much faster)
        # zr = zarr.open_consolidated(self.files[0], mode='r') if zarr.consolidate_metadata else zarr.open(file_path, mode='r')

        # # PRE-COMPUTE indices outside the loop (CRITICAL!)
        # lev_indices_temp = [round(float(x), 3) for x in zr['lev'][:]]
        # self.lev_indices_cache = {}
        # if 'lev' in self.dimension_indexers:
        #     self.lev_indices_cache['lev'] = [lev_indices_temp.index(round(float(val), 3))
        #                                   for val in self.dimension_indexers['lev'][1]]

        # self.plev_indices_cache = {}
        # if 'plev' in self.dimension_indexers:
        #     plev_list = list(zr['plev'][:])
        #     self.plev_indices_cache['plev'] = [plev_list.index(val)
        #                                    for val in self.dimension_indexers['plev'][1]]

    # def set_timestamp_bounds(self, low, high, debug=False):
    #     """Filter timestamps loaded from dataloader between bounds.

    #     If low or high is None, only filter in one direction.

    #     Args:
    #         low: lower bound, inclusive. Set to None to not filter by lower bound.
    #         high: upper bound, exclusive. Set to None to not filter by upper bound.
    #     """
    #     original_length = len(self.timestamps)

    #     if low and high:
    #         self.timestamps = [
    #             x for x in self.timestamps if low <= x[-1].astype("datetime64[s]") < high
    #         ]
    #     elif low:
    #         self.timestamps = [x for x in self.timestamps if low <= x[-1].astype("datetime64[s]")]
    #     elif high:
    #         self.timestamps = [x for x in self.timestamps if x[-1].astype("datetime64[s]") < high]
    #     if debug:
    #         print(
    #             f"Filtered timestamps from {original_length} to {len(self.timestamps)} examples: "
    #             f"{self.timestamps[0][-1].astype('datetime64[s]')} to {self.timestamps[-1][-1].astype('datetime64[s]')}."
    #         )
    #     self.id2pt = dict(enumerate(self.timestamps))

    def __len__(self):
        return len(self.id2pt)

    # def convert_to_tensordict(self, xr_dataset,file_path):

    def _ensure_cache(self):
        if getattr(self, "_file_cache", None) is None:
            from collections import OrderedDict

            self._file_cache = OrderedDict()

    def __getitem__(
        self, i, return_timestamp=False, interpolate_nans=None, warning_on_nan=None
    ):
        # import gc

        file_id, line_id, _timestamp = self.id2pt[i]
        # file_id, line_id, timestamp = 0,0,0
        file_path = self.files[file_id]
        # data = TensorDict.load_memmap(file_path)
        # # print(data)

        # out = TensorDict({'lev':data['lev'][:,line_id],
        #           'level':data['level'][:,line_id],
        #           'surface':data['surface'][:,line_id,None],
        #           'non_spatial_forcings':data['non_spatial_forcings'][:,line_id],
        #           'spatial_forcings':torch.cat([data['spatial_forcings'][:,line_id],data['ozone'][0,line_id]]) if 'ozone' in data.keys() else data['spatial_forcings'][:,line_id]})
        # # print(out)
        # return out

        self._ensure_cache()
        if file_path in self._file_cache:
            self._file_cache.move_to_end(file_path)
        else:
            if len(self._file_cache) >= self.max_cache_files:
                self._file_cache.popitem(last=False)  # evict LRU entry
            self._file_cache[file_path] = TensorDict.load_memmap(file_path)

        data = self._file_cache[file_path]
        level = data["level"][:, line_id].clone()
        # if self.plev_indices is not None:
        #     level = level[:, self.plev_indices].clone()
        surface = data["surface"][:, line_id, None].clone()
        out = TensorDict(
            {
                "level": level,
                "surface": surface,
            }
        )
        # print(out)
        return out
