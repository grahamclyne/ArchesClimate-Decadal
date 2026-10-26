import json
import os

import hydra
import numpy as np
import xarray as xr
from omegaconf import DictConfig
from typing_extensions import TypedDict

# Constants
COW_PATH = "/research/hutchinson/"
CLUSTER_PATH = "/cluster/research-groups/hutchinson"


class DATA_DICT(TypedDict):
    load_dir: str
    save_dir: str
    scenario: str
    realizations: dict[str, dict[str, list[str]]]


# How many chunks to split the data into - this will create NUM_CHUNKS separate files to hold the dat
NUM_CHUNKS = 120


def save_dataset(dataset: xr.Dataset, realization: str, save_dir: str, num_chunks):
    """Saves the dataset in chunks to many netCDF4 files for parallel loading later."""

    full_save_dir = os.path.join(save_dir, realization)
    os.makedirs(full_save_dir, exist_ok=True)

    for file in os.listdir(full_save_dir):
        os.remove(os.path.join(full_save_dir, file))

    total_time_points = len(dataset["time"])
    chunk_size = total_time_points // num_chunks

    for idx in range(num_chunks):
        start_idx = idx * chunk_size
        end_idx = start_idx + chunk_size if idx < num_chunks - 1 else None
        chunk = dataset.isel(time=slice(start_idx, end_idx))
        chunk.to_netcdf(os.path.join(full_save_dir, f"chunk_{idx}.nc"))


def process_dataset(dataset: xr.Dataset, start_year: int, end_year: int) -> xr.Dataset:
    """Does any pre-processing of the dataset necessary before we save it to chunks.

    Args:
        dataset (xr.Dataset): Our data

    Returns:
        xr.Dataset: Our processed data
    """

    dataset = dataset.drop_vars(["time_bnds", "lat_bnds", "lon_bnds"], errors="ignore")

    # Drop all the time indices that are greater than 2100
    dataset = dataset.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    return dataset


def collect_var_data(path_list: list[str], base_dir: str) -> xr.Dataset:
    """Collects all the data for a given variable and realization."""

    paths = [os.path.join(base_dir, path) for path in path_list]
    return xr.open_mfdataset(paths, combine="nested", concat_dim="time").sortby("time")


@hydra.main(version_base=None, config_path="../configs", config_name="prepare_data")
def main(cfg: DictConfig):
    """Goes through each realization in the JSON file and saves the data to chunks.

    Args:
        filename (str): Path to our JSON file
        cluster (bool, optional): Whether we are on the cluster or not. Defaults to False.
    """
    json_path = os.path.join(
        cfg.paths.json_data_dir, cfg.esm, cfg.scenario, "data.json"
    )
    # Open a json file
    with open(json_path) as f:
        data: DATA_DICT = json.load(f)

    # Construct the paths to the load and save directories
    load_dir = data["load_dir"]
    save_dir = os.path.join(
        cfg.paths.data_dir,
        cfg.esm,
        cfg.scenario,
    )

    start_year = cfg.start_year
    end_year = cfg.end_year

    # Iterate through each realization in our JSON file
    for realization, realization_data in data["realizations"].items():

        # if realization in ["r1","r2"]:
        #     print(realization)
        #     continue
        # Merge the two variables together
        datasets = [collect_var_data(path_list, load_dir) for path_list in reversed(realization_data.values())]
        datasets[1] = datasets[1].assign_coords({"time" : datasets[0].time})

        dataset = xr.merge(datasets, join="right", compat="override")

        dataset = process_dataset(dataset, start_year, end_year)
        # DCPP hindcasts overlap in calendar time; deduplicate before chunking
        _, unique_idx = np.unique(dataset.time.values, return_index=True)
        dataset = dataset.isel(time=np.sort(unique_idx))
        print(f"Finished processing realization {realization}")
        save_dataset(dataset, realization, save_dir, cfg.num_chunks)
        dataset.close()
        print(f"Finished saving realization {realization}")


if __name__ == "__main__":
    main()
