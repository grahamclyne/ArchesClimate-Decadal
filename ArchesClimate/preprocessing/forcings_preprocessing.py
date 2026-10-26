import glob
import os
import subprocess

import torch
import xarray as xr

from ArchesClimate.paths import path as artifact_path

# DO ALL CDO REGRIDDING FOR DATA HERE


# get means for aerosol data
def prep_aerosol_forcings(directory):
    files = sorted(glob.glob(directory))
    data = None
    for file in files:
        ds = xr.open_dataset(file)
        if data == None:
            data = torch.tensor(ds.to_array().values)
        else:
            data = torch.concatenate([data, torch.tensor(ds.to_array().values)], dim=1)
        ds.close()
    # aerosol_normalized = (data - data.mean(1,keepdims=True)) / data.std(axis=(1,2,3),keepdims=True)
    # torch.save(aerosol_normalized,output_path)
    return data


def process_mole_file(file):
    if "ssp" in file:
        start_index = 0
        end_index = 85 * 12 + (12)  # 2100 is 85 years after 2015
    else:
        start_index = 1850 * 12 + (1 - 1)
        end_index = 2014 * 12 + (12)
    ds = xr.open_dataset(file, decode_times=False)
    ds = ds.isel(time=slice(start_index, end_index))
    ds = ds.drop_vars(["time_bnds"])
    ds = ds.to_array().values[0]
    ds = torch.swapaxes(torch.tensor(ds), -2, -1)
    return ds


def convert_spatial_ghg_to_tensor():
    scratch = str(artifact_path("data"))
    mole_files = sorted(
        glob.glob(
            f"{scratch}/reference_data/raw_spatial_monthly_ghg_forcings/*_preprocessed.nc"
        )
    )
    for file in mole_files:
        gas = file.split("-")[3]
        ssp = next((s for s in file.split("/")[-1].split("-") if "ssp" in s), None)
        if ssp is None:
            ssp = "historical"
        ds = process_mole_file(file)
        # ds_normalized = ((ds - gas_stats[gas + '_mean']) / gas_stats[gas + '_std'])
        file_name = gas + "_" + ssp + ".pt"
        torch.save(
            ds,
            f"{scratch}/reference_data/processed_spatial_monthly_ghg_forcings/{file_name}",
        )


def run_command(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)


def main():
    scratch = str(artifact_path("data"))
    # Interpolating spatial ghg
    print("Regridding spatial ghg...")
    # interpolated_file = 'interpolated.nc'
    # for file in glob.glob(f'{scratch}/reference_data/raw_spatial_monthly_ghg_forcings/mole-*'):
    #     if('preprocessed' in file):
    #         continue
    #     output_file_name = file.split('.')[0] + '_preprocessed.nc'
    #     if os.path.exists(output_file_name):
    #         print(f"File {output_file_name} already exists.")
    #         continue
    #     try:
    #         remap_command = f"cdo enlarge,r360x360 {file} {interpolated_file}"
    #         rename_command = f"cdo remapcon,r144x144 {interpolated_file} {output_file_name}"
    #         # Execute the commands
    #         run_command(remap_command)
    #         run_command(rename_command)
    #     finally:
    #         # Step 3: Delete intermediate file
    #         if os.path.exists(interpolated_file):
    #             os.remove(interpolated_file)
    #             print(f"Deleted intermediate file: {interpolated_file}")

    # #now take all cdo regridded and convert to tensor
    # print('Converting spatial ghg to tensor and saving...')

    # convert_spatial_ghg_to_tensor()

    print("regridding aerosol...")

    # INTERPOLATE AEROSOL
    # experiments = ['ssp119','ssp126','ssp245','ssp370','ssp434','ssp460','ssp534-over','ssp585','historical']
    # for exp in experiments:
    #     print(exp)
    #     keep_vars = ['load_ASNO3M','load_CSNO3M','load_CINO3M','load_SO4','load_AIBCM','load_ASBCM']
    #     scratch = os.environ['SCRATCH']
    #     for file in glob.glob(f'{scratch}/aerosol_forcings_{exp}/*'):
    #         interpolated_file = f"interpolated_{exp}.nc"

    #         file_name = file.split('/')[-1].split('.')[0]
    #         if not os.path.exists(f'{scratch}/aerosol_forcings_{exp}_preprocessed_2d/'):
    #             os.mkdir(f'{scratch}/aerosol_forcings_{exp}_preprocessed_2d/')
    #         output_file_name = f'{scratch}/aerosol_forcings_{exp}_preprocessed_2d/' + file_name + '_preprocessed.nc'
    #         if os.path.exists(output_file_name):
    #             print(f"File {output_file_name} already exists.")
    #             continue
    #         try:
    #             remap_command = f"cdo remapcon,r144x144 {file} {interpolated_file}"
    #             # Step 2: Rename 'time_counter' to 'time'
    #             rename_command = f"ncrename -d .time_counter,time {interpolated_file}"
    #             # Execute the commands
    #             run_command(remap_command)
    #             run_command(rename_command)
    #             var_list = ",".join(keep_vars)
    #             run_command(f"ncks -C -v {var_list} {interpolated_file} {output_file_name}")

    #             # run_command(f"ncks -C -O -x -v time_counter $(ncks -m {interpolated_file} | "
    #             #         f"grep '^  load_' | awk '{{print $2}}' | tr '\\n' ',' | sed 's/,$//') {interpolated_file} {renamed_file}")
    #         finally:
    #             # Step 3: Delete intermediate file
    #             if os.path.exists(interpolated_file):
    #                 os.remove(interpolated_file)
    #                 print(f"Deleted intermediate file: {interpolated_file}")

    #     data = prep_aerosol_forcings(f'{scratch}/aerosol_forcings_{exp}_preprocessed_2d/*')
    #     torch.save(data,f'{scratch}/reference_data/aerosol_{exp}.pt')

    # INTERPOLATE OZONE
    # cdo remapcon,r144x144 ozone_forcings_ssp370/tro3_2049.nc tro3_2049_regridded.nc
    experiments = [
        "ssp119",
        "ssp126",
        "ssp245",
        "ssp370",
        "ssp434",
        "ssp460",
        "ssp534-over",
        "ssp585",
        "historical",
    ]
    for exp in experiments:
        print("regridding ozone....")
        ozone_files = glob.glob(
            f"""{artifact_path("data")}/ozone_forcings_{exp}/climoz*"""
        )
        for file in ozone_files:
            out_file = f"""{artifact_path("data")}/ozone_forcings_{exp}_regridded/{file.split("/")[-1]}"""
            if os.path.exists(out_file):
                print(f"{out_file} exists")
                continue

            regrid_command = f"cdo remapcon,r144x144 {file} {out_file}"
            run_command(regrid_command)
        print("saving ozone....")
        ozone_files = glob.glob(
            f"""{artifact_path("data")}/ozone_forcings_{exp}_regridded/climoz_*"""
        )
        ozone_files.sort()
        data = None
        for ozone_file in ozone_files:
            ds = xr.open_dataset(ozone_file, decode_times=False)
            # da_reduced = (
            #     ds
            #     .coarsen({'plev': 11}, boundary="trim")  # groups of 11
            #     .mean()
            # )
            # ds = da_reduced['tro3'].isel(time=slice(1,-1))
            ds = ds["tro3"].isel(time=slice(1, -1))
            if "rlatu" in ds.coords:
                ds = ds.rename({"rlatu": "latitude", "rlonv": "longitude"})
            else:
                ds = ds.rename({"lat": "latitude", "lon": "longitude"})
            ds = ds.transpose("time", "plev", "longitude", "latitude")
            # if('lat' in ds.coords):
            #     ds = ds.rename({'lat':'latitude','lon':'longitude'})
            #     # Suppose your dataset has dims: ("time", "lat", "lon")
            #     ds = ds.transpose('time','longitude',"latitude")  # reorder dimensions
            if data == None:
                data = torch.tensor(ds.values)
            else:
                data = torch.concatenate([data, torch.tensor(ds.values)], dim=0)
            ds.close()
        torch.save(data, f"{scratch}/reference_data/ozone_{exp}.pt")

    print("regridding and saving orography")
    # Interpolate orography and convert to tensor
    # orography_remap_command = f"cdo remapcon,r144x144 {scratch}/reference_data/orography_forcings/orog_fx_IPSL-CM6A-LR_historical_r11i1p1f1_gr.nc {scratch}/reference_data/orography_forcings/orog_144x144.nc"
    # run_command(orography_remap_command)
    # oro = xr.open_dataset(f'/{scratch}/reference_data/orography_forcings/orog_144x144.nc')
    # torch.save(torch.tensor(oro['orog'].values),f'{scratch}/reference_data/orography.pt')

    print("interpolating solar")
    # interpolate missing values in solar
    # solar_forcings = np.load(f'{scratch}/reference_data/solar_forcings_monthly.npy') #need to generate this first
    # #just some good old last minute interpolation patching nans
    # from scipy.interpolate import interp1d
    # # Sample 1D data with missing values
    # for index in range(solar_forcings.shape[1]):
    #     data = solar_forcings[:,index]
    #     # Get the indices
    #     x = np.arange(len(solar_forcings))
    #     # Mask valid values
    #     valid = ~np.isnan(data)
    #     # Create the interpolation function
    #     f_interp = interp1d(x[valid], data[valid], kind='quadratic', fill_value="extrapolate")
    #     # Fill the missing values
    #     data_filled = data.copy()
    #     data_filled[~valid] = f_interp(x[~valid])
    #     solar_forcings[:,index] = data_filled
    # torch.save(torch.tensor(solar_forcings),f'{scratch}/reference_data/solar_forcings_monthly_interpolated.pt')


main()
