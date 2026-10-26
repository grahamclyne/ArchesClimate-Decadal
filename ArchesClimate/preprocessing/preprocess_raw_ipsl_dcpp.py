import os
import time

import xarray as xr
from cdo import Cdo

from ArchesClimate.paths import path as artifact_path

# ds = ds.drop(['area','time_bounds','bounds_nav_lat','bounds_nav_lon'])
# print(ds)

# outfile = 'regridded_siconc_1970.nc'
# # print(ds)
cdo = Cdo(tempdir=str(artifact_path("work")))
# print(cdo.remapcon)

cdo.debug = True
amon_surface_variables = [
    "clt",
    "hfss",
    "hfls",
    "hurs",
    "huss",
    "pr",
    "ps",
    "psl",
    "rlds",
    "rsds",
    "rsus",
    "rlus",
    "sfcWind",
    "tasmax",
    "tasmin",
    "tas",
    "evspsbl",
]
# simon_surface_variables = ['siconc','sifb','sithick']
# lmon_surface_variables = ['cVeg','gpp','evspsblsoi']
emon_surface_variables = [
    "thetaot2000",
    "thetaot300",
    "thetaot700",
    "t20d",
    "rss",
    "rls",
]
pressure_levels = [85000, 70000, 50000, 25000]


omon_surface_variables = ["tos"]
level_variables = ["hur", "hus", "ta", "ua", "va", "wap", "zg"]


# cdo.remapdis('./grid_description.txt', input=ds, output=outfile)
for year in range(2010, 2020):
    for ensemble_member_id in range(1, 11):
        start = time.time()
        path = f"""{artifact_path("raw")}/s{year}-r{ensemble_member_id}i1p1f1/"""
        if os.path.exists(
            f"""{artifact_path("data")}/dcpp_dataset/{year}_{ensemble_member_id}.nc"""
        ):
            print(year, ensemble_member_id, "already done")
            continue

        amon = xr.open_mfdataset(f"{path}/Amon/*", compat="override", engine="netcdf4")[
            amon_surface_variables + level_variables
        ]
        # for_grid = xr.open_mfdataset(f'{path}/Amon/*',compat='override',engine='netcdf4')
        amon = amon.sel(plev=list(pressure_levels))
        amon["rsn"] = amon["rsds"] - amon["rsus"]
        amon["rln"] = amon["rlds"] - amon["rlus"]
        amon = amon.drop_vars(["rsds", "rsus", "rlds", "rlus"])
        emon = xr.open_mfdataset(f"{path}/Emon/*", compat="override", engine="netcdf4")[
            emon_surface_variables
        ]
        omon = xr.open_mfdataset(
            f"{path}/Omon/tos*", compat="minimal", engine="netcdf4"
        )[omon_surface_variables]
        if "height" in list(emon.variables.keys()):
            emon_ocean = emon[
                ["thetaot2000", "thetaot300", "thetaot700", "t20d"]
            ].drop_vars(["depth", "height"])
            emon_other = emon[["rss", "rls"]].drop_vars(["depth", "height"])
        else:
            emon_ocean = emon[
                ["thetaot2000", "thetaot300", "thetaot700", "t20d"]
            ].drop_vars(["depth"])
            emon_other = emon[["rss", "rls"]].drop_vars(["depth"])
        if "type" in list(emon.variables.keys()):
            emon_ocean = emon_ocean.drop_vars(["type"])
            emon_other = emon_other.drop_vars(["type"])
        # simon = xr.open_mfdataset(f'{path}/SImon/*',compat='override',engine='netcdf4')[simon_surface_variables]

        # print(xr.open_mfdataset(f'{path}/Lmon/gpp*',compat='minimal',engine='netcdf4'))
        # print(xr.open_mfdataset(f'{path}/Lmon/cVeg*',compat='minimal',engine='netcdf4'))
        # print(xr.open_mfdataset(f'{path}/Lmon/evspsblsoi*',compat='minimal',engine='netcdf4'))

        # lmon = xr.open_mfdataset(f'{path}/Lmon/*',compat='minimal',engine='netcdf4')[lmon_surface_variables]

        # os.system('rm temp_emon.nc')
        # os.system('rm temp_omon.nc')
        # os.system('rm temp_simon.nc')
        # grid_info = cdo.griddes(input=for_grid)
        # print(grid_info)

        # import numpy as np
        # Define grid parameters
        # lon = emon_ocean['y'].values
        # lat = emon_ocean['x'].values

        # # Approximate grid boundaries
        # lon_bnds = np.append(lon - 0.5 * np.diff(lon)[0], lon[-1] + 0.5 * np.diff(lon)[0])
        # lat_bnds = np.append(lat - 0.5 * np.diff(lat)[0], lat[-1] + 0.5 * np.diff(lat)[0])
        # print(lon_bnds)
        # # Add bounds as variables to the dataset
        # emon_ocean['bounds_nav_lon'] = (("lon", "bnds"), np.stack((lon_bnds[:-1], lon_bnds[1:]), axis=1))
        # emon_ocean['bounds_nav_lat'] = (("lat", "bnds"), np.stack((lat_bnds[:-1], lat_bnds[1:]), axis=1))
        # sethalo,0,1 infile outfile
        cdo.sethalo(
            3,
            0,
            input=emon_ocean,
            output=f"temp_emon{year}_{ensemble_member_id}_lined.nc",
        )
        # cdo.sethalo(3,0,input=simon,output=f'temp_simon{year}_{ensemble_member_id}_lined.nc')
        cdo.sethalo(
            3, 0, input=omon, output=f"temp_omon{year}_{ensemble_member_id}_lined.nc"
        )

        # cdo.setgrid('target_grid.txt', input=emon_ocean, output=f'temp_emon{year}_{ensemble_member_id}_with_grid.nc')
        cdo.remapnn(
            "./grid_description_for_dcpp.txt",
            input=f"temp_emon{year}_{ensemble_member_id}_lined.nc",
            output=f"temp_emon{year}_{ensemble_member_id}.nc",
        )
        emon_ocean_reproj = xr.open_dataset(f"temp_emon{year}_{ensemble_member_id}.nc")

        # emon_ocean_reproj = xr.open_dataset(cdo.remapbic('./grid_description.txt',input=emon_ocean,output=f'temp_emon{year}_{ensemble_member_id}.nc')) #need to do output otherwise HDF error

        omon_reproj = xr.open_dataset(
            cdo.remapnn(
                "./grid_description.txt",
                input=f"temp_omon{year}_{ensemble_member_id}_lined.nc",
                output=f"temp_omon{year}_{ensemble_member_id}.nc",
            )
        )
        # simon_reproj = xr.open_dataset(cdo.remapnn('./grid_description.txt',input=f'temp_simon{year}_{ensemble_member_id}_lined.nc',output=f'temp_simon{year}_{ensemble_member_id}.nc'))

        print("time_before_merge", time.time() - start)

        # amon['lat'] = amon.lat.round(4)
        # emon_other['lat'] = emon_other.lat.round(4)
        # print(amon.lat)
        # print(emon_other.lat)

        emon_ocean_reproj["lon"] = amon["lon"]
        emon_ocean_reproj["lat"] = amon["lat"]

        # print(lmon['lat'])
        print(emon_ocean_reproj["lat"])
        # lmon['lat'] = lmon.lat.round(4)
        # print(emon_ocean_reproj.lat)
        # print(lmon.lat)

        omon_reproj["lat"] = amon["lat"]
        # simon_reproj['lat'] =  amon['lat']
        print(omon_reproj)
        print(emon_ocean_reproj)
        print(emon_other)
        print(amon)

        # omon_reproj.lat = round(omon_reproj.lat,4)
        # simon_reproj.lat = round(simon_reproj.lat,4)
        # print(lmon)

        # print(omon_reproj.lat)
        # print(simon_reproj.lat)
        merged = xr.merge([amon, emon_other, emon_ocean_reproj, omon_reproj])
        print("time_after_merge", time.time() - start)

        loaded = merged.load()
        print("loaded", time.time() - start)
        # print(loaded.lat)
        loaded.to_netcdf(
            f"""{artifact_path("data")}/dcpp_dataset/{year}_{ensemble_member_id}.nc"""
        )
        # amon.close()
        # emon_other.close()
        # omon.close()
        # emon_ocean.close()
        # simon.close()
        # lmon.close()
        # simon_reproj.close()
        # omon_reproj.close()
        # emon_ocean_reproj.close()
        print(year, ensemble_member_id, time.time() - start)
