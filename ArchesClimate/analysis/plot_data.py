import os

import hydra
import torch
from omegaconf import DictConfig
from tensordict.tensordict import TensorDict

from ArchesClimate.analysis.evaluate import plot_all_values_raw
from ArchesClimate.analysis.gen_and_save_metrics import (
    load_climatology,
    load_experiment,
)
from ArchesClimate.utils.utils import unnormalize_data


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    num_members = cfg.analysis.num_members
    num_rollout_steps = cfg.analysis.num_rollout_steps
    var_names = (
        cfg.dataloader.dataset.surface_variables
        + cfg.dataloader.dataset.level_variables
    )
    experiment = cfg.analysis.experiment
    out_dir = experiment + f"_{num_members}_{num_rollout_steps}"
    os.makedirs(f"{cfg.paths.plots}/{out_dir}", exist_ok=True)
    start_lat = cfg.analysis.start_lat
    end_lat = None if cfg.analysis.end_lat == "None" else cfg.analysis.end_lat
    start_lon = cfg.analysis.start_lon
    end_lon = None if cfg.analysis.end_lon == "None" else cfg.analysis.end_lon

    target = cfg.analysis.experiments_to_plot[0].split("_")[-2]
    year = int(cfg.analysis.experiments_to_plot[0].split("_")[-1])
    train_spatial_norm = hydra.utils.instantiate(
        cfg.dataloader.dataset,
        domain="all",
        limit_examples=100,
        norm_scheme="spatial_norm",
    )
    train = hydra.utils.instantiate(
        cfg.dataloader.dataset, domain="all", limit_examples=100
    )
    data = load_experiment(cfg, experiment, num_members, num_rollout_steps)
    denormalized_predicted_ensemble = unnormalize_data(data[None], train)[
        :, :num_members, :num_rollout_steps
    ]
    # denormalized_predicted_ensemble['surface'] = denormalized_predicted_ensemble['surface'] - climatology_surface
    # denormalized_predicted_ensemble['level'] = denormalized_predicted_ensemble['level'] - climatology_level
    denormalized_predicted_ensemble = TensorDict(
        {
            "surface": denormalized_predicted_ensemble["surface"][
                :, :, :, :, :, start_lat:end_lat, start_lon:end_lon
            ],
            "level": denormalized_predicted_ensemble["level"][
                :, :, :, :, :, start_lat:end_lat, start_lon:end_lon
            ],
        }
    )

    # load test_data
    test_ensemble = torch.load(
        f"{cfg.paths.rollouts}/{target}_ensemble_{year}_flow_net_flux_1"
    )
    test_ensemble["surface"] = test_ensemble["surface"][
        :, :, cfg.module.surface_variable_indices
    ]
    denormalized_test_ensemble = unnormalize_data(
        test_ensemble[None], train_spatial_norm
    )[:, : min(num_members, 10), :num_rollout_steps]
    denormalized_test_ensemble = TensorDict(
        {
            "surface": denormalized_test_ensemble["surface"][
                :, :, :, :, :, start_lat:end_lat, start_lon:end_lon
            ],
            "level": denormalized_test_ensemble["level"][
                :, :, :, :, :, start_lat:end_lat, start_lon:end_lon
            ],
        }
    )

    denormalized_test_ensemble.batch_size = [1, min(num_members, 10), num_rollout_steps]
    denormalized_predicted_ensemble.batch_size = [1, num_members, num_rollout_steps]
    climatology_ensemble = load_climatology(
        cfg,
        num_members,
        num_rollout_steps,
        denormalized_predicted_ensemble,
        start_lat,
        end_lat,
        start_lon,
        end_lon,
    )

    start_year = 1969 if (target == 0) else 2011
    # plot_all_values(
    #     denormalized_predicted_ensemble,
    #     climatology_ensemble,
    #     denormalized_test_ensemble,
    #     f'{cfg.paths.plots}/{out_dir}/values',
    #     var_names,
    #     start_year
    # )
    plot_all_values_raw(
        denormalized_predicted_ensemble,
        climatology_ensemble,
        denormalized_test_ensemble,
        f"{cfg.paths.plots}/{out_dir}/values_raw",
        var_names,
        start_year,
    )

    # for var_name in ['clt','tas','huss','tos']:
    #     var_index_for_gif = cfg.module.surface_variables.index(var_name)
    #     make_gif(torch.stack([
    #         denormalized_test_ensemble['surface'][0,:,:118,var_index_for_gif,0,:,:].mean(axis=0)-denormalized_predicted_ensemble['surface'][0,:,:,var_index_for_gif,0,:,:].mean(axis=0),
    #         denormalized_test_ensemble['surface'][0,:,:118,var_index_for_gif,0,:,:].mean(axis=0)-climatology_ensemble['surface'][0,:,:,var_index_for_gif,0,:,:].mean(axis=0),
    #         climatology_ensemble['surface'][0,:,:,var_index_for_gif,0,:,:].mean(axis=0)-denormalized_predicted_ensemble['surface'][0,:,:,var_index_for_gif,0,:,:].mean(axis=0)
    #     ]),
    #              cfg.analysis.num_rollout_steps-2,
    #              f'{var_name}_pred',
    #              f'{cfg.paths.plots}/{out_dir}/{var_name}_pred_differences_gif',
    #              denormalized=True,
    #              save=True,
    #              ffmpeg=True
    #             )
    # for var_name in ['clt','tas','huss','tos']:
    #     var_index_for_gif = cfg.module.surface_variables.index(var_name)
    #     make_gif(torch.stack([
    #         denormalized_predicted_ensemble['surface'][0,:,:,var_index_for_gif,0,:,:].mean(axis=0),
    #         denormalized_test_ensemble['surface'][0,:,:118,var_index_for_gif,0,:,:].mean(axis=0)
    #     ]),
    #              cfg.analysis.num_rollout_steps-2,
    #              f'{var_name}_pred',
    #              f'{cfg.paths.plots}/{out_dir}/{var_name}_pred_actuals_gif',
    #              denormalized=True,
    #              save=True,
    #              ffmpeg=True
    #             )

    # plot_all_values(
    #     mid_lats_predicted,
    #     mid_lats_persistence,
    #     mid_lats_target,
    #     f'plots/{out_dir}/values_60_to_60',
    #     var_names,
    #     start_year
    # )
    # plot_all_values_raw(
    #     mid_lats_predicted,
    #     mid_lats_persistence,
    #     mid_lats_target,
    #     f'plots/{out_dir}/values_raw_60_to_60',
    #     var_names,
    #     start_year
    # )
    # plot_5_squares(
    #     denormalized_predicted_ensemble,
    #     denormalized_test_ensemble[:,:num_members],
    #     f'{cfg.paths.plots}/{out_dir}/five_plots_psl',
    #     var_index=6,
    #     title='Sea Level Pressure Anomaly Comparison'
    # )
    # plot_5_squares(
    #     denormalized_predicted_ensemble,
    #     denormalized_test_ensemble[:,:num_members],
    #     f'{cfg.paths.plots}/{out_dir}/five_plots_sst',
    #     var_index=-1,
    #    title='Sea Surface Temperature Anomaly Comparison'
    # )


#     plot_all_values(
#         data,
#         climatology_ensemble[None],
#         test_ensemble[None],
#         f'plots/{out_dir}/normed_values',
#         var_names,
# start_year
#     )


# SST comparison
# sst = torch.tensor(
#     np.load(f'{cfg.cluster.work_path}/sst_hadi_regridded.npy')
# )[96+2:96+119] #this 96 is for 1969 start date , is that right/??
# print('sst',sst)
# sst = torch.where(sst == -1000,torch.nan,sst)    #120,143,144
# print('sst1',sst)

# pred_mask_for_sst = torch.isnan(data['surface'][0,0,:,-1,0])
# print(pred_mask_for_sst)
# sst[pred_mask_for_sst] = torch.nan
# print(sst)

# #per https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018JC014074
# #(100°W–40°E, 20–90°N)
# na_lats = (int(110 * (143/180)), int(180 * 143//180))
# na_lons = (int(144 - (100 * 144/360)), int(20 * 144/360))
# mask = torch.isnan(sst)
# pred_mask = mask.view(1, 1, 117,143,144).expand(len(experiments), num_members, 117, 143, 144)


# denormalized_predicted_ensemble['surface'][:,:,:,-1,0][pred_mask] = torch.nan
# # print('pred mask',pred_mask)
# # print(pred_mask[0])
# denormalized_test_ensemble['surface'][0,:num_members,:,-1,0][pred_mask[0,:min(num_members,10)]] = torch.nan #TODO: double check indices are correct here


# clima = torch.load('climatology_stats.pt')

# clima_removed_pred = []
# clima_removed_test = []
# clima_removed_hadi = []

# for time_index in range(118):
#     #starting month is 3, because state is 2-15 and first pred is 3-15
#     month_index = (time_index + 2) % 12
#     # print(denormalized_predicted_ensemble['surface'][0,0,time_index])
#     # print('surface_mean',clima['surface_mean'][month_index,cfg.module.surface_variable_indices])
#     # print('surface_std',clima['surface_std'][month_index,cfg.module.surface_variable_indices])
#     clima_removed_pred.append(denormalized_predicted_ensemble['surface'][0,0,time_index] - clima['surface_mean'][month_index,cfg.module.surface_variable_indices] / clima['surface_std'][month_index,cfg.module.surface_variable_indices,None,None])
#     clima_removed_test.append(denormalized_test_ensemble['surface'][0,0,time_index] - clima['surface_mean'][month_index,cfg.module.surface_variable_indices] / clima['surface_std'][month_index,cfg.module.surface_variable_indices,None,None])
#     clima_removed_hadi.append(sst[time_index] - clima['surface_mean'][month_index,-1,0] / clima['surface_std'][month_index,-1,0,None,None])

# this does not support nans
# pred_sst_detrend = signal.detrend(torch.nan_to_num(denormalized_predicted_ensemble['surface'][0,0,:,-1,0]),axis=0)
# # sst_season = np.mean(pred_sst_detrend, axis=0)
# # sst_diff = pred_sst_detrend - sst_season
# # print('sst_diff',sst_diff.shape)
# pred_sst_detrend = np.ma.masked_array(pred_sst_detrend, mask=mask)
# # print('after mask sst_diff',sst_diff.shape)


# hadi_sst_detrend = signal.detrend(torch.nan_to_num(sst),axis=0)
# # hadi_sst_season = np.mean(hadi_sst_detrend, axis=0)
# # hadi_sst_diff = hadi_sst_detrend - hadi_sst_season
# # print('hadi_sst_diff',hadi_sst_diff.shape)
# hadi_sst_detrend = np.ma.masked_array(hadi_sst_detrend, mask=mask)

# test_sst_detrend = signal.detrend(torch.nan_to_num(denormalized_test_ensemble['surface'][0,0,:,-1,0]),axis=0)
# # test_sst_season = np.mean(test_sst_detrend, axis=0)
# # test_sst_diff = test_sst_detrend - test_sst_season
# # print('sst_diff',sst_diff.shape)
# test_sst_detrend = np.ma.masked_array(test_sst_detrend, mask=mask)

# lat_coeffs_equi = np.array(
# [np.cos(x) for x in np.arange(-np.pi / 2, np.pi / 2 + 1e-6, np.pi / 143)]
# )
# lat_coeffs_equi = lat_coeffs_equi[None]
# # print(lat_coeffs_equi.shape)
# # print(lat_coeffs_equi)

# eof_analysis(pred_sst_detrend,test_sst_detrend,hadi_sst_detrend,f'{cfg.paths.plots}/{out_dir}/',lat_coeffs_equi)


# # print(denormalized_predicted_ensemble)
# # print('pred',denormalized_predicted_ensemble['surface'][None,0,:,:,:,:,na_lats[0]:na_lats[1],na_lons[1]:na_lons[0]])

# # generated_metrics_tensor = [generate_metrics(
# #     denormalized_predicted_ensemble['surface'][None,x,:,:,:,:,na_lats[0]:na_lats[1],na_lons[1]:na_lons[0]],
# #     sst[...,na_lats[0]:na_lats[1],na_lons[1]:na_lons[0]],
# #     is_sst=True
# # ) for x in range(denormalized_predicted_ensemble.shape[0])]

# # sst = sst[None,None]

# # ipsl_metrics_tensor = generate_metrics(
# #     denormalized_test_ensemble['surface'][None,0,:,:,:,:,na_lats[0]:na_lats[1],na_lons[1]:na_lons[0]],
# #     sst[...,na_lats[0]:na_lats[1],na_lons[1]:na_lons[0]],
# #     is_sst=True)


# #TODO: replace this with plot_values
# # plot_sst_values(
# #     denormalized_predicted_ensemble,
# #     denormalized_test_ensemble,
# #     sst,f'plots/{out_dir}/pred_sst_values'
# # )


# # #we want a function that takes in multiple ensembles of different compositions and plots them against each other
# # leave_one_out_dict = {}

# # for i in range(1,10):
# #     mixed_ens = torch.cat(
# #         [denormalized_test_ensemble[:1,i:10,:118],
# #         denormalized_predicted_ensemble[:,0:i,:118]]
# #         ,dim=1)
# #     leave_one_out_dict[i] = generate_metrics(
# #     mixed_ens['surface'][None,0,:,:,:,:,na_lats[0]:na_lats[1],na_lons[1]:na_lons[0]],
# #     sst[...,na_lats[0]:na_lats[1],na_lons[1]:na_lons[0]],
# #         is_sst=True
# # )


if __name__ == "__main__":
    main()
