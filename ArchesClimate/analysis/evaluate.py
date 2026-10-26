import datetime
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from eofs.standard import Eof
from matplotlib import cm
from scipy import stats
from tensordict.tensordict import TensorDict

from ArchesClimate.metrics.ensemble_metrics import EnsembleMetrics
from ArchesClimate.paths import path as artifact_path

na_lats = (int(110 * (143 / 180)), (180 * 143 // 180))
na_lons = (int(144 - (100 * 144 / 360)), int(20 * 144 / 360))


# this function takes in an ensemble and a target (single member, such as observations) and returns a tensor of the same rollout length
def generate_monthly_dates(start_year, start_month, num_months):
    dates = []
    current_year = start_year
    current_month = start_month

    for _ in range(num_months):
        # Create a date for the current year and month
        date = datetime.datetime(current_year, current_month, 1)  # noqa: DTZ001 -- timezone-naive climate calendar
        dates.append(date)

        # Increment the month
        current_month += 1
        # If month exceeds 12, reset to January and increment the year
        if current_month > 12:
            current_month = 1
            current_year += 1

    return dates


# def generate_sst_metrics(input_ensemble,sst):
#     print(sst.shape)
#     print(input_ensemble.shape)

#     # experiments = input_ensemble.shape[0]
#     # num_members = input_ensemble.shape[1]
#     num_rollout_steps = input_ensemble.shape[2]
# # torch.Size([1, 1, 21, 10, 1, 143, 144])
# # torch.Size([1, 1, 21, 10, 1, 143, 144])
# # torch.Size([1, 10, 31, 10, 1, 143, 144])

#     surface_output_metrics = []
#     for rollout_index in range(num_rollout_steps-1):
#         surface_metrics = EnsembleMetrics(data_shape=(1,1))
#         # level_metrics = EnsembleMetrics(data_shape=(1,8,3))
#         #needs to be [1, num_pred_members_per_sample (should be 1 for batch), variables, lat, lon]
#         surface_metrics.update(
#             sst[:,0,rollout_index,:,:],
#             input_ensemble['surface'][None,0,:,rollout_index,:,0,na_lats[0]:na_lats[1],na_lons[1]:na_lons[0]]
#         )

#         for metric in [surface_metrics]:
#             out = metric.compute()
#             surface_output_metrics.append(torch.stack(list(out.values())).squeeze())

#     output = torch.stack(surface_output_metrics)
#     # level_output = torch.stack(level_output_metrics)
#     return TensorDict(surface=output)


def generate_metrics(input_ensemble, target_ensemble, is_sst=False):
    # batch_examples = input_ensemble.shape[1]
    # num_members = input_ensemble.shape[1]
    num_rollout_steps = input_ensemble.shape[2]
    #     torch.Size([1, 1, 21, 10, 1, 143, 144])
    # torch.Size([1, 1, 21, 10, 1, 143, 144])
    # for ic_index in range(batch_examples):
    surface_output_metrics = []
    level_output_metrics = []
    for rollout_index in range(num_rollout_steps):
        # for member_index in range(target_ensemble.shape[1]):
        if is_sst:
            surface_metrics = EnsembleMetrics(data_shape=(1, 1), reference_members=1)

        else:
            surface_metrics = EnsembleMetrics(
                data_shape=(1, input_ensemble["surface"].shape[3]),
                reference_members=target_ensemble["surface"].shape[1],
            )
        level_metrics = EnsembleMetrics(
            data_shape=(
                1,
                input_ensemble["level"].shape[3],
                input_ensemble["level"].shape[4],
            ),
            reference_members=1 if is_sst else target_ensemble["level"].shape[1],
        )
        # needs to be [1, num_pred_members_per_sample (should be 1 for batch), variables, lat, lon]
        # for target needs to be [1, variables, lat, lon]

        # print('target',target_ensemble['surface'][0,:,rollout_index,:,0,:,:].mean(axis=0)[None].shape)
        # print('input',input_ensemble['surface'][None,0,:,rollout_index,:,0,:,:].shape)
        if is_sst:
            # print(target_ensemble[:,member_index,rollout_index,:,:])
            # print(input_ensemble[None,0,:,rollout_index,-1,0,:,:])
            surface_metrics.update(
                target_ensemble[:, 0, rollout_index, :, :],  # aka hadisst data
                input_ensemble[None, 0, :, rollout_index, -1, 0, :, :],
            )
            for metric in [surface_metrics]:
                out = metric.compute()
        else:
            surface_metrics.update(
                target_ensemble["surface"][0, :, rollout_index, :, 0, :, :].mean(
                    axis=0
                )[None],
                input_ensemble["surface"][None, 0, :, rollout_index, :, 0, :, :],
            )
            for metric in [surface_metrics]:
                out = metric.compute()
                surface_output_metrics.append(torch.stack(list(out.values())).squeeze())
            level_metrics.update(
                target_ensemble["level"][0, :, rollout_index, :, :, :, :].mean(axis=0)[
                    None
                ],
                input_ensemble["level"][None, 0, :, rollout_index, :, :, :, :],
            )
            for metric in [level_metrics]:
                out = metric.compute()
                level_output_metrics.append(torch.stack(list(out.values())).squeeze())
        # if(not is_sst):
        #     level_output_metrics.append(torch.stack(level_surface_metrics).mean(axis=0))
    output = torch.stack(surface_output_metrics)
    if not is_sst:  # whats a better way? this is so ugly
        level_output = torch.stack(level_output_metrics)
    else:
        level_output = []
    return TensorDict(surface=output, level=level_output)


def get_headline(metrics_tensor_dict):
    """
    old headline is
    tas - surface index 0
    tos - surface index 9
    hus 850 - level index 1 0
    va 850 - level index 5 0
    zg 850 - level index 7 0
    """
    """
    new headline
    siconc - sea-ice concentration index 12
    ta 850 - air temperature index 2 0 
    psl - sea surface pressure index 6
    huss - specifc humidity at surface index 3
    va 850 - northward wind index 3 0 
    thetaot300 - Depth Average Potential Temperature of Upper 300m index 19
    """
    """
    third version
    huss 3 
    tasmax 10 
    tasmin 11 
    tos 17 
    ta 2 
    """
    return torch.concat(  # this works because we only take one level var and then the plevs are the var dim for level
        # [metrics_tensor_dict['surface'][:,:,[1,6,-5,-1]],
        # metrics_tensor_dict['level'][:,:,[0,2,5,6],1].reshape(metrics_tensor_dict['level'].shape[0],
        #                                                metrics_tensor_dict['level'].shape[1],                                                 metrics_tensor_dict['level'].shape[2]*metrics_tensor_dict['level'].shape[3])],
        [
            metrics_tensor_dict["surface"][:, :, [1, 6, -5]],
            metrics_tensor_dict["level"][:, :, [2], 1].reshape(
                metrics_tensor_dict["level"].shape[0],
                metrics_tensor_dict["level"].shape[1],
                1,
            ),
        ],
        dim=2,
    )
    # all but cveg
    # return  torch.concat([ # this works because we only take one level var and then the plevs are the var dim for level
    #     torch.concat([metrics_tensor_dict['surface'][:,:,:-6],metrics_tensor_dict['surface'][:,:,-5:]],dim=2),
    #     metrics_tensor_dict['level'][:,:,:,:].reshape(metrics_tensor_dict['level'].shape[0],
    #                                                    metrics_tensor_dict['level'].shape[1],                                                 metrics_tensor_dict['level'].shape[2]*metrics_tensor_dict['level'].shape[3])],
    #     dim=2
    # )


def plot_leave_one_out(gen_metrics, ipsl, file_name):
    # ipsl = ipsl_metrics['surface'][:,:,9]
    fig, axes = plt.subplots(3, figsize=(16, 16))
    axes = axes.flatten()
    for count, (exp, pred_head) in enumerate(gen_metrics.items()):
        pred_sst = pred_head["surface"][:, :]
        axes[0].plot(
            1 - (pred_sst[:, 1] / ipsl[:, 1]), label=exp, color=cm.brg(count / 10)
        )
        # axes[1].plot(1 - (pred_sst[:,2]/ipsl[:,2]),color=cm.brg(count/10))
        axes[1].plot(1 - (pred_sst[:, 3] / ipsl[:, 3]), color=cm.brg(count / 10))
        axes[2].plot(1 - (pred_sst[:, 4] / ipsl[:, 4]), color=cm.brg(count / 10))
    axes[0].set_title("rmse")
    #  axes[1].set_title('var')
    axes[1].set_title("sprskr")
    axes[2].set_title("crps")
    plt.title("SST vs HadiSST Replacement Test (more red is more predicted members)")
    fig.tight_layout()
    plt.figlegend()
    plt.title("Leave-one-out experiment vs. HadiSST")
    fig.savefig(file_name + ".png")


def plot_headline_metrics_agg(
    output_metrics_list, climatology_metrics_tensor_dict, file_name, legend_title
):
    from cycler import cycler

    # Tableau Colorblind 10 palette (safe for most forms of color blindness)
    colorblind_colors = [
        "#0173b2",
        "#de8f05",
        "#029e73",
        "#d55e00",
        "#cc78bc",
        "#ca9161",
        "#fbafe4",
        "#949494",
        "#ece133",
        "#56b4e9",
    ]
    color_cycle = cycler(color=colorblind_colors)

    # Make an iterator
    color_iterator = iter(color_cycle)

    pred_headlines = {k: get_headline(v) for k, v in output_metrics_list.items()}
    clima_headline = get_headline(climatology_metrics_tensor_dict)
    fig, axes = plt.subplots(2, figsize=(16, 8))
    axes = axes.flatten()

    for exp, pred_head in pred_headlines.items():
        next_color = next(color_iterator)["color"]
        axes[0].plot((pred_head[:, 2]).nanmean(axis=-1), label=exp, color=next_color)
        axes[1].plot((pred_head[1:, 3]).nanmean(axis=-1), color=next_color)

    next_color = next(color_iterator)["color"]
    axes[0].plot(
        (clima_headline[:, 2]).nanmean(axis=-1),
        linestyle="--",
        label="IPSL",
        color=next_color,
    )
    axes[1].plot(
        (clima_headline[1:, 3]).nanmean(axis=-1), linestyle="--", color=next_color
    )

    #  axes[2].plot(1 - (pred_head[:,4]/ clima_headline[:,4]).mean(axis=-1))

    axes[0].set_title("RMSE (↑)", fontsize=18)
    # axes[1].set_title('var =')
    axes[1].set_title("Var", fontsize=18)
    # axes[2].set_title('CRPS (↑)',fontsize=18)

    # axes[0].axhline(0, color='gray', linestyle='--', linewidth=1)  # Dotted line at zero
    # # axes[1].axhline(0, color='gray', linestyle='--', linewidth=1)  # Dotted line at zero
    # axes[1].axhline(1, color='gray', linestyle='--', linewidth=1)
    # axes[2].axhline(0, color='gray', linestyle='--', linewidth=1)  # Dotted line at zero
    axes[1].set_xlabel("Month", fontsize=17)
    axes[0].tick_params(axis="both", which="major", labelsize=16)
    axes[1].tick_params(axis="both", which="major", labelsize=16)
    # axes[2].tick_params(axis='both', which='major', labelsize=16)

    fig.tight_layout()
    plt.figlegend(title=legend_title, title_fontsize=16, fontsize=15)
    # fig.suptitle('10 Year Rollout',fontsize=8)
    fig.savefig(file_name + ".png")


def plot_headline_metrics(
    output_metrics_list, climatology_metrics_tensor_dict, file_name
):
    # pred_headline = get_headline(output_metrics_tensor_dict)
    pred_headlines = {k: get_headline(v) for k, v in output_metrics_list.items()}

    clima_headline = get_headline(climatology_metrics_tensor_dict)
    fig, axes = plt.subplots(4, figsize=(16, 16))
    axes = axes.flatten()
    cycol = cycle(["red", "black", "blue"])
    for pred_head in pred_headlines.values():
        col = next(cycol)
        axes[0].plot(1 - (pred_head[:, 1] / clima_headline[:, 1]), color=col)
        axes[1].plot(1 - (pred_head[:, 2] / clima_headline[:, 2]), color=col)
        axes[2].plot(1 - (pred_head[:, 3] / clima_headline[:, 3]), color=col)
        axes[3].plot(1 - (pred_head[:, 4] / clima_headline[:, 4]), color=col)
    axes[0].set_title("rmse")
    axes[1].set_title("var")
    axes[2].set_title("sprskr")
    axes[3].set_title("crps")
    fig.tight_layout()
    plt.figlegend()

    fig.savefig(file_name + ".png")


def plot_individual_metrics(
    output_metrics_list, climatology_metrics_tensor_dict, file_name, var_names
):
    # plot metrics for each var
    fig, axes = plt.subplots(
        output_metrics_list[0]["surface"].shape[2]
        + (
            output_metrics_list[0]["level"].shape[2]
            * output_metrics_list[0]["level"].shape[3]
        ),
        figsize=(16, 64),
    )
    axes = axes.flatten()
    cycol = cycle(["red", "black", "blue"])
    for experiment in output_metrics_list:
        col = next(cycol)
        for index in range(experiment["surface"].shape[2]):
            axes[index].plot(
                1
                - (
                    experiment["surface"][:, 1, index]
                    / climatology_metrics_tensor_dict["surface"][:, 1, index]
                ),
                linestyle="dashed",
                color=col,
                label="rmse",
            )
            # axes[index].plot(1 - (experiment['surface'][:,2,index]/climatology_metrics_tensor_dict['surface'][:,2,index]),linestyle='solid',color=col,label='var')
            experiment["surface"][0, 3, index] = torch.nan  # sssr explodes at beginning
            axes[index].plot(
                (experiment["surface"][:, 3, index]),
                linestyle="dotted",
                color=col,
                label="SSSR",
            )
            # axes[index].plot((experiment['surface'][:,0,index]),linestyle='solid',color=col,label='MSE')
            # axes[index].plot((experiment['surface'][:,2,index]),linestyle='dashdot',color=col,label='Var')

            # axes[index].plot(1 - (experiment['surface'][:,4,index]/climatology_metrics_tensor_dict['surface'][:,4,index]),marker='H',color=col,label='crps')
            if index == 0:
                axes[index].legend()
            axes[index].set_title(var_names[index])
            axes[index].axhline(
                0, color="gray", linestyle="--", linewidth=1
            )  # Dotted line at zero
        for index in range(experiment["level"].shape[2]):
            for plev_index in range(experiment["level"].shape[3]):
                axes_index = (
                    experiment["surface"].shape[2]
                    + (experiment["level"].shape[3] * index)
                    + plev_index
                )
                axes[axes_index].plot(
                    1
                    - (
                        experiment["level"][:, 1, index, plev_index]
                        / climatology_metrics_tensor_dict["level"][
                            :, 1, index, plev_index
                        ]
                    ),
                    linestyle="dashed",
                    color=col,
                )
                # axes[axes_index].plot(1 - (experiment['level'][:,2,index,plev_index]/climatology_metrics_tensor_dict['level'][:,2,index,plev_index]),linestyle='solid',color=col)
                experiment["level"][0, 3, index, plev_index] = (
                    torch.nan
                )  # sssr explodes at beginning

                axes[axes_index].plot(
                    (experiment["level"][:, 3, index, plev_index]),
                    linestyle="dotted",
                    color=col,
                )
                # axes[axes_index].plot(1 - (experiment['level'][:,4,index,plev_index]/climatology_metrics_tensor_dict['level'][:,4,index,plev_index]),marker='H',color=col)
                axes[axes_index].set_title(
                    var_names[experiment["surface"].shape[2] + index]
                )
                axes[axes_index].axhline(
                    0, color="gray", linestyle="--", linewidth=1
                )  # Dotted line at zero

            # axes[(4*index)+1].plot(pred_head[:,2]/clima_headline[:,2])
            # axes[(4*index)+2].plot(pred_head[:,3]/clima_headline[:,3])
            # axes[(4*index)+3].plot(pred_head[:,4]/clima_headline[:,4])

    fig.tight_layout()
    # plt.figlegend()
    # plt.legend()
    fig.savefig(file_name + ".png")


def plot_headline_individual_metrics(
    output_metrics_list,
    climatology_metrics_tensor_dict,
    file_name,
    var_names,
    metric_index,
    legend_title,
    pred_dict_land=None,
    itself_metrics_land=None,
    start_year=1969,
    start_month=3,
):
    # start_year/start_month default to the original hardcoded value (correct
    # for the common case: test_0 on the "dcpp" filter, whose real test years
    # genuinely start at 1969) -- pass the real start date explicitly for any
    # other target/filter combination (e.g. val/test on 2010-2016).
    # plot metrics for each var

    _palette = [
        "#0173b2",
        "#de8f05",
        "#029e73",
        "#d55e00",
        "#cc78bc",
        "#ca9161",
        "#fbafe4",
        "#949494",
        "#ece133",
        "#56b4e9",
    ]
    exp_colors = {
        exp: _palette[i % len(_palette)] for i, exp in enumerate(output_metrics_list)
    }

    pred_headlines = {k: get_headline(v) for k, v in output_metrics_list.items()}
    headline_var_names = ["net_flux", "psl", "thetaot2000", "ta"]
    headline_ylabels = ["W m-2", "Pa", "degC", "K"]

    has_tas = pred_dict_land is not None and itself_metrics_land is not None
    n_rows = 6 if has_tas else 4
    TAS_IDX = 10

    T = next(iter(output_metrics_list.values()))["surface"].shape[0]
    x = generate_monthly_dates(start_year, start_month, T)

    clima_headline = get_headline(climatology_metrics_tensor_dict)
    fig, axes = plt.subplots(n_rows, sharex=True)
    axes = axes.flatten()

    for exp, pred_head in pred_headlines.items():
        col = exp_colors[exp]
        for index in range(pred_head.shape[2]):
            axes[index].plot(
                x,
                pred_head[:, metric_index, index],
                linestyle="solid",
                color=col,
                label=exp,
            )
            axes[index].set_ylabel(headline_ylabels[index])
            axes[index].tick_params(axis="both", which="major")

    for index in range(pred_headlines[next(iter(pred_headlines))].shape[2]):
        axes[index].set_title(headline_var_names[index])
        axes[index].plot(
            x,
            clima_headline[:, metric_index, index],
            linestyle="dashed",
            color="black",
            label="ipsl",
        )

    if has_tas:
        # TAS global (subplot 4): all experiments from global metrics
        for exp, metrics in output_metrics_list.items():
            axes[4].plot(
                x,
                metrics["surface"][:, metric_index, TAS_IDX],
                linestyle="solid",
                color=exp_colors[exp],
                label=exp,
            )
        axes[4].plot(
            x,
            climatology_metrics_tensor_dict["surface"][:, metric_index, TAS_IDX],
            linestyle="dashed",
            color="black",
            label="ipsl",
        )
        axes[4].set_title("tas (global)")
        axes[4].set_ylabel("K")
        axes[4].tick_params(axis="both", which="major")

        # TAS land-only (subplot 5): experiments from land metrics
        land_colors = {
            exp: _palette[i % len(_palette)] for i, exp in enumerate(pred_dict_land)
        }
        for exp, metrics in pred_dict_land.items():
            axes[5].plot(
                x,
                metrics["surface"][:, metric_index, TAS_IDX],
                linestyle="solid",
                color=land_colors[exp],
                label=exp,
            )
        axes[5].plot(
            x,
            itself_metrics_land["surface"][:, metric_index, TAS_IDX],
            linestyle="dashed",
            color="black",
            label="ipsl",
        )
        axes[5].set_title("tas (land)")
        axes[5].set_ylabel("K")
        axes[5].tick_params(axis="both", which="major")

    if metric_index == 4:
        axes[0].legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=2,
            fontsize=10,
            frameon=False,
        )
    axes[-1].set_xlabel("Month")
    for ax in axes[:-1]:
        ax.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )

    plt.subplots_adjust(hspace=0.4)
    fig.set_size_inches((7, 1.5 * n_rows))
    fig.savefig(file_name + ".pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_all_values_raw(
    generated_data, ipsl_data, test_data, file_name, var_names, start_year
):

    fig, axes = plt.subplots(
        generated_data["level"].shape[3] * 3 + generated_data["surface"].shape[3],
        figsize=(16, 64),
    )
    axes = axes.flatten()
    num_surface_vars = generated_data["surface"].shape[3]
    # plot all surface variables
    for index in range(num_surface_vars):
        for exp_index in range(generated_data.shape[0]):
            for member_index in range(generated_data.shape[1]):
                axes[index].plot(
                    generated_data["surface"][
                        exp_index, member_index, :, index, :, :
                    ].nanmean(axis=(-1, -2)),
                    color="blue",
                    lw=0.3,
                )

        for member_index in range(ipsl_data["surface"].shape[1]):
            axes[index].plot(
                ipsl_data["surface"][:, member_index, :, index, :, :].nanmean(
                    axis=(0, -1, -2)
                ),
                color="black",
                lw=0.3,
            )
        for member_index in range(test_data["surface"].shape[1]):
            axes[index].plot(
                test_data["surface"][:, member_index, :, index, :, :].nanmean(
                    axis=(0, -1, -2)
                ),
                color="red",
                lw=0.3,
            )
        axes[index].set_title(var_names[index])
    for var_index in range(7):
        for plev in range(3):
            for exp_index in range(generated_data.shape[0]):
                for member_index in range(generated_data.shape[1]):
                    axes[num_surface_vars + (var_index * 3) + plev].plot(
                        generated_data["level"][
                            exp_index, member_index, :, var_index, plev, :, :
                        ].nanmean(axis=(-1, -2)),
                        color="blue",
                        lw=0.1,
                    )
            axes[num_surface_vars + (var_index * 3) + plev].plot(
                ipsl_data["level"][:, :, :, var_index, plev, :, :].nanmean(
                    axis=(0, 1, -1, -2)
                ),
                color="black",
            )
            axes[num_surface_vars + (var_index * 3) + plev].plot(
                test_data["level"][:, :, :, var_index, plev, :, :].nanmean(
                    axis=(0, 1, -1, -2)
                ),
                color="red",
            )
            axes[num_surface_vars + (var_index * 3) + plev].set_title(
                var_names[num_surface_vars + (var_index)]
            )
    fig.tight_layout()
    plt.figlegend()
    fig.savefig(f"{file_name}.png")


def generate_map(data, axes, ds, title, vmin, vmax, datatype="mesh", colorbar=False):
    from mpl_toolkits.basemap import Basemap

    m = Basemap(
        projection="cyl",
        llcrnrlon=min(ds.lon),
        llcrnrlat=min(ds.lat),
        urcrnrlon=max(ds.lon),
        urcrnrlat=max(ds.lat),
        ax=axes,
    )
    x, y = m(*np.meshgrid(ds.lon, ds.lat))
    if datatype == "mesh":
        cs = m.pcolormesh(x, y, data, cmap=plt.cm.RdBu_r, vmin=vmin, vmax=vmax)

    else:
        clevs = np.linspace(np.nanmin(data), np.nanmax(data), 21)
        cs = m.contourf(x, y, data, clevs, cmap=plt.cm.RdBu_r, vmin=vmin, vmax=vmax)
    m.drawcoastlines()
    # m.drawparallels(parallels, labels=[1,0,0,0])
    # m.drawmeridians(meridians, labels=[1,0,0,1])
    from matplotlib import ticker

    def myfmt(x, pos):
        return f"{x:.1f}"

    # cb = plt.cm.ScalarMappable(cmap=cm.coolwarm)
    if colorbar:
        # pad='2%',
        cb = m.colorbar(cs, "right", format=ticker.FuncFormatter(myfmt), shrink=1.5)
        cb.ax.tick_params(labelsize=12)

    # # cb.set_label('EOF', fontsize=12)
    # cb.set_clim(-0.1, 0.1)
    plt.title(title, fontsize=12)


def plot_5_squares(generated_data, test_data, file_name, var_index, title):
    eof_fig, axes = plt.subplots(4, 5, figsize=(16, 16))
    eof_fig.set_size_inches((12, 12))

    axes = axes.flatten()
    scratch = str(artifact_path("data"))
    ds = xr.open_dataset(f"{scratch}/ipsl_processed/1960_1.nc")
    psl_ipsl_ens_mean_month_0 = test_data["surface"][0, :, 1, var_index, 0].mean(axis=0)
    psl_ipsl_ens_mean_month_119 = test_data["surface"][0, :, -1, var_index, 0].mean(
        axis=0
    )
    vmax = torch.nan_to_num(
        torch.stack(
            [
                generated_data["surface"][0, :10, -1, var_index]
                - psl_ipsl_ens_mean_month_119,
                test_data["surface"][0, :, -1, var_index] - psl_ipsl_ens_mean_month_119,
            ]
        )
    ).max()
    vmin = torch.nan_to_num(
        torch.stack(
            [
                generated_data["surface"][0, :10, 0, var_index]
                - psl_ipsl_ens_mean_month_0,
                test_data["surface"][0, :, 0, var_index] - psl_ipsl_ens_mean_month_0,
            ]
        )
    ).min()
    row_names = [
        "March 1969 generated",
        "March 1969 IPSL",
        "February 1979 generated",
        "February 1979 IPSL",
    ]

    for row_index in range(4):
        for member_index in range(5):
            if row_index == 0:
                eof_fig.text(
                    0.2 + (member_index * 0.16),
                    0.7,  # Adjust positioning based on layout
                    f"Ensemble Member {member_index}",
                    ha="center",
                    fontsize=12,
                )
            if member_index == 2:
                eof_fig.text(
                    0.19 + (member_index * 0.16),
                    0.395 + (0.095 * row_index),
                    f"{row_names[row_index]}",
                    ha="center",
                    fontsize=10,
                )
            if member_index == 4:
                colorbar = True
            else:
                colorbar = False
            if row_index < 2:
                time_index = 0
                mean = psl_ipsl_ens_mean_month_0
            else:
                time_index = -1
                mean = psl_ipsl_ens_mean_month_119
            if row_index % 2 == 0:
                generate_map(
                    generated_data["surface"][0, member_index, time_index, var_index, 0]
                    - mean,
                    axes[(row_index * 5) + member_index],
                    ds,
                    "",
                    vmin,
                    vmax,
                    colorbar=colorbar,
                )
            else:
                generate_map(
                    test_data["surface"][0, member_index, time_index, var_index, 0]
                    - mean,
                    axes[(row_index * 5) + member_index],
                    ds,
                    "",
                    vmin,
                    vmax,
                    colorbar=colorbar,
                )

    plt.subplots_adjust(hspace=-0.80, wspace=0.1)
    eof_fig.suptitle(title, y=0.73, fontsize=14)
    eof_fig.savefig(file_name, bbox_inches="tight")


def plot_5_squares_old(generated_data, test_data, file_name, var_index, title):
    # sst = torch.nan_to_num(sst)
    # fig, axes = plt.subplots(2,5, figsize=(32, 16))

    # axes = axes.flatten()
    # # for i in range(5):
    #     axes[i].pcolormesh(generated_data['level'][0,0,0,2,1])
    # for i in range(5):
    #     axes[5+i].pcolormesh(test_data['level'][0,0,0,2,1])
    scratch = str(artifact_path("data"))

    ds = xr.open_dataset(f"{scratch}/ipsl_processed/1960_1.nc")

    fig = plt.figure(figsize=(50, 50))
    # fig.suptitle(title)
    # fig.set_figheight(30)
    # fig.set_figwidth(30)
    # Create 5 columns (ensemble members), each with 4 rows
    num_cols = 5
    num_rows = 4
    row_names = [
        "March 1969 generated",
        "March 1969 IPSL",
        "February 1979 generated",
        "February 1979 IPSL",
    ]
    psl_ipsl_ens_mean_month_0 = test_data["surface"][0, :, 1, var_index, 0].mean(axis=0)
    psl_ipsl_ens_mean_month_119 = test_data["surface"][0, :, -1, var_index, 0].mean(
        axis=0
    )
    vmax = torch.nan_to_num(
        torch.stack(
            [
                generated_data["surface"][0, :10, -1, var_index]
                - psl_ipsl_ens_mean_month_119,
                test_data["surface"][0, :, -1, var_index] - psl_ipsl_ens_mean_month_119,
            ]
        )
    ).max()
    vmin = torch.nan_to_num(
        torch.stack(
            [
                generated_data["surface"][0, :10, 0, var_index]
                - psl_ipsl_ens_mean_month_0,
                test_data["surface"][0, :, 0, var_index] - psl_ipsl_ens_mean_month_0,
            ]
        )
    ).min()
    psl_dses = [
        generated_data["surface"][0, :, 0, var_index, 0],
        test_data["surface"][0, :, 0, var_index, 0],
        generated_data["surface"][0, :, -1, var_index, 0],
        test_data["surface"][0, :, -1, var_index, 0],
    ]
    # Loop through columns (ensemble members)
    for col in range(num_cols):
        # Add a title for each column
        fig.text(
            0.2 + (col * 0.16),
            0.7,  # Adjust positioning based on layout
            f"Ens. Member {col}",
            ha="center",
            fontsize=25,
        )

        # Loop through rows for each column
        for row in range(num_rows):
            ax = fig.add_subplot(num_rows, num_cols, col + 1 + row * num_cols)

            if row < 2:
                data = psl_dses[row][col] - psl_ipsl_ens_mean_month_0
            else:
                data = psl_dses[row][col] - psl_ipsl_ens_mean_month_119
            print(psl_dses[row][col])
            if col == 4:
                generate_map(
                    (data).numpy(),
                    ax,
                    ds,
                    "",
                    vmin.numpy(),
                    vmax.numpy(),
                    colorbar=True,
                )
            else:
                generate_map(
                    (data).numpy(),
                    ax,
                    ds,
                    "",
                    vmin.numpy(),
                    vmax.numpy(),
                    colorbar=False,
                )

            # # Get the color mesh for color bar
            # for PCM in ax.get_children():
            #     if isinstance(PCM, mpl.collections.QuadMesh):
            #         break
            # ax.set_aspect('equal')

            # Configure axes
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 2:
                ax.set_title(f"{row_names[row]}", fontsize=25, y=0.98)
            # if col == 4:
            #     fig.colorbar(im, ax=ax, orientation='vertical', location='right')
    # fig = plt.figure(constrained_layout=True)

    # # create 3x1 subfigs
    # subfigs = fig.subfigures(nrows=1, ncols=5)
    # row_names = ['1969 predicted','1969 target','1979 predicted','1979 target']
    # psl_ipsl_ens_mean_year_0 = test_data['surface'][0,:,0,6,0].mean(axis=0)
    # psl_ipsl_ens_mean_year_10 = test_data['surface'][0,:,-1,6,0].mean(axis=0)

    # # dses = [generated_data['level'][0,:,0,2,1],test_data['level'][0,:,0,2,1],generated_data['level'][0,:,-1,2,1],test_data['level'][0,:,-1,2,1]]
    # psl_dses = [generated_data['surface'][0,:,0,6,0],test_data['surface'][0,:,0,6,0],generated_data['surface'][0,:,-1,6,0],test_data['surface'][0,:,-1,6,0]]

    # for col, subfig in enumerate(subfigs):

    #     subfig.suptitle(f'Ensemble Member {col}',fontsize=8)

    #     # create 1x3 subplots per subfig
    #     axs = subfig.subplots(nrows=4, ncols=1)
    #     for row, ax in enumerate(axs):
    #         if(row < 2):
    #             im = ax.pcolormesh(psl_dses[row][col] - psl_ipsl_ens_mean_year_0,cmap='seismic')
    #         else:
    #             im = ax.pcolormesh(psl_dses[row][col] - psl_ipsl_ens_mean_year_10,cmap='seismic')
    #         PCM=ax.get_children()[2] #https://stackoverflow.com/questions/23090791/matplotlib-colorbar-not-working-due-to-garbage-collection
    #         import matplotlib as mpl
    #         for PCM in ax.get_children():
    #             if isinstance(PCM, mpl.collections.QuadMesh):
    #                 break
    #         # ax.tick_params(axis='y', labelsize=5)
    #         # ax.tick_params(axis='x', labelsize=5)
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         if(col == 2):
    #             ax.set_title(f'{row_names[row]}',fontsize=8,y=1.2)
    #         if(col == 4):
    #             subfig.colorbar(im, cmap='seismic',ax=ax, location='right',use_gridspec=True)
    # fig.suptitle('Sea Level Pressure Anomaly Comparison')

    plt.subplots_adjust(hspace=-0.80, wspace=0.05)

    fig.suptitle(title, y=0.73, fontsize=25)
    # fig.tight_layout() #fig tight layout does not work at all with subplots and colorbars
    fig.savefig(f"{file_name}.png", bbox_inches="tight")


# def plot_sst_values(generated_data,ipsl_data,sst,file_name):

#     # sst = torch.nan_to_num(sst)
#     fig, axes = plt.subplots(1, figsize=(16, 16))
#     #torch.Size([3, 2, 3, 26, 1, 143, 144])
#     for exp_index in range(generated_data.shape[0]):
#         axes.plot(
#             generated_data['surface'][exp_index,:,:,-1,0,na_lats[0]:na_lats[1],na_lons[1]:na_lons[0]].nanmean(axis=(0,-1,-2)),
#             label=exp_index,color=cm.ocean(exp_index))
#     axes.plot(
#         ipsl_data['surface'][...,-1,0,na_lats[0]:na_lats[1],na_lons[1]:na_lons[0]].nanmean(axis=(0,1,-1,-2)),
#         label='clima')
#     axes.plot(sst[...,:,na_lats[0]:na_lats[1],na_lons[1]:na_lons[0]].nanmean(axis=(0,1,-1,-2)),label='HadiSST')
#     fig.tight_layout()
#     plt.figlegend()
#     fig.savefig(f'{file_name}.png',bbox_inches='tight')
#     return


def eof_analysis(pred_data_for_eof, ipsl_data_for_eof, hadisst, out_dir, weights):
    # import lovely_tensors as lt
    # lt.monkey_patch()
    # ipsl_data_for_eof = denormalized_test_ensemble.reshape(-1,143,144)
    # pred_data_for_eof = denormalized_predicted_ensemble.reshape(-1,143,144)
    # hadisst = hadisst.reshape(-1,143,144)
    # instead of taking mean, flatten by members per Guillaume Gastineau's suggestion

    ipsl_solver = Eof(ipsl_data_for_eof, weights=weights)
    pred_solver = Eof(pred_data_for_eof)
    hadi_solver = Eof(hadisst)

    number_of_eofs = 3
    ipsl_eofs = ipsl_solver.eofs(neofs=number_of_eofs).squeeze()
    pred_eofs = pred_solver.eofs(neofs=number_of_eofs).squeeze()
    hadi_eofs = hadi_solver.eofs(neofs=number_of_eofs).squeeze()
    # print('hadi_eofs',np.nanmax(hadi_eofs[0]))
    # print('pred_eofs',np.nanmax(pred_eofs[0]))
    # print('hadi_eofs',np.nanmax(hadi_eofs[1]))
    # print('pred_eofs',np.nanmax(pred_eofs[1]))
    # print('hadi_eofs',np.nanmax(hadi_eofs[2]))
    # print('pred_eofs',np.nanmax(pred_eofs[2]))
    # vmax = np.nanmax(np.stack([ipsl_eofs,pred_eofs,hadi_eofs]))
    # vmin = np.nanmin(np.stack([ipsl_eofs,pred_eofs,hadi_eofs]))
    eof_fig, eof_axes = plt.subplots(number_of_eofs, 3, figsize=(16, 16))
    eof_axes = eof_axes.flatten()
    scratch = str(artifact_path("data"))
    ds = xr.open_dataset(f"{scratch}/ipsl_processed/1960_1.nc")
    # parallels = np.arange(-90,90,30.)
    # meridians = np.arange(-180,180,30)
    for eof_index in range(number_of_eofs):
        vmax = np.nanmax(
            np.stack([ipsl_eofs[eof_index], pred_eofs[eof_index], hadi_eofs[eof_index]])
        )
        vmin = np.nanmin(
            np.stack([ipsl_eofs[eof_index], pred_eofs[eof_index], hadi_eofs[eof_index]])
        )
        # if(eof_index==1):
        #     shell.data = -1*ipsl_eofs[eof_index]
        # else:
        #     shell.data = ipsl_eofs[eof_index]

        generate_map(
            ipsl_eofs[eof_index, :, :].squeeze(),
            eof_axes[eof_index * 3],
            ds,
            f"IPSL EOF{eof_index}",
            vmin,
            vmax,
        )

        generate_map(
            pred_eofs[eof_index, :, :].squeeze(),
            eof_axes[eof_index * 3 + 1],
            ds,
            f"Generated EOF{eof_index}",
            vmin,
            vmax,
        )
        generate_map(
            hadi_eofs[eof_index, :, :].squeeze(),
            eof_axes[eof_index * 3 + 2],
            ds,
            f"HadiSST EOF{eof_index}",
            vmin,
            vmax,
        )
        # cbar = fig.colorbar(mesh, cax=eof_axes[eof_index*3+2])

        # shell.plot.pcolormesh(ax=eof_axes[eof_index*3],add_colorbar=True,add_labels=False,cmap='seismic')
        # eof_axes[eof_index*3].set_title(f'IPSL EOF {eof_index}')

        # m = Basemap(projection='cyl',
        #             llcrnrlon=min(ds.lon),
        #             llcrnrlat=min(ds.lat),
        #             urcrnrlon=max(ds.lon),
        #             urcrnrlat=max(ds.lat),
        #             ax=eof_axes[eof_index*3]
        #            )
        # x, y = m(*np.meshgrid(ds.lon, ds.lat))
        # clevs = np.linspace(np.min(pred_eofs[eof_index,:,:].squeeze()), np.max(pred_eofs[eof_index,:,:].squeeze()), 21)
        # cs = m.contourf(x, y, pred_eofs[eof_index,:,:].squeeze(), clevs, cmap=plt.cm.RdBu_r)
        # m.drawcoastlines()
        # # m.drawparallels(parallels, labels=[1,0,0,0])
        # # m.drawmeridians(meridians, labels=[1,0,0,1])

        # cb = m.colorbar(cs, 'right', size='5%', pad='2%')
        # cb.set_label('EOF', fontsize=12)
        # plt.title('EOF ' + str(eof_index), fontsize=16)

        # shell.data = pred_eofs[eof_index]
        # shell.plot.pcolormesh(ax=eof_axes[eof_index*3+1],add_colorbar=True,add_labels=False,cmap='seismic')
        # eof_axes[eof_index*3+1].set_title(f'Predicted EOF {eof_index}')
        # shell.data = hadi_eofs[eof_index]
        # shell.plot.pcolormesh(ax=eof_axes[eof_index*3+2],add_colorbar=True,cmap='seismic')
        # eof_axes[eof_index*3+2].set_title(f'HadiSST EOF {eof_index}')
    # eof_fig.tight_layout(pad=0)
    # eof_fig.tight_layout()
    # eof_fig.set_size_inches(12, 12)  # OR
    plt.subplots_adjust(hspace=-0.80, wspace=0.1)

    eof_fig.savefig(f"{out_dir}/eof_comparison.png", bbox_inches="tight")


def get_power_spectral_density(input):
    input = input[: input.shape[0], : input.shape[1] - 1]
    # I think this just makes square image
    npix = input.shape[1]

    fourier_image = np.fft.fftn(input)
    fourier_amplitudes = np.abs(fourier_image) ** 2

    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, npix // 2 + 1, 1.0)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(
        knrm, fourier_amplitudes, statistic="mean", bins=kbins
    )
    Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
    return kvals, Abins


def power_spectra(
    denormalized_predicted_ensemble, denorm_pers, denormalized_test_ensemble, out_file
):
    fig, ax = plt.subplots(1, 7, figsize=(60, 10))
    ax = ax.flatten()
    variables = [
        "Relative Humidity",
        "Specific Humidity",
        "Air Temperature",
        "Eastward Wind",
        "Northward Wind",
        "Omega",
        "Geopotential",
    ]

    timestep_index = 3

    # for member_index in range(10):
    # var_indices = [['level',6,2],['level',-1,2],['level',2,2],['level',3,2],['level',-1,2]]
    for ax_index in range(7):
        pers_freq = []
        pers_amp = []
        pred_freq = []
        pred_amp = []
        test_freq = []
        test_amp = []
        for timestep_index in range(denormalized_predicted_ensemble["level"].shape[2]):
            # timestep_index = 0
            # for member_index in range(10):
            for var_index in range(7):
                logfreq1, logamp1 = get_power_spectral_density(
                    denormalized_predicted_ensemble["level"][
                        0, :, timestep_index, ax_index, 2
                    ].mean(axis=0)
                )
                pred_freq.append(logfreq1)
                pred_amp.append(logamp1)

                logfreq2, logamp2 = get_power_spectral_density(
                    denorm_pers["level"][0, :, timestep_index, ax_index, 2].mean(axis=0)
                )
                pers_freq.append(logfreq2)
                pers_amp.append(logamp2)

                logfreq3, logamp3 = get_power_spectral_density(
                    denormalized_test_ensemble["level"][
                        0, :, timestep_index, ax_index, 2
                    ].mean(axis=0)
                )

                test_freq.append(logfreq3)
                test_amp.append(logamp3)

        ax[ax_index].loglog(
            np.nanmean(np.stack(pred_freq), axis=0),
            np.nanmean(np.stack(pred_amp), axis=0),
            label="ArchesClimate",
        )
        ax[ax_index].loglog(
            np.nanmean(np.stack(pers_freq), axis=0),
            np.nanmean(np.stack(pers_amp), axis=0),
            label="persistence",
        )
        ax[ax_index].loglog(
            np.nanmean(np.stack(test_freq), axis=0),
            np.nanmean(np.stack(test_amp), axis=0),
            label="target",
        )
        # if(ax_index > 0):
        #     ax[ax_index].set_yticks([])
        # axes.loglog(logfreq2, logamp2, label='output')
        ax[ax_index].set_title(variables[ax_index], fontsize=29)
        ax[ax_index].grid()
        ax[ax_index].tick_params(axis="both", which="major", labelsize=28)
        ax[ax_index].set_xlabel("k")
        ax[ax_index].set_ylabel("P(k)")

        if ax_index == 6:
            ax[ax_index].legend(fontsize=30)
    fig.suptitle("Power spectral density 1969-1979", y=1, fontsize=32)
    plt.savefig(out_file, bbox_inches="tight")
