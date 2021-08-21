import pickle
import pickle5
from typing import Any, Union, List
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from dotmap import DotMap
import collections

sns.set(
    context="poster",
    style="white",
    palette="Paired",
    font="sans-serif",
    font_scale=1.0,
    color_codes=True,
    rc=None,
)


def save_pkl_object(obj, filename: str) -> None:
    """Helper to store pickle objects."""
    with open(filename, "wb") as output:
        # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_pkl_object(filename: str) -> Any:
    """Helper to reload pickle objects."""
    with open(filename, "rb") as input:
        obj = pickle5.load(input)
    return obj


def write_to_hdf5(
    log_fname: str, log_path: str, data_to_log: Any, dtype: str = "S200"
) -> None:
    # Store figure paths if any where created
    if dtype == "S200":
        try:
            data_to_store = [t.encode("ascii", "ignore") for t in data_to_log]
        except AttributeError:
            data_to_store = data_to_log
    else:
        data_to_store = np.array(data_to_log)

    h5f = h5py.File(log_fname, "a")
    if h5f.get(log_path):
        del h5f[log_path]
    h5f.create_dataset(
        name=log_path,
        data=data_to_store,
        compression="gzip",
        compression_opts=4,
        dtype=dtype,
    )
    h5f.flush()
    h5f.close()


def moving_smooth_ts(ts, window_size: int = 20):
    """Smoothes a time series using a moving average filter."""
    smooth_df = pd.DataFrame(ts)
    mean_ts = smooth_df[0].rolling(window_size, min_periods=1).mean()
    std_ts = smooth_df[0].rolling(window_size, min_periods=1).std()
    return mean_ts, std_ts


def visualize_1D_lcurves(
    main_log: dict,
    iter_to_plot: str = "num_episodes",
    target_to_plot: Union[List[str], str] = "ep_reward",
    smooth_window: int = 1,
    plot_title: Union[str, None] = None,
    xy_labels: Union[list, None] = None,
    base_label: str = "{}",
    curve_labels: list = [],
    every_nth_tick: Union[int, None] = None,
    plot_std_bar: bool = False,
    run_ids: Union[None, list] = None,
    rgb_tuples: Union[List[tuple], None] = None,
    num_legend_cols: Union[int, None] = 1,
    fig=None,
    ax=None,
    figsize: tuple = (9, 6),
):
    """Plot learning curves from meta_log. Select data and customize plot."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Make robust for list/str target variable name input
    if type(target_to_plot) is str:
        target_to_plot = [target_to_plot]
        multi_target = False
    else:
        multi_target = True

    # If single run - add placeholder key run_id
    if run_ids is None:
        run_ids = ["ph_run"]
        log_to_plot = {"ph_run": main_log}
    else:
        log_to_plot = main_log
        run_ids.sort(key=tokenize)

    # Plot all curves if not subselected
    single_level = collections.Counter(
        log_to_plot[run_ids[0]].keys()
    ) == collections.Counter(["stats", "time", "meta"])

    # If single seed/aggregated - add placeholder key seed_id
    if single_level:
        for run_id in run_ids:
            log_to_plot[run_id] = {"ph_seed": log_to_plot[run_id]}
        seed_ids = ["ph_seed"]
        single_seed = True
    else:
        seed_ids = list(log_to_plot[run_ids[0]].keys())
        single_seed = False

    if len(curve_labels) == 0:
        curve_labels = []
        for r_id in run_ids:
            for s_id in seed_ids:
                for target in target_to_plot:
                    c_label = f"{r_id}"
                    if multi_target:
                        c_label = f"{target}: " + c_label
                    if not single_seed:
                        c_label += f"/{s_id}"
                    curve_labels.append(c_label)

    if rgb_tuples is None:
        # Default colormap is blue to red diverging seaborn palette
        color_by = sns.diverging_palette(
            240, 10, sep=1, n=len(run_ids) * len(seed_ids) * len(target_to_plot)
        )
        # color_by = sns.light_palette("navy", len(run_ids), reverse=False)
    else:
        color_by = rgb_tuples

    """
    1. Single config - single seed = no aggregation possible
    2. Single config - multi seed + aggregated
    3. Single config - multi seed + non-aggregated
    5. Multi config - single seed = no aggregation possible
    5. Multi config - multi seed + aggregated
    6. Multi config - multi seed + non-aggregated
    """
    plot_counter = 0
    for i in range(len(run_ids)):
        run_id = run_ids[i]
        for j in range(len(seed_ids)):
            seed_id = seed_ids[j]
            for target in target_to_plot:
                label = curve_labels[plot_counter]
                if (
                    type(log_to_plot[run_id][seed_id].stats[target]) == dict
                    or type(log_to_plot[run_id][seed_id].stats[target]) == DotMap
                ):
                    plot_mean = True
                    mean_to_plot = log_to_plot[run_id][seed_id].stats[target]["mean"]
                    std_to_plot = log_to_plot[run_id][seed_id].stats[target]["std"]
                    smooth_std, _ = moving_smooth_ts(std_to_plot, smooth_window)
                else:
                    plot_mean = False
                    mean_to_plot = log_to_plot[run_id][seed_id].stats[target]

                # Smooth the curve to plot for a specified window (1 = no smoothing)
                smooth_mean, _ = moving_smooth_ts(mean_to_plot, smooth_window)
                ax.plot(
                    log_to_plot[run_id][seed_id].time[iter_to_plot],
                    smooth_mean,
                    color=color_by[plot_counter],
                    label=base_label.format(label),
                    alpha=0.85,
                )

                if plot_std_bar and plot_mean:
                    ax.fill_between(
                        log_to_plot[run_id][seed_id].time[iter_to_plot],
                        smooth_mean - smooth_std,
                        smooth_mean + smooth_std,
                        color=color_by[plot_counter],
                        alpha=0.25,
                    )
                plot_counter += 1

    full_range_x = log_to_plot[run_id][seed_id].time[iter_to_plot]
    # Either plot every nth time tic or 5 equally spaced ones
    if every_nth_tick is not None:
        ax.set_xticks(full_range_x)
        ax.set_xticklabels([str(int(label)) for label in full_range_x])
        for n, label in enumerate(ax.xaxis.get_ticklabels()):
            if n % every_nth_tick != 0:
                label.set_visible(False)
    else:
        idx = np.round(np.linspace(0, len(full_range_x) - 1, 5)).astype(int)
        range_x = full_range_x[idx]
        ax.set_xticks(range_x)
        ax.set_xticklabels([str(int(label)) for label in range_x])

    if len(run_ids) < 20 and len(curve_labels) > 1:
        ax.legend(fontsize=15, ncol=num_legend_cols)
    # ax.set_ylim(0, 0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if plot_title is None:
        plot_title = ", ".join(target_to_plot)
    ax.set_title(plot_title)
    if xy_labels is None:
        xy_labels = [iter_to_plot, ", ".join(target_to_plot)]
    ax.set_xlabel(xy_labels[0])
    ax.set_ylabel(xy_labels[1])
    fig.tight_layout()
    return fig, ax


def tokenize(filename):
    """Helper to sort the log files adequately."""
    digits = re.compile(r"(\d+)")
    return tuple(
        int(token) if match else token
        for token, match in (
            (fragment, digits.search(fragment)) for fragment in digits.split(filename)
        )
    )
