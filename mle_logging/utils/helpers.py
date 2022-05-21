import os
import sys
from typing import Any, Union, List, Tuple
import h5py
import yaml
import commentjson
import numpy as np
import pandas as pd
import re
from dotmap import DotMap
import collections

if sys.version_info < (3, 8):
    # Load with pickle5 for python version compatibility
    import pickle5 as pickle
else:
    import pickle


def save_pkl_object(obj: Any, filename: str) -> None:
    """Store objects as pickle files.

    Args:
        obj (Any): Object to pickle.
        filename (str): File path to store object in.
    """
    with open(filename, "wb") as output:
        # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_pkl_object(filename: str) -> Any:
    """Reload pickle objects from path.

    Args:
        filename (str): File path to load object from.

    Returns:
        Any: Reloaded object.
    """
    with open(filename, "rb") as input:
        obj = pickle.load(input)
    return obj


def load_config(
    config_fname: str, return_dotmap: bool = False
) -> Union[dict, DotMap]:
    """Load JSON/YAML config depending on file ending.

    Args:
        config_fname (str):
            File path to YAML/JSON configuration file.
        return_dotmap (bool, optional):
            Option to return dot indexable dictionary. Defaults to False.

    Raises:
        ValueError: Only YAML/JSON files can be loaded.

    Returns:
        Union[dict, DotMap]: Loaded dictionary from file.
    """
    fname, fext = os.path.splitext(config_fname)
    if fext == ".yaml":
        config = load_yaml_config(config_fname, return_dotmap)
    elif fext == ".json":
        config = load_json_config(config_fname, return_dotmap)
    else:
        raise ValueError("Only YAML & JSON configuration can be loaded.")
    return config


def load_yaml_config(
    config_fname: str, return_dotmap: bool = False
) -> Union[dict, DotMap]:
    """Load in YAML config file.

    Args:
        config_fname (str):
            File path to YAML configuration file.
        return_dotmap (bool, optional):
            Option to return dot indexable dictionary. Defaults to False.

    Returns:
        Union[dict, DotMap]: Loaded dictionary from YAML file.
    """
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )
    with open(config_fname) as file:
        yaml_config = yaml.load(file, Loader=loader)
    if not return_dotmap:
        return yaml_config
    else:
        return DotMap(yaml_config)


def load_json_config(
    config_fname: str, return_dotmap: bool = False
) -> Union[dict, DotMap]:
    """Load in JSON config file.

    Args:
        config_fname (str):
            File path to JSON configuration file.
        return_dotmap (bool, optional):
            Option to return dot indexable dictionary. Defaults to False.

    Returns:
        Union[dict, DotMap]: Loaded dictionary from JSON file.
    """
    json_config = commentjson.loads(open(config_fname, "r").read())
    if not return_dotmap:
        return json_config
    else:
        return DotMap(json_config)


def write_to_hdf5(
    log_fname: str, log_path: str, data_to_log: Any, dtype: str = "S5000"
) -> None:
    """Writes data to an hdf5 file and specified log path within.

    Args:
        log_fname (str): Path of hdf5 file.
        log_path (str): Path within hdf5 file to store data at.
        data_to_log (Any): Data (array, list, etc.) to store at `log_path`
        dtype (str, optional): Data type to store as. Defaults to "S5000".
    """
    # Store figure paths if any where created
    if dtype == "S5000":
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


def moving_smooth_ts(
    ts, window_size: int = 20
) -> Tuple[pd.core.series.Series, pd.core.series.Series]:
    """Smoothes a time series using a moving average filter.

    Args:
        ts:
            Time series to smooth.
        window_size (int, optional):
            Window size to apply for moving average. Defaults to 20.

    Returns:
        Tuple[pd.core.series.Series, pd.core.series.Series]:
            Smoothed mean and standard deviation of time series.
    """
    smooth_df = pd.DataFrame(ts)
    mean_ts = smooth_df[0].rolling(window_size, min_periods=1).mean()
    std_ts = smooth_df[0].rolling(window_size, min_periods=1).std()
    return mean_ts, std_ts


def visualize_1D_lcurves(  # noqa: C901
    main_log: dict,
    iter_to_plot: str = "num_updates",
    target_to_plot: Union[List[str], str] = "loss",
    smooth_window: int = 1,
    plot_title: Union[str, None] = None,
    xy_labels: Union[List[str], None] = None,
    base_label: str = "{}",
    curve_labels: list = [],
    every_nth_tick: Union[int, None] = None,
    plot_std_bar: bool = False,
    run_ids: Union[None, List[str]] = None,
    rgb_tuples: Union[List[tuple], None] = None,
    num_legend_cols: Union[int, None] = 1,
    fig=None,
    ax=None,
    figsize: tuple = (9, 6),
    plot_labels: bool = True,
    legend_title: Union[None, str] = None,
    ax_lims: Union[None, list] = None,
) -> tuple:
    """Plot stats curves over time from meta_log. Select data and customize plot.

    Args:
        iter_to_plot (str, optional):
            Time variable to plot in log `time`. Defaults to "num_updates".
        target_to_plot (Union[List[str], str], optional):
            Stats variable to plot in log `stats`. Defaults to "loss".
        smooth_window (int, optional):
            Time series moving average smoothing window. Defaults to 1.
        plot_title (Union[str, None], optional):
            Title for plot. Defaults to None.
        xy_labels (Union[List[str], None], optional):
            List of x & y plot labels. Defaults to None.
        base_label (str, optional):
            Base start of line labels. Defaults to "{}".
        curve_labels (list, optional):
            Explicit labels for individual lines. Defaults to [].
        every_nth_tick (Union[int, None], optional):
            Only plot every nth tick. Leave others out. Defaults to None.
        plot_std_bar (bool, optional):
            Whether to also plot standard deviation. Defaults to False.
        run_ids (Union[None, List[str]], optional):
            Explicit string id of runs to plot from log. Defaults to None.
        rgb_tuples (Union[List[tuple], None], optional):
            Color tuple to use in color palette. Defaults to None.
        num_legend_cols (Union[int, None], optional):
            Number of columns to split legend in. Defaults to 1.
        fig (Union[matplotlib.figure.Figure, None], optional):
            Matplotlib figure to modify. Defaults to None.
        ax (Union[matplotlib.axes._subplots.AxesSubplot, None], optional):
            Matplotlib axis to modify. Defaults to None.
        figsize (tuple, optional):
            Desired figure size. Defaults to (9, 6).
        plot_labels (bool):
            Whether to plot curve labels
        legend_title (str, optional):
            Title of legend. Defaults to None.
        ax_lims (list, optional):
            Max/min axis range. Defaults to None.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot]:
            Modified matplotlib figure and axis.
    """

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set(
            context="poster",
            style="white",
            palette="Paired",
            font="sans-serif",
            font_scale=1.0,
            color_codes=True,
            rc=None,
        )
    except ImportError:
        raise ImportError(
            "You need to install `matplotlib` & `seaborn` to use `mle-logging`"
            " visualization utilities."
        )

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

    plot_counter = 0
    for i in range(len(run_ids)):
        run_id = run_ids[i]
        for j in range(len(seed_ids)):
            seed_id = seed_ids[j]
            for target in target_to_plot:
                label = curve_labels[plot_counter]
                if (
                    type(log_to_plot[run_id][seed_id].stats[target]) == dict
                    or type(log_to_plot[run_id][seed_id].stats[target])
                    == DotMap
                ):
                    plot_mean = True
                    mean_to_plot = log_to_plot[run_id][seed_id].stats[target][
                        "mean"
                    ]
                    std_to_plot = log_to_plot[run_id][seed_id].stats[target][
                        "std"
                    ]
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

    if len(curve_labels) > 1 and plot_labels:
        if legend_title is None:
            ax.legend(fontsize=7, ncol=num_legend_cols)
        else:
            lg = ax.legend(fontsize=7, ncol=num_legend_cols, title=legend_title)
            title = lg.get_title()
            title.set_fontsize(10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if ax_lims is not None:
        ax.set_ylim(ax_lims)
    if plot_title is None:
        plot_title = ", ".join(target_to_plot)
    ax.set_title(plot_title)
    if xy_labels is None:
        xy_labels = [iter_to_plot, ", ".join(target_to_plot)]
    ax.set_xlabel(xy_labels[0])
    ax.set_ylabel(xy_labels[1])
    fig.tight_layout()
    return fig, ax


def tokenize(filename: str):
    """Helper to sort the log files alphanumerically.

    Args:
        filename (str): Name of run.
    """
    digits = re.compile(r"(\d+)")
    return tuple(
        int(token) if match else token
        for token, match in (
            (fragment, digits.search(fragment))
            for fragment in digits.split(filename)
        )
    )
