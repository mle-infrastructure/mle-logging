import numpy as np
import pandas as pd
import os
import shutil
import time
import datetime
import h5py
from rich.console import Console
from rich.table import Table
from rich import box
from typing import Union, List, Dict
from .utils import save_pkl_object, print_startup


class MLELogger(object):
    """
    Logging object for Machine Learning experiments

    Args:
        ======= TRACKING AND PRINTING VARIABLE NAMES
        time_to_track (List[str]): column names of pandas df - time
        what_to_track (List[str]): column names of pandas df - statistics
        time_to_print (List[str]): columns of time df to print out
        what_to_print (List[str]): columns of stats df to print out
        ======= TRACKING AND PRINTING VARIABLE NAMES
        config_fname (str): file path of configuration of experiment
        experiment_dir (str): base experiment directory
        seed_id (str): seed id to distinguish logs with (e.g. seed_0)
        overwrite_experiment_dir (bool): delete old log file/tboard dir
        ======= VERBOSITY/TBOARD LOGGING
        use_tboard (bool): whether to log to tensorboard
        log_every_j_steps (int): steps between log updates
        print_every_k_updates (int): after how many log updates - verbose
        ======= MODEL STORAGE
        model_type (str): ["torch", "jax", "sklearn", "numpy"]
        ckpt_time_to_track (str): Variable name/score key to save
        save_every_k_ckpt (int): save every other checkpoint
        save_top_k_ckpt (int): save top k performing checkpoints
        top_k_metric_name (str): Variable name/score key to save
        top_k_minimize_metric (str): Boolean for min/max score in top k logging
    """

    def __init__(
        self,
        time_to_track: List[str],
        what_to_track: List[str],
        time_to_print: Union[List[str], None] = None,
        what_to_print: Union[List[str], None] = None,
        config_fname: Union[str, None] = None,
        experiment_dir: str = "/",
        seed_id: str = "no_seed_provided",
        overwrite_experiment_dir: bool = False,
        use_tboard: bool = False,
        log_every_j_steps: Union[int, None] = None,
        print_every_k_updates: Union[int, None] = None,
        model_type: str = "no-model-type-provided",
        ckpt_time_to_track: Union[str, None] = None,
        save_every_k_ckpt: Union[int, None] = None,
        save_top_k_ckpt: Union[int, None] = None,
        top_k_metric_name: Union[str, None] = None,
        top_k_minimize_metric: Union[bool, None] = None,
    ):
        # Initialize counters of log - log, model, figures
        self.log_update_counter = 0
        self.log_save_counter = 0
        self.model_save_counter = 0
        self.extra_save_counter = 0
        self.extra_storage_paths: List[str] = []
        self.fig_save_counter = 0
        self.fig_storage_paths: List[str] = []

        self.log_every_j_steps = log_every_j_steps
        self.print_every_k_updates = print_every_k_updates

        # MODEL LOGGING SETUP: Type of model/every k-th ckpt/top k ckpt
        assert model_type in ["torch", "tensorflow", "jax", "sklearn", "numpy"]
        self.model_type = model_type
        self.ckpt_time_to_track = ckpt_time_to_track
        self.save_every_k_ckpt = save_every_k_ckpt
        self.save_top_k_ckpt = save_top_k_ckpt
        self.top_k_metric_name = top_k_metric_name
        self.top_k_minimize_metric = top_k_minimize_metric

        # Initialize lists for top k scores and to track storage times
        if self.save_every_k_ckpt is not None:
            self.every_k_storage_time: List[int] = []
        if self.save_top_k_ckpt is not None:
            self.top_k_performance: List[float] = []
            self.top_k_storage_time: List[int] = []

        # Set up the logging directories - save the timestamped config file
        self.setup_experiment_dir(
            experiment_dir,
            config_fname,
            seed_id,
            use_tboard,
            overwrite_experiment_dir,
        )

        # Initialize pd dataframes to store logging stats/times
        self.time_to_track = ["time"] + time_to_track + ["time_elapsed"]
        self.what_to_track = what_to_track
        self.clock_to_track = pd.DataFrame(columns=self.time_to_track)
        self.stats_to_track = pd.DataFrame(columns=self.what_to_track)

        # Set up what to print
        if time_to_print is not None:
            self.time_to_print = ["time"] + time_to_print
        else:
            self.time_to_print = None
        self.what_to_print = what_to_print

        if self.what_to_print is None:
            self.verbose = False
        else:
            self.verbose = len(self.what_to_print) > 0
            print_startup(
                self.experiment_dir,
                self.time_to_track,
                self.what_to_track,
                model_type,
                ckpt_time_to_track,
                save_every_k_ckpt,
                save_top_k_ckpt,
                top_k_metric_name,
                top_k_minimize_metric,
            )

        # Keep the seed id around
        self.seed_id = seed_id

        # Start stop-watch/clock of experiment
        self.start_time = time.time()

    def extend_tracking(self, add_track_vars: List[str]) -> None:
        """Add string names of variables to track."""
        assert self.log_update_counter == 0
        self.what_to_track += add_track_vars
        self.stats_to_track = pd.DataFrame(columns=self.what_to_track)

    def setup_experiment_dir(  # noqa: C901
        self,
        base_exp_dir: str,
        config_fname: Union[str, None],
        seed_id: str,
        use_tboard: bool = False,
        overwrite_experiment_dir: bool = False,
    ) -> None:
        """Setup a directory for experiment & copy over config."""
        # Get timestamp of experiment & create new directories
        timestr = datetime.datetime.today().strftime("%Y-%m-%d")[2:]
        if config_fname is not None:
            self.base_str = "_" + os.path.split(config_fname)[1].split(".")[0]
            self.experiment_dir = os.path.join(
                base_exp_dir, timestr + self.base_str + "/"
            )
        else:
            self.base_str = ""
            self.experiment_dir = base_exp_dir

        # Create a new empty directory for the experiment
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "logs/"), exist_ok=True)

        exp_time_base = self.experiment_dir + timestr + self.base_str
        config_copy = exp_time_base + ".json"
        if not os.path.exists(config_copy) and config_fname is not None:
            shutil.copy(config_fname, config_copy)
            self.config_copy = config_copy
        else:
            self.config_copy = "config-json-not-provided"

        # Set where to log to (Stats - .hdf5, model - .ckpth)
        self.log_save_fname = (
            self.experiment_dir
            + "logs/"
            + timestr
            + self.base_str
            + "_"
            + seed_id
            + ".hdf5"
        )

        # Create separate filenames for checkpoints & final trained model
        self.final_model_save_fname = (
            self.experiment_dir
            + "models/final/"
            + timestr
            + self.base_str
            + "_"
            + seed_id
        )
        if self.save_every_k_ckpt is not None:
            self.every_k_ckpt_list: List[str] = []
            self.every_k_model_save_fname = (
                self.experiment_dir
                + "models/every_k/"
                + timestr
                + self.base_str
                + "_"
                + seed_id
                + "_k_"
            )
        if self.save_top_k_ckpt is not None:
            self.top_k_ckpt_list: List[str] = []
            self.top_k_model_save_fname = (
                self.experiment_dir
                + "models/top_k/"
                + timestr
                + self.base_str
                + "_"
                + seed_id
                + "_top_"
            )

        # Different extensions to model checkpoints based on model type
        if self.model_type in ["torch", "tensorflow"]:
            self.final_model_save_fname += ".pt"
        elif self.model_type in ["jax", "sklearn"]:
            self.final_model_save_fname += ".pkl"
        elif self.model_type == "numpy":
            self.final_model_save_fname += ".npy"

        # Delete old log file and tboard dir if overwrite allowed
        if overwrite_experiment_dir:
            if os.path.exists(self.log_save_fname):
                os.remove(self.log_save_fname)
            if use_tboard:
                if os.path.exists(self.experiment_dir + "tboards/"):
                    shutil.rmtree(self.experiment_dir + "tboards/")

        # Initialize tensorboard logger/summary writer
        if use_tboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ModuleNotFoundError as err:
                raise ModuleNotFoundError(
                    f"{err}. You need to install "
                    "`torch` if you want that "
                    "MLELogger logs to tensorboard."
                )
            self.writer = SummaryWriter(
                self.experiment_dir
                + "tboards/"
                + timestr
                + self.base_str
                + "_"
                + seed_id
            )
        else:
            self.writer = None

    def update(  # noqa: C901
        self,
        clock_tick: Dict[str, int],
        stats_tick: Dict[str, float],
        model=None,
        plot_fig=None,
        extra_obj=None,
        save=False,
    ):
        """Update with the newest tick of performance stats, net weights"""
        # Check all keys do exist in data dicts to log [exclude time_elapsed]
        for k in self.time_to_track[1:-1]:
            assert k in clock_tick.keys(), f"{k} not in clock_tick keys."
        for k in self.stats_to_track:
            assert k in stats_tick.keys(), f"{k} not in stats_tick keys."

        # Transform clock_tick, stats_tick lists into pd arrays
        timestr = datetime.datetime.today().strftime("%m-%d|%H:%M:%S")
        c_tick = pd.DataFrame(columns=self.time_to_track)
        c_tick.loc[0] = (
            [timestr]
            + [clock_tick[k] for k in self.time_to_track[1:-1]]
            + [time.time()]
        )
        s_tick = pd.DataFrame(columns=self.what_to_track)
        s_tick.loc[0] = [stats_tick[k] for k in self.stats_to_track]

        # Append time tick & results to pandas dataframes
        self.clock_to_track = pd.concat([self.clock_to_track, c_tick], axis=0)
        self.stats_to_track = pd.concat([self.stats_to_track, s_tick], axis=0)

        # Tick up the update counter
        self.log_update_counter += 1

        # Update the tensorboard log with the newest event
        if self.writer is not None:
            self.update_tboard(clock_tick, stats_tick, model, plot_fig)

        # Print the most current results
        if self.verbose and self.print_every_k_updates is not None:
            if self.log_update_counter % self.print_every_k_updates == 0:
                console = Console()
                table = Table(
                    show_header=True,
                    row_styles=["none"],
                    border_style="red",
                    box=box.SIMPLE,
                )
                for i, c_label in enumerate(self.time_to_print):
                    if i == 0:
                        table.add_column(
                            ":watch: [red]" + c_label + "[/red]",
                            style="red",
                            width=14,
                            justify="left",
                        )
                    else:
                        table.add_column(
                            "[red]" + c_label + "[/red]",
                            style="red",
                            width=12,
                            justify="center",
                        )
                for i, c_label in enumerate(self.what_to_print):
                    if i == 0:
                        table.add_column(
                            ":open_book: [blue]" + c_label + "[/blue]",
                            style="blue",
                            width=14,
                            justify="center",
                        )
                    else:
                        table.add_column(
                            "[blue]" + c_label + "[/blue]",
                            style="blue",
                            width=12,
                            justify="center",
                        )

                row_list = pd.concat(
                    [c_tick[self.time_to_print], s_tick[self.what_to_print]], axis=1
                ).values.tolist()[0]
                row_str_list = [str(v) for v in row_list]
                table.add_row(*row_str_list)
                console.print(table, justify="center")

        # Save the log if boolean says so
        if save:
            # Save the most recent model checkpoint
            if model is not None:
                self.save_model(model)
            # Save fig from matplotlib
            if plot_fig is not None:
                self.save_plot(plot_fig)
            # Save .pkl object
            if extra_obj is not None:
                self.save_extra(extra_obj)
            self.save()

    def update_tboard(  # noqa: C901
        self, clock_tick: dict, stats_tick: dict, model=None, plot_to_tboard=None
    ):
        """Update the tensorboard with the newest events"""
        # Set the x-axis time variable to first key provided in time key dict
        time_var_id = clock_tick[self.time_to_track[1]]

        # Add performance & step counters
        for k in self.what_to_track:
            self.writer.add_scalar(
                "performance/" + k, np.mean(stats_tick[k]), time_var_id
            )

        # Log the model params & gradients
        if model is not None:
            if self.model_type == "torch":
                for name, param in model.named_parameters():
                    self.writer.add_histogram(
                        "weights/" + name, param.clone().cpu().data.numpy(), time_var_id
                    )
                    # Try getting gradients from torch model
                    try:
                        self.writer.add_histogram(
                            "gradients/" + name,
                            param.grad.clone().cpu().data.numpy(),
                            time_var_id,
                        )
                    except Exception:
                        continue
            elif self.model_type == "jax":
                # Try to add parameters from nested dict first - then simple
                # TODO: Add gradient tracking for JAX models
                for layer in model.keys():
                    try:
                        for w in model[layer].keys():
                            self.writer.add_histogram(
                                "weights/" + layer + "/" + w,
                                np.array(model[layer][w]),
                                time_var_id,
                            )
                    except Exception:
                        self.writer.add_histogram(
                            "weights/" + layer, np.array(model[layer]), time_var_id
                        )

        # Add the plot of interest to tboard
        if plot_to_tboard is not None:
            self.writer.add_figure("plot", plot_to_tboard, time_var_id)

        # Flush the log event
        self.writer.flush()

    def save(self):  # noqa: C901
        """Create compressed .hdf5 file containing group <random-seed-id>"""
        h5f = h5py.File(self.log_save_fname, "a")

        # Create "datasets" to store in the hdf5 file [time, stats]
        # Store all relevant meta data (log filename, checkpoint filename)
        if self.log_save_counter == 0:
            h5f.create_dataset(
                name=self.seed_id + "/meta/model_ckpt",
                data=[self.final_model_save_fname.encode("ascii", "ignore")],
                compression="gzip",
                compression_opts=4,
                dtype="S200",
            )
            h5f.create_dataset(
                name=self.seed_id + "/meta/log_paths",
                data=[self.log_save_fname.encode("ascii", "ignore")],
                compression="gzip",
                compression_opts=4,
                dtype="S200",
            )
            h5f.create_dataset(
                name=self.seed_id + "/meta/experiment_dir",
                data=[self.experiment_dir.encode("ascii", "ignore")],
                compression="gzip",
                compression_opts=4,
                dtype="S200",
            )
            h5f.create_dataset(
                name=self.seed_id + "/meta/config_fname",
                data=[self.config_copy.encode("ascii", "ignore")],
                compression="gzip",
                compression_opts=4,
                dtype="S200",
            )
            h5f.create_dataset(
                name=self.seed_id + "/meta/eval_id",
                data=[self.base_str.encode("ascii", "ignore")],
                compression="gzip",
                compression_opts=4,
                dtype="S200",
            )
            h5f.create_dataset(
                name=self.seed_id + "/meta/model_type",
                data=[self.model_type.encode("ascii", "ignore")],
                compression="gzip",
                compression_opts=4,
                dtype="S200",
            )

            if self.save_top_k_ckpt or self.save_every_k_ckpt:
                h5f.create_dataset(
                    name=self.seed_id + "/meta/ckpt_time_to_track",
                    data=[self.ckpt_time_to_track.encode("ascii", "ignore")],
                    compression="gzip",
                    compression_opts=4,
                    dtype="S200",
                )

            if self.save_top_k_ckpt:
                h5f.create_dataset(
                    name=self.seed_id + "/meta/top_k_metric_name",
                    data=[self.top_k_metric_name.encode("ascii", "ignore")],
                    compression="gzip",
                    compression_opts=4,
                    dtype="S200",
                )

        # Store all time_to_track variables
        for o_name in self.time_to_track:
            if self.log_save_counter >= 1:
                if h5f.get(self.seed_id + "/time/" + o_name):
                    del h5f[self.seed_id + "/time/" + o_name]
            if o_name != "time":
                h5f.create_dataset(
                    name=self.seed_id + "/time/" + o_name,
                    data=self.clock_to_track[o_name],
                    compression="gzip",
                    compression_opts=4,
                    dtype="float32",
                )
            else:
                h5f.create_dataset(
                    name=self.seed_id + "/time/" + o_name,
                    data=[
                        t.encode("ascii", "ignore")
                        for t in self.clock_to_track[o_name].values.tolist()
                    ],
                    compression="gzip",
                    compression_opts=4,
                    dtype="S200",
                )

        # Store all what_to_track variables
        for o_name in self.what_to_track:
            if self.log_save_counter >= 1:
                if h5f.get(self.seed_id + "/stats/" + o_name):
                    del h5f[self.seed_id + "/stats/" + o_name]
            data_to_store = self.stats_to_track[o_name].to_numpy()
            if type(data_to_store[0]) == np.ndarray:
                data_to_store = np.stack(data_to_store)
            if type(data_to_store[0]) in [np.str_, str]:
                data_to_store = [t.encode("ascii", "ignore") for t in data_to_store]
            if type(data_to_store[0]) in [bytes, np.str_]:
                data_type = np.dtype("S200")
            elif type(data_to_store[0]) == int:
                data_type = np.dtype("int32")
            else:
                data_type = np.dtype("float32")
            h5f.create_dataset(
                name=self.seed_id + "/stats/" + o_name,
                data=np.array(data_to_store).astype(data_type),
                compression="gzip",
                compression_opts=4,
                dtype=data_type,
            )

        # Store data on stored checkpoints - stored every k updates
        if self.save_every_k_ckpt is not None:
            if self.log_save_counter >= 1:
                for o_name in ["every_k_storage_time", "every_k_ckpt_list"]:
                    if h5f.get(self.seed_id + "/meta/" + o_name):
                        del h5f[self.seed_id + "/meta/" + o_name]
            h5f.create_dataset(
                name=self.seed_id + "/meta/every_k_storage_time",
                data=np.array(self.every_k_storage_time),
                compression="gzip",
                compression_opts=4,
                dtype="float32",
            )
            h5f.create_dataset(
                name=self.seed_id + "/meta/every_k_ckpt_list",
                data=[t.encode("ascii", "ignore") for t in self.every_k_ckpt_list],
                compression="gzip",
                compression_opts=4,
                dtype="S200",
            )

        #  Store data on stored checkpoints - stored top k ckpt
        if self.save_top_k_ckpt is not None:
            if self.log_save_counter >= 1:
                for o_name in [
                    "top_k_storage_time",
                    "top_k_ckpt_list",
                    "top_k_performance",
                ]:
                    if h5f.get(self.seed_id + "/meta/" + o_name):
                        del h5f[self.seed_id + "/meta/" + o_name]
            h5f.create_dataset(
                name=self.seed_id + "/meta/top_k_storage_time",
                data=np.array(self.top_k_storage_time),
                compression="gzip",
                compression_opts=4,
                dtype="float32",
            )
            h5f.create_dataset(
                name=self.seed_id + "/meta/top_k_ckpt_list",
                data=[t.encode("ascii", "ignore") for t in self.top_k_ckpt_list],
                compression="gzip",
                compression_opts=4,
                dtype="S200",
            )
            h5f.create_dataset(
                name=self.seed_id + "/meta/top_k_performance",
                data=np.array(self.top_k_performance),
                compression="gzip",
                compression_opts=4,
                dtype="float32",
            )

        h5f.flush()
        h5f.close()

        # Tick the log save counter
        self.log_save_counter += 1

    def setup_model_ckpt_dir(self):
        """Create separate sub-dirs for checkpoints & final trained model."""
        os.makedirs(os.path.join(self.experiment_dir, "models/final/"), exist_ok=True)
        if self.save_every_k_ckpt is not None:
            os.makedirs(
                os.path.join(self.experiment_dir, "models/every_k/"), exist_ok=True
            )
        if self.save_top_k_ckpt is not None:
            os.makedirs(
                os.path.join(self.experiment_dir, "models/top_k/"), exist_ok=True
            )

    def save_model(self, model):  # noqa: C901
        """Save current state of the model as a checkpoint."""
        # If first model ckpt is saved - generate necessary directories
        self.model_save_counter += 1
        if self.model_save_counter == 1:
            self.setup_model_ckpt_dir()

        # CASE 1: SIMPLE STORAGE OF MOST RECENTLY LOGGED MODEL STATE
        if self.model_type == "torch":
            # Torch model case - save model state dict as .pt checkpoint
            self.save_torch_model(self.final_model_save_fname, model)
        elif self.model_type == "tensorflow":
            model.save_weights(self.final_model_save_fname)
        elif self.model_type in ["jax", "sklearn"]:
            # JAX/sklearn save parameter dict/model as dictionary
            save_pkl_object(model, self.final_model_save_fname)
        elif self.model_type == "numpy":
            np.save(self.final_model_save_fname, model)
        else:
            raise ValueError("Provide valid model_type [torch, jax, sklearn, numpy].")

        # CASE 2: SEPARATE STORAGE OF EVERY K-TH LOGGED MODEL STATE
        if self.save_every_k_ckpt is not None:
            if self.log_save_counter % self.save_every_k_ckpt == 0:
                if self.model_type == "torch":
                    ckpt_path = (
                        self.every_k_model_save_fname
                        + str(self.model_save_counter)
                        + ".pt"
                    )
                    self.save_torch_model(ckpt_path, model)
                elif self.model_type == "tensorflow":
                    ckpt_path = (
                        self.every_k_model_save_fname
                        + str(self.model_save_counter)
                        + ".pt"
                    )
                    model.save_weights(ckpt_path)
                elif self.model_type in ["jax", "sklearn"]:
                    ckpt_path = (
                        self.every_k_model_save_fname
                        + str(self.model_save_counter)
                        + ".pkl"
                    )
                    save_pkl_object(model, ckpt_path)
                elif self.model_type == "numpy":
                    ckpt_path = (
                        self.every_k_model_save_fname
                        + str(self.model_save_counter)
                        + ".npy"
                    )
                    np.save(ckpt_path, model)
                # Update model save count & time point of storage
                # Use latest update performance for last checkpoint
                time = self.clock_to_track[self.ckpt_time_to_track].to_numpy()[-1]
                self.every_k_storage_time.append(time)
                self.every_k_ckpt_list.append(ckpt_path)

        # CASE 3: STORE TOP-K MODEL STATES BY SOME SCORE
        if self.save_top_k_ckpt is not None:
            updated_top_k = False
            # Use latest update performance for last checkpoint
            score = self.stats_to_track[self.top_k_metric_name].to_numpy()[-1]
            time = self.clock_to_track[self.ckpt_time_to_track].to_numpy()[-1]
            # Fill up empty top k slots
            if len(self.top_k_performance) < self.save_top_k_ckpt:
                if self.model_type == "torch":
                    ckpt_path = (
                        self.top_k_model_save_fname
                        + str(len(self.top_k_performance))
                        + ".pt"
                    )
                    self.save_torch_model(ckpt_path, model)
                elif self.model_type == "tensorflow":
                    ckpt_path = (
                        self.top_k_model_save_fname
                        + str(len(self.top_k_performance))
                        + ".pt"
                    )
                    model.save_weights(ckpt_path)
                elif self.model_type in ["jax", "sklearn"]:
                    ckpt_path = (
                        self.top_k_model_save_fname
                        + str(len(self.top_k_performance))
                        + ".pkl"
                    )
                    save_pkl_object(model, ckpt_path)
                elif self.model_type == "numpy":
                    ckpt_path = (
                        self.top_k_model_save_fname
                        + str(len(self.top_k_performance))
                        + ".npy"
                    )
                    np.save(ckpt_path, model)
                updated_top_k = True
                self.top_k_performance.append(score)
                self.top_k_storage_time.append(time)
                self.top_k_ckpt_list.append(ckpt_path)

            # If minimize = replace worst performing model (max score)
            if (
                self.top_k_minimize_metric
                and max(self.top_k_performance) > score
                and not updated_top_k
            ):
                id_to_replace = np.argmax(self.top_k_performance)
                self.top_k_performance[id_to_replace] = score
                self.top_k_storage_time[id_to_replace] = time
                if self.model_type == "torch":
                    ckpt_path = self.top_k_model_save_fname + str(id_to_replace) + ".pt"
                    self.save_torch_model(ckpt_path, model)
                elif self.model_type == "tensorflow":
                    ckpt_path = self.top_k_model_save_fname + str(id_to_replace) + ".pt"
                    model.save_weights(ckpt_path)
                elif self.model_type in ["jax", "sklearn"]:
                    ckpt_path = (
                        self.top_k_model_save_fname + str(id_to_replace) + ".pkl"
                    )
                    save_pkl_object(model, ckpt_path)
                elif self.model_type == "numpy":
                    ckpt_path = (
                        self.top_k_model_save_fname + str(id_to_replace) + ".npy"
                    )
                    np.save(ckpt_path, model)
                updated_top_k = True

            # If minimize = replace worst performing model (max score)
            if (
                not self.top_k_minimize_metric
                and min(self.top_k_performance) > score
                and not updated_top_k
            ):
                id_to_replace = np.argmin(self.top_k_performance)
                self.top_k_performance[id_to_replace] = score
                self.top_k_storage_time[id_to_replace] = self.clock_to_track[
                    self.ckpt_time_to_track
                ].to_numpy()[-1]
                if self.model_type == "torch":
                    ckpt_path = (
                        self.top_k_model_save_fname
                        + "_top_"
                        + str(id_to_replace)
                        + ".pt"
                    )
                    self.save_torch_model(ckpt_path, model)
                elif self.model_type == "tensorflow":
                    ckpt_path = (
                        self.top_k_model_save_fname
                        + "_top_"
                        + str(id_to_replace)
                        + ".pt"
                    )
                    model.save_weights(ckpt_path)
                elif self.model_type in ["jax", "sklearn"]:
                    ckpt_path = (
                        self.top_k_model_save_fname
                        + "_top_"
                        + str(id_to_replace)
                        + ".pkl"
                    )
                    save_pkl_object(model, ckpt_path)
                elif self.model_type in ["numpy"]:
                    ckpt_path = (
                        self.top_k_model_save_fname
                        + "_top_"
                        + str(id_to_replace)
                        + ".npy"
                    )
                    np.save(ckpt_path, model)
                updated_top_k = True

    def save_torch_model(self, path_to_store, model):
        """Store a torch checkpoint for a model."""
        try:
            import torch
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                f"{err}. You need to install "
                "`torch` if you want to save a model "
                "checkpoint."
            )
        # Update the saved weights in a single file!
        torch.save(model.state_dict(), path_to_store)

    def save_plot(self, fig, fig_fname: Union[str, None] = None):
        """Store a figure in a experiment_id/figures directory."""
        # Create new directory to store figures - if it doesn't exist yet
        figures_dir = os.path.join(self.experiment_dir, "figures/")
        if not os.path.exists(figures_dir):
            try:
                os.makedirs(figures_dir)
            except Exception:
                pass

        # Tick up counter, save figure, store new path to figure
        if fig_fname is None:
            self.fig_save_counter += 1
            figure_fname = os.path.join(
                figures_dir,
                "fig_" + str(self.fig_save_counter) + "_" + str(self.seed_id) + ".png",
            )
        else:
            figure_fname = os.path.join(
                figures_dir,
                fig_fname,
            )

        fig.savefig(figure_fname, dpi=300)
        self.fig_storage_paths.append(figure_fname)

        # Store figure paths if any where created
        h5f = h5py.File(self.log_save_fname, "a")
        if h5f.get(self.seed_id + "/meta/fig_storage_paths"):
            del h5f[self.seed_id + "/meta/fig_storage_paths"]
        h5f.create_dataset(
            name=self.seed_id + "/meta/fig_storage_paths",
            data=[t.encode("ascii", "ignore") for t in self.fig_storage_paths],
            compression="gzip",
            compression_opts=4,
            dtype="S200",
        )
        h5f.flush()
        h5f.close()

    def save_extra(self, obj, obj_fname: Union[str, None] = None):
        """Helper fct. to save object (dict/etc.) as .pkl in exp. subdir."""
        extra_dir = os.path.join(self.experiment_dir, "extra/")
        # Create a new empty directory for the experiment
        if not os.path.exists(extra_dir):
            try:
                os.makedirs(extra_dir)
            except Exception:
                pass

        # Tick up counter, save figure, store new path to figure
        if obj_fname is None:
            self.extra_save_counter += 1
            obj_fname = os.path.join(
                extra_dir,
                "extra_"
                + str(self.extra_save_counter)
                + "_"
                + str(self.seed_id)
                + ".pkl",
            )
        else:
            obj_fname = os.path.join(
                extra_dir,
                obj_fname,
            )

        save_pkl_object(obj, obj_fname)
        self.extra_storage_paths.append(obj_fname)

        # Store figure paths if any where created
        h5f = h5py.File(self.log_save_fname, "a")
        if h5f.get(self.seed_id + "/meta/extra_storage_paths"):
            del h5f[self.seed_id + "/meta/extra_storage_paths"]
        h5f.create_dataset(
            name=self.seed_id + "/meta/extra_storage_paths",
            data=[t.encode("ascii", "ignore") for t in self.extra_storage_paths],
            compression="gzip",
            compression_opts=4,
            dtype="S200",
        )
        h5f.flush()
        h5f.close()
