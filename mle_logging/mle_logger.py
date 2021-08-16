import numpy as np
import pandas as pd
import os
import shutil
import time
import datetime
import h5py
from typing import Union, List, Dict
from .comms import print_welcome, print_startup, print_update
from .save import ModelLog, FigureLog, ExtraLog


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
        self.seed_id = seed_id

        # Set when to log and when to print
        self.log_every_j_steps = log_every_j_steps
        self.print_every_k_updates = print_every_k_updates

        # Set up the logging directories - save the timestamped config file
        self.setup_experiment_dir(
            experiment_dir,
            config_fname,
            seed_id,
            use_tboard,
            overwrite_experiment_dir,
        )

        # MODEL LOGGING SETUP: Type of model/every k-th ckpt/top k ckpt
        self.model_log = ModelLog(self.experiment_dir,
                                  self.base_str,
                                  self.seed_id,
                                  model_type,
                                  ckpt_time_to_track,
                                  save_every_k_ckpt,
                                  save_top_k_ckpt,
                                  top_k_metric_name,
                                  top_k_minimize_metric)

        # FIGURE & EXTRA LOGGING SETUP
        self.figure_log = FigureLog(self.experiment_dir, self.seed_id)
        self.extra_log = ExtraLog(self.experiment_dir, self.seed_id)

        # Initialize pd dataframes to store logging stats/times
        self.time_to_track = ["time"] + time_to_track + ["time_elapsed"]
        self.what_to_track = what_to_track
        self.clock_to_track = pd.DataFrame(columns=self.time_to_track)
        self.stats_to_track = pd.DataFrame(columns=self.what_to_track)

        # Set up what to print
        if time_to_print is not None:
            self.time_to_print = ["time"] + time_to_print
        else:
            self.time_to_print = []
        self.what_to_print = what_to_print

        if self.what_to_print is None:
            self.verbose = False
        else:
            self.verbose = len(self.what_to_print) > 0
            print_welcome()
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
                print_update(self.time_to_print, self.what_to_print, c_tick, s_tick)

        # Save the log if boolean says so
        if save:
            self.save()

        # Save the most recent model checkpoint
        if model is not None:
            self.save_model(model)
        # Save fig from matplotlib
        if plot_fig is not None:
            self.save_plot(plot_fig)
        # Save .pkl object
        if extra_obj is not None:
            self.save_extra(extra_obj)

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
            if self.model_log.model_type == "torch":
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
            elif self.model_log.model_type == "jax":
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
                data=[self.model_log.final_model_save_fname.encode("ascii", "ignore")],
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
                data=[self.model_log.model_type.encode("ascii", "ignore")],
                compression="gzip",
                compression_opts=4,
                dtype="S200",
            )

            if self.model_log.save_top_k_ckpt or self.model_log.save_every_k_ckpt:
                h5f.create_dataset(
                    name=self.seed_id + "/meta/ckpt_time_to_track",
                    data=[self.model_log.ckpt_time_to_track.encode("ascii", "ignore")],
                    compression="gzip",
                    compression_opts=4,
                    dtype="S200",
                )

            if self.model_log.save_top_k_ckpt:
                h5f.create_dataset(
                    name=self.seed_id + "/meta/top_k_metric_name",
                    data=[self.model_log.top_k_metric_name.encode("ascii", "ignore")],
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
        if self.model_log.save_every_k_ckpt is not None:
            if self.model_log.model_save_counter >= 1:
                for o_name in ["every_k_storage_time", "every_k_ckpt_list"]:
                    if h5f.get(self.seed_id + "/meta/" + o_name):
                        del h5f[self.seed_id + "/meta/" + o_name]
            h5f.create_dataset(
                name=self.seed_id + "/meta/every_k_storage_time",
                data=np.array(self.model_log.every_k_storage_time),
                compression="gzip",
                compression_opts=4,
                dtype="float32",
            )
            h5f.create_dataset(
                name=self.seed_id + "/meta/every_k_ckpt_list",
                data=[t.encode("ascii", "ignore") for t
                      in self.model_log.every_k_ckpt_list],
                compression="gzip",
                compression_opts=4,
                dtype="S200",
            )

        #  Store data on stored checkpoints - stored top k ckpt
        if self.model_log.save_top_k_ckpt is not None:
            if self.model_log.model_save_counter >= 1:
                for o_name in [
                    "top_k_storage_time",
                    "top_k_ckpt_list",
                    "top_k_performance",
                ]:
                    if h5f.get(self.seed_id + "/meta/" + o_name):
                        del h5f[self.seed_id + "/meta/" + o_name]
            h5f.create_dataset(
                name=self.seed_id + "/meta/top_k_storage_time",
                data=np.array(self.model_log.top_k_storage_time),
                compression="gzip",
                compression_opts=4,
                dtype="float32",
            )
            h5f.create_dataset(
                name=self.seed_id + "/meta/top_k_ckpt_list",
                data=[t.encode("ascii", "ignore") for t
                      in self.model_log.top_k_ckpt_list],
                compression="gzip",
                compression_opts=4,
                dtype="S200",
            )
            h5f.create_dataset(
                name=self.seed_id + "/meta/top_k_performance",
                data=np.array(self.model_log.top_k_performance),
                compression="gzip",
                compression_opts=4,
                dtype="float32",
            )

        h5f.flush()
        h5f.close()

        # Tick the log save counter
        self.log_save_counter += 1

    def save_model(self, model):
        """ Save a model checkpoint. """
        self.model_log.save(model, self.clock_to_track, self.stats_to_track)

    def save_plot(self, fig, fig_fname: Union[str, None] = None):
        """Store a figure in a experiment_id/figures directory."""
        self.figure_log.save(fig, fig_fname)
        write_to_hdf5_log(self.log_save_fname,
                          self.seed_id + "/meta/fig_storage_paths",
                          self.figure_log.fig_storage_paths)

    def save_extra(self, obj, obj_fname: Union[str, None] = None):
        """Helper fct. to save object (dict/etc.) as .pkl in exp. subdir."""
        self.extra_log.save(obj, obj_fname)
        write_to_hdf5_log(self.log_save_fname,
                          self.seed_id + "/meta/extra_storage_paths",
                          self.extra_log.extra_storage_paths)


def write_to_hdf5_log(log_fname: str, log_path: str, data_to_log):
    # Store figure paths if any where created
    h5f = h5py.File(log_fname, "a")
    if h5f.get(log_path):
        del h5f[log_path]
    h5f.create_dataset(
        name=log_path,
        data=[t.encode("ascii", "ignore") for t
              in data_to_log],
        compression="gzip",
        compression_opts=4,
        dtype="S200",
    )
    h5f.flush()
    h5f.close()
