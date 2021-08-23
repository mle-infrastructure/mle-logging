import numpy as np
import os
import shutil
import datetime
from typing import Union, List, Dict
from .utils import write_to_hdf5
from .comms import (print_welcome,
                    print_startup,
                    print_update,
                    print_reload,
                    print_storage)
from .save import StatsLog, TboardLog, ModelLog, FigureLog, ExtraLog


class MLELogger(object):
    """
    Logging object for Machine Learning experiments

    Args:
        ======= TRACKING AND PRINTING VARIABLE NAMES
        time_to_track (List[str]): column names of pandas df - time
        what_to_track (List[str]): column names of pandas df - statistics
        time_to_print (List[str]): subset columns of time df to print out
        what_to_print (List[str]): subset columns of stats df to print out
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
        seed_id: Union[str, int] = "no_seed_provided",
        overwrite: bool = False,
        use_tboard: bool = False,
        log_every_j_steps: Union[int, None] = None,
        print_every_k_updates: Union[int, None] = 1,
        model_type: str = "no-model-type",
        ckpt_time_to_track: Union[str, None] = None,
        save_every_k_ckpt: Union[int, None] = None,
        save_top_k_ckpt: Union[int, None] = None,
        top_k_metric_name: Union[str, None] = None,
        top_k_minimize_metric: Union[bool, None] = None,
        reload: bool = False,
        verbose: bool = False,
    ):
        # Set up tensorboard when/where to log and when to print
        self.use_tboard = use_tboard
        self.log_every_j_steps = log_every_j_steps
        self.print_every_k_updates = print_every_k_updates
        self.timestr = datetime.datetime.today().strftime("%Y-%m-%d")[2:]
        self.log_save_counter = 0
        self.seed_id = "seed_" + str(seed_id) if type(seed_id) == int else seed_id

        # Set up the logging directories - copy timestamped config file
        self.setup_experiment_dir(
            experiment_dir,
            config_fname,
            self.seed_id,
            overwrite,
            reload,
        )
        os.makedirs(os.path.join(self.experiment_dir, "logs/"), exist_ok=True)

        # STATS & TENSORBOARD LOGGING SETUP
        self.stats_log = StatsLog(
            self.experiment_dir,
            self.seed_id,
            time_to_track,
            what_to_track,
            reload,
        )
        if self.use_tboard:
            self.tboard_log = TboardLog(
                self.experiment_dir,
                self.seed_id,
            )

        # MODEL, FIGURE & EXTRA LOGGING SETUP
        self.model_log = ModelLog(
            self.experiment_dir,
            self.seed_id,
            model_type,
            ckpt_time_to_track,
            save_every_k_ckpt,
            save_top_k_ckpt,
            top_k_metric_name,
            top_k_minimize_metric,
            reload,
        )

        self.figure_log = FigureLog(
            self.experiment_dir,
            self.seed_id,
            reload,
        )
        self.extra_log = ExtraLog(
            self.experiment_dir,
            self.seed_id,
            reload,
        )

        # VERBOSITY SETUP: Set up what to print
        self.verbose = verbose
        self.print_counter = 0
        if time_to_print is None:
            self.time_to_print = ["time"] + time_to_track
        else:
            self.time_to_print = ["time"] + time_to_print
        if what_to_print is None:
            self.what_to_print = what_to_track
        else:
            self.what_to_print = what_to_print

        if not reload and verbose:
            print_welcome()
            print_startup(
                self.experiment_dir,
                config_fname,
                time_to_track,
                what_to_track,
                model_type,
                seed_id,
                use_tboard,
                reload,
                print_every_k_updates,
                ckpt_time_to_track,
                save_every_k_ckpt,
                save_top_k_ckpt,
                top_k_metric_name,
                top_k_minimize_metric,
            )
        elif reload and verbose:
            print_reload(self.experiment_dir,)

    def setup_experiment_dir(  # noqa: C901
        self,
        base_exp_dir: str,
        config_fname: Union[str, None],
        seed_id: str,
        overwrite_experiment_dir: bool = False,
        reload: bool = False,
    ) -> None:
        """Setup a directory for experiment & copy over config."""
        # Get timestamp of experiment & create new directories
        if config_fname is not None:
            self.base_str = "_" + os.path.split(config_fname)[1].split(".")[0]
            if not reload:
                self.experiment_dir = os.path.join(
                    base_exp_dir, self.timestr + self.base_str + "/"
                )
            else:
                # Don't redefine experiment directory but get already existing
                exp_dir = [
                    f for f in os.listdir(base_exp_dir) if f.endswith(self.base_str)
                ][0]
                self.experiment_dir = os.path.join(base_exp_dir, exp_dir)
        else:
            self.base_str = ""
            self.experiment_dir = base_exp_dir

        self.log_save_fname = os.path.join(
            self.experiment_dir, "logs/", "log_" + seed_id + ".hdf5"
        )

        # Delete old experiment logging directory
        if overwrite_experiment_dir and not reload:
            if os.path.exists(self.log_save_fname):
                os.remove(self.log_save_fname)
            if self.use_tboard:
                if os.path.exists(os.path.join(self.experiment_dir, "tboards/")):
                    shutil.rmtree(os.path.join(self.experiment_dir, "tboards/"))

        # Create a new empty directory for the experiment (if not existing)
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Copy over json configuration file if it exists
        config_copy = os.path.join(
            self.experiment_dir, self.timestr + self.base_str + ".json"
        )
        if not os.path.exists(config_copy) and config_fname is not None:
            shutil.copy(config_fname, config_copy)
            self.config_copy = config_copy
        else:
            self.config_copy = "config-json-not-provided"

    def update(
        self,
        clock_tick: Dict[str, int],
        stats_tick: Dict[str, float],
        model=None,
        plot_fig=None,
        extra_obj=None,
        save=False,
    ) -> None:
        """Update with the newest tick of performance stats, net weights"""
        # Update the stats log with newest timeseries data
        c_tick, s_tick = self.stats_log.update(clock_tick, stats_tick)
        # Update the tensorboard log with the newest event
        if self.use_tboard:
            self.tboard_log.update(
                self.stats_log.time_to_track, clock_tick, stats_tick, model, plot_fig
            )
        # Save the most recent model checkpoint
        if model is not None:
            self.save_model(model)
        # Save fig from matplotlib
        if plot_fig is not None:
            self.save_plot(plot_fig)
        # Save .pkl object
        if extra_obj is not None:
            self.save_extra(extra_obj)
        # Save the .hdf5 log if boolean says so
        if save:
            self.save()

        # Print the most current results
        if self.verbose and self.print_every_k_updates is not None:
            if self.stats_log.stats_update_counter % self.print_every_k_updates == 0:
                # Only print column name header at 1st print!
                print_update(
                    self.time_to_print,
                    self.what_to_print,
                    c_tick,
                    s_tick,
                    self.print_counter == 0,
                )
                print_storage(
                    fig_path=(self.figure_log.fig_storage_paths[-1]
                              if plot_fig is not None else None),
                    extra_path=(self.extra_log.extra_storage_paths[-1]
                                if extra_obj is not None else None),
                    final_model_path=(self.model_log.final_model_save_fname
                                      if model is not None else None),
                    every_k_model_path=(self.model_log.every_k_ckpt_list[-1]
                                        if model is not None and
                                        self.model_log.stored_every_k else None),
                    top_k_model_path=(self.model_log.top_k_ckpt_list[-1]
                                      if model is not None and
                                      self.model_log.stored_top_k else None),
                )
                self.print_counter += 1

    def save_model(self, model):
        """Save a model checkpoint."""
        self.model_log.save(
            model, self.stats_log.clock_to_track, self.stats_log.stats_to_track
        )

    def save_plot(self, fig, fig_fname: Union[str, None] = None):
        """Store a figure in a experiment_id/figures directory."""
        self.figure_log.save(fig, fig_fname)
        write_to_hdf5(
            self.log_save_fname,
            self.seed_id + "/meta/fig_storage_paths",
            self.figure_log.fig_storage_paths,
        )

    def save_extra(self, obj, obj_fname: Union[str, None] = None):
        """Helper fct. to save object (dict/etc.) as .pkl in exp. subdir."""
        self.extra_log.save(obj, obj_fname)
        write_to_hdf5(
            self.log_save_fname,
            self.seed_id + "/meta/extra_storage_paths",
            self.extra_log.extra_storage_paths,
        )

    def save(self):  # noqa: C901
        """Create compressed .hdf5 file containing group <random-seed-id>"""
        # Create "datasets" to store in the hdf5 file [time, stats]
        # Store all relevant meta data (log filename, checkpoint filename)
        if self.log_save_counter == 0:
            data_paths = [
                self.seed_id + "/meta/model_ckpt",
                self.seed_id + "/meta/log_paths",
                self.seed_id + "/meta/experiment_dir",
                self.seed_id + "/meta/config_fname",
                self.seed_id + "/meta/eval_id",
                self.seed_id + "/meta/model_type",
            ]
            data_to_log = [
                [self.model_log.final_model_save_fname],
                [self.log_save_fname],
                [self.experiment_dir],
                [self.config_copy],
                [self.base_str],
                [self.model_log.model_type],
            ]

            for i in range(len(data_paths)):
                write_to_hdf5(self.log_save_fname, data_paths[i], data_to_log[i])

            if self.model_log.save_top_k_ckpt or self.model_log.save_every_k_ckpt:
                write_to_hdf5(
                    self.log_save_fname,
                    self.seed_id + "/meta/ckpt_time_to_track",
                    [self.model_log.ckpt_time_to_track],
                )

            if self.model_log.save_top_k_ckpt:
                write_to_hdf5(
                    self.log_save_fname,
                    self.seed_id + "/meta/top_k_metric_name",
                    [self.model_log.top_k_metric_name],
                )

        # Store all time_to_track variables
        for o_name in self.stats_log.time_to_track:
            if o_name != "time":
                write_to_hdf5(
                    self.log_save_fname,
                    self.seed_id + "/time/" + o_name,
                    self.stats_log.clock_to_track[o_name].values.tolist(),
                    dtype="float32",
                )
            else:
                write_to_hdf5(
                    self.log_save_fname,
                    self.seed_id + "/time/" + o_name,
                    self.stats_log.clock_to_track[o_name].values.tolist(),
                )

        # Store all what_to_track variables
        for o_name in self.stats_log.what_to_track:
            data_to_store = self.stats_log.stats_to_track[o_name].to_numpy()
            if type(data_to_store[0]) == np.ndarray:
                data_to_store = np.stack(data_to_store)
                dtype = np.dtype("float32")
            if type(data_to_store[0]) in [np.str_, str]:
                dtype = "S200"
            if type(data_to_store[0]) in [bytes, np.str_]:
                dtype = np.dtype("S200")
            elif type(data_to_store[0]) == int:
                dtype = np.dtype("int32")
            else:
                dtype = np.dtype("float32")
            write_to_hdf5(
                self.log_save_fname,
                self.seed_id + "/stats/" + o_name,
                data_to_store,
                dtype,
            )

        # Store data on stored checkpoints - stored every k updates
        if self.model_log.save_every_k_ckpt is not None:
            data_paths = [
                self.seed_id + "/meta/" + "every_k_storage_time",
                self.seed_id + "/meta/" + "every_k_ckpt_list",
            ]
            data_to_log = [
                self.model_log.every_k_storage_time,
                self.model_log.every_k_ckpt_list,
            ]
            data_types = ["int32", "S200"]
            for i in range(len(data_paths)):
                write_to_hdf5(
                    self.log_save_fname, data_paths[i], data_to_log[i], data_types[i]
                )

        #  Store data on stored checkpoints - stored top k ckpt
        if self.model_log.save_top_k_ckpt is not None:
            data_paths = [
                self.seed_id + "/meta/" + "top_k_storage_time",
                self.seed_id + "/meta/" + "top_k_ckpt_list",
                self.seed_id + "/meta/" + "top_k_performance",
            ]
            data_to_log = [
                self.model_log.top_k_storage_time,
                self.model_log.top_k_ckpt_list,
                self.model_log.top_k_performance,
            ]
            data_types = ["int32", "S200", "float32"]
            for i in range(len(data_paths)):
                write_to_hdf5(
                    self.log_save_fname, data_paths[i], data_to_log[i], data_types[i]
                )

        # Tick the log save counter
        self.log_save_counter += 1

    def extend_tracking(self, add_track_vars: List[str]) -> None:
        """Add string names of variables to track."""
        self.stats_log.extend_tracking(add_track_vars)

    def ready_to_log(self, update_counter: int) -> bool:
        """Check whether update_counter is modulo of log_every_k_steps."""
        assert (
            self.log_every_j_steps is not None
        ), "Provide `log_every_j_steps` in your `log_config`"
        return update_counter % self.log_every_j_steps == 0 or update_counter == 0
