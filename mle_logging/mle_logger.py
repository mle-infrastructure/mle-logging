import numpy as np
import os
import shutil
import yaml
from typing import Optional, Union, List, Dict
from rich.console import Console
from .utils import (
    write_to_hdf5,
    load_config,
    print_welcome,
    print_startup,
    print_update,
    print_reload,
    print_storage,
)
from .save import StatsLog, TboardLog, WandbLog, ModelLog, FigureLog, ExtraLog


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
        config_dict(dict): dictionary of experiment config to store in yaml
        experiment_dir (str): base experiment directory
        seed_id (str): seed id to distinguish logs with (e.g. seed_0)
        overwrite (bool): delete old log file/tboard dir
        ======= VERBOSITY/TBOARD LOGGING
        use_tboard (bool): whether to log to tensorboard
        use_wandb (bool): whether to log to wandb
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
        experiment_dir: str = "/",
        time_to_track: List[str] = [],
        what_to_track: List[str] = [],
        time_to_print: Optional[List[str]] = None,
        what_to_print: Optional[List[str]] = None,
        config_fname: Optional[str] = None,
        config_dict: Optional[dict] = None,
        seed_id: Union[str, int] = "no_seed_provided",
        overwrite: bool = False,
        use_tboard: bool = False,
        use_wandb: bool = False,
        wandb_config: Optional[dict] = None,
        log_every_j_steps: Optional[int] = None,
        print_every_k_updates: Optional[int] = 1,
        model_type: str = "no-model-type",
        ckpt_time_to_track: Optional[str] = None,
        save_every_k_ckpt: Optional[int] = None,
        save_top_k_ckpt: Optional[int] = None,
        top_k_metric_name: Optional[str] = None,
        top_k_minimize_metric: Optional[bool] = None,
        reload: bool = False,
        verbose: bool = False,
    ):
        # Set os hdf file to non locking mode
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        # Set up tensorboard when/where to log and when to print
        self.use_tboard = use_tboard
        self.use_wandb = use_wandb
        self.log_every_j_steps = log_every_j_steps
        self.print_every_k_updates = print_every_k_updates
        self.log_save_counter = reload
        self.log_setup_counter = reload
        self.seed_id = (
            "seed_" + str(seed_id) if type(seed_id) == int else seed_id
        )
        self.config_fname = config_fname
        self.config_dict = config_dict

        self.get_configs_ready(self.config_fname, self.config_dict)

        # Set up the logging directories - copy timestamped config file
        self.setup_experiment(
            experiment_dir,
            config_fname,
            self.seed_id,
            overwrite,
            reload,
        )

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
        if self.use_wandb:
            self.wandb_log = WandbLog(
                self.config_dict, self.config_fname, self.seed_id, wandb_config
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
        self.time_to_print = time_to_print
        self.what_to_print = what_to_print

        if not reload and verbose:
            print_welcome()
            print_startup(
                self.experiment_dir,
                self.config_fname,
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
            print_reload(
                self.experiment_dir,
            )

    def setup_experiment(
        self,
        base_exp_dir: str,
        config_fname: Union[str, None],
        seed_id: str,
        overwrite_experiment_dir: bool = False,
        reload: bool = False,
    ) -> None:
        """Setup directory name and clean up previous logging data."""
        # Get timestamp of experiment & create new directories
        if config_fname is not None:
            self.base_str = os.path.split(config_fname)[1].split(".")[0]
            if not reload:
                self.experiment_dir = os.path.join(base_exp_dir, self.base_str)
            else:
                # Don't redefine experiment directory but get already existing
                exp_dir = [
                    f
                    for f in os.listdir(base_exp_dir)
                    if f.endswith(self.base_str)
                ][0]
                self.experiment_dir = os.path.join(base_exp_dir, exp_dir)
        else:
            self.base_str = ""
            self.experiment_dir = base_exp_dir

        self.log_save_fname = os.path.join(
            self.experiment_dir, "logs/", "log_" + seed_id + ".hdf5"
        )
        aggregated_log_save_fname = os.path.join(
            self.experiment_dir, "logs/", "log.hdf5"
        )

        # Delete old experiment logging directory
        if overwrite_experiment_dir and not reload:
            if os.path.exists(self.log_save_fname):
                Console().log(
                    "Be careful - you are overwriting an existing log."
                )
                os.remove(self.log_save_fname)
            if os.path.exists(aggregated_log_save_fname):
                Console().log(
                    "Be careful - you are overwriting an existing aggregated"
                    " log."
                )
                os.remove(aggregated_log_save_fname)
            if self.use_tboard:
                Console().log(
                    "Be careful - you are overwriting existing tboards."
                )
                if os.path.exists(
                    os.path.join(self.experiment_dir, "tboards/")
                ):
                    shutil.rmtree(os.path.join(self.experiment_dir, "tboards/"))

    def get_configs_ready(
        self, config_fname: Union[str, None], config_dict: Union[dict, None]
    ):
        """Load configuration if provided and set config_dict."""
        if config_fname is not None:
            self.config_dict = load_config(config_fname)
        elif config_dict is not None:
            self.config_dict = config_dict
        else:
            self.config_dict = {}

    def create_logging_dir(
        self,
        config_fname: Union[str, None],
        config_dict: Union[dict, None],
    ):
        """Create new empty dir for experiment (if not existing)."""
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Copy over json configuration file if it exists
        if config_fname is not None:
            fname, fext = os.path.splitext(config_fname)
        else:
            fname, fext = "pholder", ".yaml"

        if config_fname is not None:
            config_copy = os.path.join(
                self.experiment_dir, self.base_str + fext
            )
            shutil.copy(config_fname, config_copy)
            self.config_copy = config_copy
        elif config_dict is not None:
            config_copy = os.path.join(
                self.experiment_dir, "config_dict" + fext
            )
            with open(config_copy, "w") as outfile:
                yaml.dump(config_dict, outfile, default_flow_style=False)
            self.config_copy = config_copy
        else:
            self.config_copy = "config-not-provided"

        # Create .hdf5 logging sub-directory
        os.makedirs(os.path.join(self.experiment_dir, "logs/"), exist_ok=True)

    def update(
        self,
        clock_tick: Dict[str, int],
        stats_tick: Dict[str, float],
        model=None,
        plot_fig=None,
        extra_obj=None,
        grads=None,
        save=False,
    ) -> None:
        """Update with the newest tick of performance stats, net weights"""
        # Make sure that timeseries data consists of floats
        stats_tick = {
            key: float(value) if type(value) != np.ndarray else value
            for (key, value) in stats_tick.items()
        }

        # Update the stats log with newest timeseries data
        c_tick, s_tick = self.stats_log.update(clock_tick, stats_tick)
        # Update the tensorboard log with the newest event
        if self.use_tboard:
            self.tboard_log.update(
                self.stats_log.time_to_track,
                clock_tick,
                stats_tick,
                self.model_log.model_type,
                model,
                grads,
                plot_fig,
            )
        if self.use_wandb:
            self.wandb_log.update(
                clock_tick,
                stats_tick,
                self.model_log.model_type,
                model,
                grads,
                plot_fig,
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
            if (
                self.stats_log.stats_update_counter % self.print_every_k_updates
                == 0
            ):
                # Print storage paths generated/updated
                print_storage(
                    fig_path=(
                        self.figure_log.fig_storage_paths[-1]
                        if plot_fig is not None
                        else None
                    ),
                    extra_path=(
                        self.extra_log.extra_storage_paths[-1]
                        if extra_obj is not None
                        else None
                    ),
                    init_model_path=(
                        self.model_log.init_model_save_fname
                        if model is not None and self.model_log.init_model_saved
                        else None
                    ),
                    final_model_path=(
                        self.model_log.final_model_save_fname
                        if model is not None
                        else None
                    ),
                    every_k_model_path=(
                        self.model_log.every_k_ckpt_list[-1]
                        if model is not None and self.model_log.stored_every_k
                        else None
                    ),
                    top_k_model_path=(
                        self.model_log.top_k_ckpt_list[-1]
                        if model is not None and self.model_log.stored_top_k
                        else None
                    ),
                    print_first=self.print_counter == 0,
                )
                # Only print column name header at 1st print!
                if self.time_to_print is None:
                    time_to_p = self.stats_log.time_to_track
                else:
                    time_to_p = ["time", "time_elapsed", "num_updates"]
                if self.what_to_print is None:
                    what_to_p = self.stats_log.what_to_track
                else:
                    what_to_p = self.what_to_print
                print_update(
                    time_to_p,
                    what_to_p,
                    c_tick,
                    s_tick,
                    self.print_counter == 0,
                )
                self.print_counter += 1

    def save_init_model(self, model):
        """Save initial model checkpoint."""
        self.model_log.save_init_model(model)

    def save_model(self, model):
        """Save a model checkpoint."""
        self.model_log.save(
            model, self.stats_log.clock_tracked, self.stats_log.stats_tracked
        )

    def save_plot(self, fig, fig_fname: Union[str, None] = None):
        """Store a figure in a experiment_id/figures directory."""
        # Create main logging dir and .hdf5 sub-directory
        if not self.log_setup_counter:
            self.create_logging_dir(self.config_fname, self.config_dict)
            self.log_setup_counter += 1
        self.figure_log.save(fig, fig_fname)
        write_to_hdf5(
            self.log_save_fname,
            self.seed_id + "/meta/fig_storage_paths",
            self.figure_log.fig_storage_paths,
        )

    def save_extra(self, obj, obj_fname: Union[str, None] = None):
        """Helper fct. to save object (dict/etc.) as .pkl in exp. subdir."""
        # Create main logging dir and .hdf5 sub-directory
        if not self.log_setup_counter:
            self.create_logging_dir(self.config_fname, self.config_dict)
            self.log_setup_counter += 1
        self.extra_log.save(obj, obj_fname)
        write_to_hdf5(
            self.log_save_fname,
            self.seed_id + "/meta/extra_storage_paths",
            self.extra_log.extra_storage_paths,
        )

    def save(self):
        """Create compressed .hdf5 file containing group <random-seed-id>"""
        # Create main logging dir and .hdf5 sub-directory
        if not self.log_setup_counter:
            self.create_logging_dir(self.config_fname, self.config_dict)
            self.log_setup_counter += 1

        # Create "datasets" to store in the hdf5 file [time, stats]
        # Store all relevant meta data (log filename, checkpoint filename)
        if self.log_save_counter == 0:
            data_paths = [
                self.seed_id + "/meta/log_paths",
                self.seed_id + "/meta/experiment_dir",
                self.seed_id + "/meta/config_fname",
                self.seed_id + "/meta/eval_id",
                self.seed_id + "/meta/model_type",
                self.seed_id + "/meta/config_dict",
            ]

            data_to_log = [
                [self.log_save_fname],
                [self.experiment_dir],
                [self.config_copy],
                [self.base_str],
                [self.model_log.model_type],
                [str(self.config_dict)],
            ]

            for i in range(len(data_paths)):
                write_to_hdf5(
                    self.log_save_fname, data_paths[i], data_to_log[i]
                )

            if (
                self.model_log.save_top_k_ckpt
                or self.model_log.save_every_k_ckpt
            ):
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

        # Store final and initial checkpoint if provided
        if self.model_log.model_save_counter > 0:
            write_to_hdf5(
                self.log_save_fname,
                self.seed_id + "/meta/model_ckpt",
                [self.model_log.final_model_save_fname],
            )

        if self.model_log.init_model_saved:
            write_to_hdf5(
                self.log_save_fname,
                self.seed_id + "/meta/init_ckpt",
                [self.model_log.init_model_save_fname],
            )

        # Store all time_to_track variables
        for o_name in self.stats_log.time_to_track:
            if o_name != "time":
                write_to_hdf5(
                    self.log_save_fname,
                    self.seed_id + "/time/" + o_name,
                    self.stats_log.clock_tracked[o_name],
                    dtype="float32",
                )
            else:
                write_to_hdf5(
                    self.log_save_fname,
                    self.seed_id + "/time/" + o_name,
                    self.stats_log.clock_tracked[o_name],
                )

        # Store all what_to_track variables
        for o_name in self.stats_log.what_to_track:
            data_to_store = self.stats_log.stats_tracked[o_name]
            data_to_store = np.array(data_to_store)
            if len(data_to_store) > 0:
                if type(data_to_store[0]) == np.ndarray:
                    data_to_store = np.stack(data_to_store)
                    dtype = np.dtype("float32")
                if type(data_to_store[0]) in [np.str_, str]:
                    dtype = "S5000"
                if type(data_to_store[0]) in [bytes, np.str_]:
                    dtype = np.dtype("S5000")
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
            data_types = ["int32", "S5000"]
            for i in range(len(data_paths)):
                write_to_hdf5(
                    self.log_save_fname,
                    data_paths[i],
                    data_to_log[i],
                    data_types[i],
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
            data_types = ["int32", "S5000", "float32"]
            for i in range(len(data_paths)):
                write_to_hdf5(
                    self.log_save_fname,
                    data_paths[i],
                    data_to_log[i],
                    data_types[i],
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
        return (
            update_counter + 1
        ) % self.log_every_j_steps == 0 or update_counter == 0
