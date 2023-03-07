import os
from typing import Dict, Optional
import numpy as np
import subprocess as sp


def setup_wandb_env(wandb_config: dict):
    """Set up environment variables for W&B logging."""
    if "key" in wandb_config.keys():
        os.environ["WANDB_API_KEY"] = wandb_config["key"]
    if "entity" in wandb_config.keys():
        os.environ["WANDB_ENTITY"] = wandb_config["entity"]
    if "project" in wandb_config.keys():
        os.environ["WANDB_PROJECT"] = wandb_config["project"]
    else:
        os.environ["WANDB_PROJECT"] = "prototyping"
    if "name" in wandb_config.keys():
        os.environ["WANDB_NAME"] = wandb_config["name"]
    if "group" in wandb_config.keys():
        if wandb_config["group"] is not None:
            os.environ["WANDB_RUN_GROUP"] = wandb_config["group"]
    if "job_type" in wandb_config.keys():
        os.environ["WANDB_JOB_TYPE"] = wandb_config["job_type"]
        os.environ["WANDB_TAGS"] = "{}, {}".format(
            wandb_config["name"], wandb_config["job_type"]
        )
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_DISABLE_SERVICE"] = "true"


class WandbLog(object):
    """Weights&Biases Logger Class Instance."""

    def __init__(
        self,
        config_dict: Optional[dict],
        config_fname: Optional[str],
        seed_id: str,
        wandb_config: Optional[dict],
    ):
        # Setup figure logging directories
        try:
            import wandb

            global wandb
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                f"{err}. You need to install "
                "`wandb` if you want that "
                "MLELogger logs to Weights&Biases."
            )
        self.wandb_config = wandb_config
        # config should contain - key, entity, project, group (experiment)
        for k in ["key", "entity", "project", "group"]:
            assert k in self.wandb_config.keys()

        # Setup the environment variables for W&B logging.
        if config_fname is None:
            config_fname = "pholder_config"
        else:
            path = os.path.normpath(config_fname)
            path_norm = path.split(os.sep)
            config_fname, _ = os.path.splitext(path_norm[-1])

        if config_dict is None:
            config_dict = {}
        self.setup(config_dict, config_fname, seed_id)

    def setup(self, config_dict: dict, config_fname: str, seed_id: str):
        """Setup wandb process for logging."""
        if self.wandb_config["group"] is None:
            self.wandb_config["job_type"] = seed_id
        else:
            self.wandb_config["job_type"] = config_fname
        # Replace name by seed if not otherwise specified
        if self.wandb_config["name"] == "seed0":
            self.wandb_config["name"] = seed_id
        setup_wandb_env(self.wandb_config)

        # Try opening port 10 times
        for _ in range(10):
            try:
                wandb.init(config=config_dict)
                self.correct_setup = True
                break
            except Exception:
                self.correct_setup = False
                pass
        self.step_counter = 0

    def update(
        self,
        clock_tick: Dict[str, int],
        stats_tick: Dict[str, float],
        model_type: Optional[str] = None,
        model=None,
        grads=None,
        plot_to_wandb: Optional[str] = None,
    ):
        """Update the wandb with the newest events"""
        if self.correct_setup:
            log_dict = {}
            for k, v in clock_tick.items():
                log_dict["time/" + k] = v
            for k, v in stats_tick.items():
                log_dict["stats/" + k] = v
            if plot_to_wandb is not None:
                log_dict["img"] = wandb.Image(plot_to_wandb)
            # Log stats to W&B log
            wandb.log(
                log_dict,
                step=self.step_counter,
            )

            # Log model parameters and gradients if provided
            if model is not None and model_type == "jax":
                w_norm, w_hist = get_jax_norm_hist(model)
                wandb.log(
                    {"params_norm/": w_norm, "params_hist/": w_hist},
                    step=self.step_counter,
                )
            if grads is not None and model_type == "jax":
                g_norm, g_hist = get_jax_norm_hist(grads)
                wandb.log(
                    {"grads_norm/": g_norm, "grads_hist/": g_hist},
                    step=self.step_counter,
                )
            # Log model gradients if provided
            self.step_counter += 1

    def upload_gif(self, gif_path: str, video_name: Optional[str] = "video"):
        """Upload a gif file to W&B based on path"""
        wandb.log(
            {"video_name": wandb.Video(gif_path)},
            step=self.step_counter,
        )


def get_jax_norm_hist(model):
    """Get norm of modules in jax model."""
    import jax
    from flax.core import unfreeze

    def norm(val):
        return jax.tree_map(lambda x: np.linalg.norm(x), val)

    def histogram(val):
        return jax.tree_map(lambda x: np.histogram(x, density=True), val)

    w_norm = unfreeze(norm(model))
    hist = histogram(model)
    hist = jax.tree_map(lambda x: jax.device_get(x), unfreeze(hist))
    w_hist = jax.tree_map(
        lambda x: wandb.Histogram(np_histogram=x),
        hist,
        is_leaf=lambda x: isinstance(x, tuple),
    )
    return w_norm, w_hist
