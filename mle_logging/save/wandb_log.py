import os
from typing import Dict, Optional


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
            self.wandb_config["name"] = "{}-{}".format(config_fname, seed_id)
        else:
            self.wandb_config["job_type"] = config_fname
            self.wandb_config["name"] = seed_id
        setup_wandb_env(self.wandb_config)
        wandb.init(config=config_dict)
        self.step_counter = 0

    def update(
        self,
        clock_tick: Dict[str, int],
        stats_tick: Dict[str, float],
        model_type: Optional[str] = None,
        model=None,
        plot_to_wandb: Optional[str] = None,
    ):
        """Update the wandb with the newest events"""
        log_dict = {**stats_tick, **clock_tick}
        if plot_to_wandb is not None:
            log_dict["img"] = wandb.Image(plot_to_wandb)
        wandb.log(
            log_dict,
            step=self.step_counter,
        )
        self.step_counter += 1
