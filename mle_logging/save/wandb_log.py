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
    if "name" in wandb_config.keys():
        os.environ["WANDB_NAME"] = wandb_config["name"]
    if "group" in wandb_config.keys():
        os.environ["WANDB_RUN_GROUP"] = wandb_config["group"]
    if "job_type" in wandb_config.keys():
        os.environ["WANDB_JOB_TYPE"] = wandb_config["job_type"]
        os.environ["WANDB_TAGS"] = "{}, {}".format(
            wandb_config["name"], wandb_config["job_type"]
        )
    os.environ["WANDB_SILENT"] = "true"


class WandbLog(object):
    """Weights&Biases Logger Class Instance."""

    def __init__(self, wandb_config: Optional[dict]):
        # Setup figure logging directories
        try:
            import wandb
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                f"{err}. You need to install "
                "`wandb` if you want that "
                "MLELogger logs to Weights&Biases."
            )
        self.wandb_config = wandb_config
        # config should contain - key, entity, project, group (experiment)

    def setup(self, config_dict: dict, config_fname: str, seed_id: str):
        """Setup wandb process for logging."""
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
        log_dict = {**clock_tick, **stats_tick}
        if plot_to_wandb is not None:
            log_dict["img"] = wandb.Image(plot_to_wandb)
        wandb.log(
            log_dict,
            step=self.step_counter,
        )
        self.step_counter += 1
