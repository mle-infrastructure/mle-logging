import numpy as np
from typing import Dict


class TboardLog(object):
    """Tensorboard Logger Class Instance."""

    def __init__(
        self,
        experiment_dir: str,
        seed_id: str,
    ):
        # Setup figure logging directories
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                f"{err}. You need to install "
                "`torch` if you want that "
                "MLELogger logs to tensorboard."
            )
        self.writer = SummaryWriter(
            experiment_dir + "/tboards/" + "tboard" + "_" + seed_id
        )

    def update(  # noqa: C901
        self,
        time_to_track: list,
        clock_tick: Dict[str, int],
        stats_tick: Dict[str, float],
        model_type: str,
        model=None,
        plot_to_tboard=None,
    ):
        """Update the tensorboard with the newest events"""
        # Set the x-axis time variable to first key provided in time key dict
        time_var_id = clock_tick[time_to_track[1]]

        # Add performance & step counters
        for k in stats_tick.keys():
            self.writer.add_scalar(
                "performance/" + k, np.mean(stats_tick[k]), time_var_id
            )

        # Log the model params & gradients
        if model is not None:
            if model_type == "torch":
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
            elif model_type == "jax":
                # Try to add parameters from nested dict first - then simple
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
