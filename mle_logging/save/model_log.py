import os
from typing import Union
from ..utils import save_pkl_object


class ModelLog(object):
    """ Model Logger Class Instance. """
    def __init__(self,
                 experiment_dir: str = "/",
                 model_type: str = "no-model-type-provided",
                 ckpt_time_to_track: Union[str, None] = None,
                 save_every_k_ckpt: Union[int, None] = None,
                 save_top_k_ckpt: Union[int, None] = None,
                 top_k_metric_name: Union[str, None] = None,
                 top_k_minimize_metric: Union[bool, None] = None,):
        # Setup model logging directories
        self.experiment_dir = experiment_dir
        self.save_every_k_ckpt = save_every_k_ckpt
        self.save_top_k_ckpt = save_top_k_ckpt
        self.ckpt_time_to_track = ckpt_time_to_track
        self.top_k_metric_name = top_k_metric_name
        self.top_k_minimize_metric = top_k_minimize_metric

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


def save_torch_model(path_to_store: str, model) -> None:
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


def save_tensorflow_model(path_to_store: str, model) -> None:
    """Store a tensorflow checkpoint for a model."""
    model.save_weights(path_to_store)


def save_model_ckpt(model, model_save_fname: str, model_type: str) -> None:
    """Save the most recent model checkpoint."""
    if model_type == "torch":
        # Torch model case - save model state dict as .pt checkpoint
        save_torch_model(model_save_fname, model)
    elif model_type == "tensorflow":
        model.save_weights(model_save_fname)
    elif model_type in ["jax", "sklearn", "numpy"]:
        # JAX/sklearn save parameter dict/model as dictionary
        save_pkl_object(model, model_save_fname)
    else:
        raise ValueError("Provide valid model_type [torch, jax, sklearn, numpy].")
