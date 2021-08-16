import numpy as np
from ..utils import load_pkl_object


def load_model(ckpt_path: str, model_type: str, model=None):
    """Helper to reload stored checkpoint/pkl & return trained model."""
    if model_type == "torch":
        try:
            import torch
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                f"{err}. You need to install "
                "`torch` if you want to save a model "
                "checkpoint."
            )

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if model is not None:
            # raise ValueError("Please provide a torch model instance.")
            model.load_state_dict(checkpoint)
            return model
        else:
            return checkpoint
    elif model_type == "tensorflow":
        model.load_weights(ckpt_path)
    elif model_type in ["jax", "sklearn"]:
        model = load_pkl_object(ckpt_path)
        return model
    elif model_type == "numpy":
        model = np.load(ckpt_path, allow_pickle=True)
        return model
    else:
        raise ValueError(
            "Please provide a valid model type ('torch', 'jax'," " 'sklearn', 'numpy')."
        )
