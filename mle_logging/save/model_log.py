import os
import numpy as np
from typing import Union, List
from ..utils import save_pkl_object
from ..load import load_log


class ModelLog(object):
    """Model Logger Class Instance."""

    def __init__(
        self,
        experiment_dir: str = "/",
        seed_id: str = "no_seed_provided",
        model_type: str = "no-model-type-provided",
        ckpt_time_to_track: Union[str, None] = None,
        save_every_k_ckpt: Union[int, None] = None,
        save_top_k_ckpt: Union[int, None] = None,
        top_k_metric_name: Union[str, None] = None,
        top_k_minimize_metric: Union[bool, None] = None,
        reload: bool = False,
    ):
        # Setup model logging
        self.experiment_dir = experiment_dir
        assert model_type in [
            "torch",
            "tensorflow",
            "jax",
            "sklearn",
            "numpy",
            "no-model-type",
        ]
        self.model_type = model_type
        self.save_every_k_ckpt = save_every_k_ckpt
        self.save_top_k_ckpt = save_top_k_ckpt
        self.ckpt_time_to_track = ckpt_time_to_track
        self.top_k_metric_name = top_k_metric_name
        self.top_k_minimize_metric = top_k_minimize_metric

        # Create separate filenames for checkpoints & final trained model
        self.final_ckpt_dir = os.path.join(self.experiment_dir, "models/final/")
        self.final_model_save_fname = os.path.join(
            self.final_ckpt_dir, "final_" + seed_id
        )
        if self.save_every_k_ckpt is not None:
            self.every_k_ckpt_list: List[str] = []
            self.every_k_dir = os.path.join(self.experiment_dir, "models/every_k/")
            self.every_k_model_save_fname = os.path.join(
                self.every_k_dir, "every_k_" + seed_id + "_k_"
            )
        if self.save_top_k_ckpt is not None:
            self.top_k_ckpt_list: List[str] = []
            self.top_k_dir = os.path.join(self.experiment_dir, "models/top_k/")
            self.top_k_model_save_fname = os.path.join(
                self.top_k_dir, "top_k_" + seed_id + "_top_"
            )

        # Different extensions to model checkpoints based on model type
        if self.model_type in ["torch", "tensorflow", "jax", "sklearn", "numpy"]:
            if self.model_type in ["torch", "tensorflow"]:
                self.model_fname_ext = ".pt"
            elif self.model_type in ["jax", "sklearn", "numpy"]:
                self.model_fname_ext = ".pkl"
            self.final_model_save_fname += self.model_fname_ext

        # Initialize counter & lists for top k scores and storage time to track
        if reload:
            self.reload()
        else:
            self.model_save_counter = 0
            if self.save_every_k_ckpt is not None:
                self.every_k_storage_time: List[int] = []
            if self.save_top_k_ckpt is not None:
                self.top_k_performance: List[float] = []
                self.top_k_storage_time: List[int] = []

    def setup_model_ckpt_dir(self):
        """Create separate sub-dirs for checkpoints & final trained model."""
        os.makedirs(self.final_ckpt_dir, exist_ok=True)
        if self.save_every_k_ckpt is not None:
            os.makedirs(self.every_k_dir, exist_ok=True)
        if self.save_top_k_ckpt is not None:
            os.makedirs(self.top_k_dir, exist_ok=True)

    def save(self, model, clock_to_track, stats_to_track):  # noqa: C901
        """Save current state of the model as a checkpoint."""
        # If first model ckpt is saved - generate necessary directories
        self.model_save_counter += 1
        self.stored_every_k = False
        self.stored_top_k = False
        if self.model_save_counter == 1:
            self.setup_model_ckpt_dir()

        # CASE 1: SIMPLE STORAGE OF MOST RECENTLY LOGGED MODEL STATE
        self.save_final_model(model)

        # CASE 2: SEPARATE STORAGE OF EVERY K-TH LOGGED MODEL STATE
        if self.save_every_k_ckpt is not None:
            self.save_every_k_model(model, clock_to_track)

        # CASE 3: STORE TOP-K MODEL STATES BY SOME SCORE
        if self.save_top_k_ckpt is not None:
            self.save_top_k_model(model, clock_to_track, stats_to_track)

    def save_final_model(self, model):
        """Store the most recent model checkpoint and replace old ckpt."""
        save_model_ckpt(model, self.final_model_save_fname, self.model_type)

    def save_every_k_model(self, model, clock_to_track):
        """Store every kth provided checkpoint."""
        if self.model_save_counter % self.save_every_k_ckpt == 0:
            ckpt_path = (
                self.every_k_model_save_fname
                + str(self.model_save_counter)
                + self.model_fname_ext
            )
            save_model_ckpt(model, ckpt_path, self.model_type)
            # Use latest update performance for last checkpoint
            time = clock_to_track[self.ckpt_time_to_track].to_numpy()[-1]
            self.every_k_storage_time.append(time)
            self.every_k_ckpt_list.append(ckpt_path)
            self.stored_every_k = True

    def save_top_k_model(self, model, clock_to_track, stats_to_track):
        """Store top-k checkpoints by performance."""
        # Use latest update performance for last checkpoint
        score = stats_to_track[self.top_k_metric_name].to_numpy()[-1]
        time = clock_to_track[self.ckpt_time_to_track].to_numpy()[-1]

        # Fill up empty top k slots
        if len(self.top_k_performance) < self.save_top_k_ckpt:
            ckpt_path = (
                self.top_k_model_save_fname
                + str(len(self.top_k_performance))
                + self.model_fname_ext
            )
            save_model_ckpt(model, ckpt_path, self.model_type)
            self.top_k_performance.append(score)
            self.top_k_storage_time.append(time)
            self.top_k_ckpt_list.append(ckpt_path)
            self.stored_top_k = True
            return

        # If minimize = replace worst performing model (max score)
        # Note: The archive of checkpoints is not sorted by performance!
        if not self.top_k_minimize_metric:
            top_k_scores = [-1 * s for s in self.top_k_performance]
            score_to_eval = -1 * score
        else:
            top_k_scores = [s for s in self.top_k_performance]
            score_to_eval = score
        if max(top_k_scores) > score_to_eval:
            id_to_replace = np.argmax(top_k_scores)
            self.top_k_performance[id_to_replace] = score
            self.top_k_storage_time[id_to_replace] = time
            ckpt_path = (
                self.top_k_model_save_fname + str(id_to_replace) + self.model_fname_ext
            )
            save_model_ckpt(model, ckpt_path, self.model_type)
            self.stored_top_k = True

    def reload(self):
        """Reload results from previous experiment run."""
        reloaded_log = load_log(self.experiment_dir)
        # Make sure to reload in results for correct seed
        if reloaded_log.eval_ids is None:
            meta_data = reloaded_log.meta
        else:
            meta_data = reloaded_log[self.seed_id].meta

        # Reload counter & lists for top k scores and storage time to track
        if self.save_every_k_ckpt is not None:
            self.every_k_ckpt_list = [ck.decode() for ck in meta_data.every_k_ckpt_list]
            self.every_k_storage_time = meta_data.every_k_storage_time
            self.model_save_counter = int(
                self.every_k_ckpt_list[-1].split(self.model_fname_ext)[0][-1]
            )
        else:
            self.model_save_counter = 0
        if self.save_top_k_ckpt is not None:
            self.top_k_ckpt_list = [ck.decode() for ck in meta_data.top_k_ckpt_list]
            self.top_k_storage_time = meta_data.top_k_storage_time
            self.top_k_performance = meta_data.top_k_performance.tolist()


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
