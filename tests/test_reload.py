import shutil
import collections
import os
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from mle_logging import MLELogger

time_tic1 = {"num_steps": 10, "num_epochs": 1}
stats_tic1 = {"train_loss": 0.1234, "test_loss": 0.1235}
time_tic2 = {"num_steps": 20, "num_epochs": 1}
stats_tic2 = {"train_loss": 0.2, "test_loss": 0.1}
time_tic3 = {"num_steps": 30, "num_epochs": 1}
stats_tic3 = {"train_loss": 0.223, "test_loss": 0.097}
time_tic4 = {"num_steps": 40, "num_epochs": 1}
stats_tic4 = {"train_loss": 0.123, "test_loss": 0.085}


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


model = DummyModel()

fig, ax = plt.subplots()
ax.plot(np.random.normal(0, 1, 20))

some_dict = {"hi": "there"}

log_config = {
    "time_to_track": ["num_steps", "num_epochs"],
    "what_to_track": ["train_loss", "test_loss"],
    "experiment_dir": "reload_dir/",
    "model_type": "torch",
    "ckpt_time_to_track": "num_steps",
    "save_every_k_ckpt": 2,
    "save_top_k_ckpt": 2,
    "top_k_metric_name": "test_loss",
    "top_k_minimize_metric": True,
}


def test_reload():
    """Test reloading/continuation of previous log with top/every k."""
    if os.path.exists(log_config["experiment_dir"]) and os.path.isdir(
        log_config["experiment_dir"]
    ):
        shutil.rmtree(log_config["experiment_dir"])

    log = MLELogger(**log_config)
    log.update(time_tic1, stats_tic1, model, fig, some_dict, save=True)
    log.update(time_tic2, stats_tic2, model, fig, some_dict, save=True)
    log.update(time_tic3, stats_tic3, model, fig, some_dict, save=True)

    # Reload the previously instantiated logger from the directory
    relog = MLELogger(**log_config, reload=True)
    # Check correctness of checkpoints
    assert collections.Counter(relog.model_log.top_k_ckpt_list) == collections.Counter(
        [
            "reload_dir/models/top_k/top_k_no_seed_provided_top_0.pt",
            "reload_dir/models/top_k/top_k_no_seed_provided_top_1.pt",
        ]
    )
    assert collections.Counter(
        relog.model_log.top_k_storage_time
    ) == collections.Counter([20, 30])
    assert np.allclose(relog.model_log.top_k_performance, [0.097, 0.1])
    assert collections.Counter(
        relog.model_log.every_k_storage_time
    ) == collections.Counter([20])
    assert collections.Counter(
        relog.model_log.every_k_ckpt_list
    ) == collections.Counter(
        ["reload_dir/models/every_k/every_k_no_seed_provided_k_2.pt"]
    )

    # Check correctness of figure paths
    assert collections.Counter(
        relog.figure_log.fig_storage_paths
    ) == collections.Counter(
        [
            "reload_dir/figures/fig_1_no_seed_provided.png",
            "reload_dir/figures/fig_2_no_seed_provided.png",
            "reload_dir/figures/fig_3_no_seed_provided.png",
        ]
    )
    # Check correctness of extra paths
    assert collections.Counter(
        relog.extra_log.extra_storage_paths
    ) == collections.Counter(
        [
            "reload_dir/extra/extra_1_no_seed_provided.pkl",
            "reload_dir/extra/extra_2_no_seed_provided.pkl",
            "reload_dir/extra/extra_3_no_seed_provided.pkl",
        ]
    )

    # Check correctness of reloaded statistics
    assert np.allclose(
        relog.stats_log.stats_tracked["test_loss"], np.array([0.1235, 0.1, 0.097])
    )
    assert np.allclose(
        relog.stats_log.clock_tracked["num_steps"], np.array([10, 20, 30])
    )

    # Add new result to log
    relog.update(time_tic4, stats_tic4, model, fig, some_dict, save=True)

    # Check correctness of figure paths
    assert collections.Counter(
        relog.figure_log.fig_storage_paths
    ) == collections.Counter(
        [
            "reload_dir/figures/fig_1_no_seed_provided.png",
            "reload_dir/figures/fig_2_no_seed_provided.png",
            "reload_dir/figures/fig_3_no_seed_provided.png",
            "reload_dir/figures/fig_4_no_seed_provided.png",
        ]
    )
    # Check correctness of extra paths
    assert collections.Counter(
        relog.extra_log.extra_storage_paths
    ) == collections.Counter(
        [
            "reload_dir/extra/extra_1_no_seed_provided.pkl",
            "reload_dir/extra/extra_2_no_seed_provided.pkl",
            "reload_dir/extra/extra_3_no_seed_provided.pkl",
            "reload_dir/extra/extra_4_no_seed_provided.pkl",
        ]
    )

    # Check correctness of reloaded statistics
    assert np.allclose(
        np.array(relog.stats_log.stats_tracked["test_loss"]),
        np.array([0.1235, 0.1, 0.097, 0.085]),
    )
    assert np.allclose(
        np.array(relog.stats_log.clock_tracked["num_steps"]),
        np.array([10, 20, 30, 40]),
    )

    # Clean up/delete files
    shutil.rmtree(log_config["experiment_dir"])
