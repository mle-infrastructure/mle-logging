import os
import shutil
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from mle_logging import MLELogger


log_config = {
    "time_to_track": ["num_updates", "num_epochs"],
    "what_to_track": ["train_loss", "test_loss"],
    "experiment_dir": "experiment_dir/",
    "config_fname": None,
    "use_tboard": True,
    "model_type": "torch",
}

time_tic = {"num_updates": 10, "num_epochs": 1}
stats_tic = {"train_loss": 0.1234, "test_loss": 0.1235}


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


def test_update_log():
    # Remove experiment dir at start of test
    if os.path.exists(log_config["experiment_dir"]) and os.path.isdir(
        log_config["experiment_dir"]
    ):
        shutil.rmtree(log_config["experiment_dir"])

    # Instantiate logging to experiment_dir
    log = MLELogger(**log_config)
    # Update the log with collected data & save it to .hdf5
    log.update(time_tic, stats_tic)
    log.save()

    # Assert the existence of the files
    assert os.path.exists(os.path.join(log_config["experiment_dir"], "logs"))
    assert os.path.exists(os.path.join(log_config["experiment_dir"], "tboards"))
    file_to_check = os.path.join(
        log_config["experiment_dir"], "logs", "log_no_seed_provided.hdf5"
    )
    assert os.path.exists(file_to_check)

    # Finally -- clean up
    shutil.rmtree(log_config["experiment_dir"])


def test_save_plot():
    # Remove experiment dir at start of test
    if os.path.exists(log_config["experiment_dir"]) and os.path.isdir(
        log_config["experiment_dir"]
    ):
        shutil.rmtree(log_config["experiment_dir"])

    # Instantiate logging to experiment_dir
    log = MLELogger(**log_config)

    # Save a matplotlib figure as .png
    log.save_plot(fig)

    # Assert the existence of the files
    file_to_check = os.path.join(
        log_config["experiment_dir"], "figures", "fig_1_no_seed_provided.png"
    )
    assert os.path.exists(file_to_check)

    # Finally -- clean up
    shutil.rmtree(log_config["experiment_dir"])


def test_save_extra():
    # Remove experiment dir at start of test
    if os.path.exists(log_config["experiment_dir"]) and os.path.isdir(
        log_config["experiment_dir"]
    ):
        shutil.rmtree(log_config["experiment_dir"])

    # Instantiate logging to experiment_dir
    log = MLELogger(**log_config)

    # Save a dict as a .pkl object
    log.save_extra(some_dict)

    # Assert the existence of the files
    file_to_check = os.path.join(
        log_config["experiment_dir"], "extra", "extra_1_no_seed_provided.pkl"
    )
    assert os.path.exists(file_to_check)

    # Finally -- clean up
    shutil.rmtree(log_config["experiment_dir"])


def test_all_in_one():
    # Remove experiment dir at start of test
    if os.path.exists(log_config["experiment_dir"]) and os.path.isdir(
        log_config["experiment_dir"]
    ):
        shutil.rmtree(log_config["experiment_dir"])

    # Instantiate logging to experiment_dir
    log = MLELogger(**log_config)

    # Save a dict as a .pkl object
    log.save_init_model(model)
    log.update(time_tic, stats_tic, model, fig, some_dict, save=True)

    # Assert the existence of the files
    assert os.path.exists(os.path.join(log_config["experiment_dir"], "logs"))
    assert os.path.exists(os.path.join(log_config["experiment_dir"], "tboards"))
    file_to_check = os.path.join(
        log_config["experiment_dir"], "logs", "log_no_seed_provided.hdf5"
    )
    assert os.path.exists(file_to_check)

    file_to_check = os.path.join(
        log_config["experiment_dir"], "models/init", "init_no_seed_provided.pt"
    )
    assert os.path.exists(file_to_check)

    file_to_check = os.path.join(
        log_config["experiment_dir"], "models/final", "final_no_seed_provided.pt"
    )
    assert os.path.exists(file_to_check)

    file_to_check = os.path.join(
        log_config["experiment_dir"], "figures", "fig_1_no_seed_provided.png"
    )
    assert os.path.exists(file_to_check)

    file_to_check = os.path.join(
        log_config["experiment_dir"], "extra", "extra_1_no_seed_provided.pkl"
    )
    assert os.path.exists(file_to_check)

    # Finally -- clean up
    shutil.rmtree(log_config["experiment_dir"])


def test_comms():
    """Test functional verbose statements."""
    log = MLELogger(**log_config, verbose=True)
    log.update(time_tic, stats_tic, model, fig, some_dict, save=True)
