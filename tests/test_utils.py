import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from mle_logging import MLELogger, load_log, load_config
from mle_logging.utils import visualize_1D_lcurves


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


def test_comms():
    """Test functional verbose statements."""
    log = MLELogger(**log_config, verbose=True)
    log.update(time_tic, stats_tic, model, fig, some_dict, save=True)


def test_load_config():
    config = load_config("tests/fixtures/eval_0.yaml", True)
    assert config.lrate == 0.360379148648584


def test_plot_lcurves():
    # Load the merged log - Individual seeds can be accessed via log.seed_1, etc.
    log = load_log("tests/fixtures")
    log.plot("train_loss", "num_updates")

    log = load_log("tests/fixtures", aggregate_seeds=True)
    log.plot("train_loss", "num_updates")
