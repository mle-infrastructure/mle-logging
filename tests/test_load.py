import os
import shutil
import datetime
import collections
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from mle_logging import MLELogger
from mle_logging import merge_seed_logs, load_log
from mle_logging import merge_config_logs, load_meta_log


log_config = {
    "time_to_track": ["num_updates", "num_epochs"],
    "what_to_track": ["train_loss", "test_loss"],
    "experiment_dir": "experiment_dir/",
    "model_type": "torch",
}

log_config1_seed1 = {
    "time_to_track": ["num_updates", "num_epochs"],
    "what_to_track": ["train_loss", "test_loss"],
    "experiment_dir": "experiment_dir/",
    "config_fname": "examples/config_1.json",
    "model_type": "torch",
    "seed_id": "seed_1"
}

log_config1_seed2 = {
    "time_to_track": ["num_updates", "num_epochs"],
    "what_to_track": ["train_loss", "test_loss"],
    "experiment_dir": "experiment_dir/",
    "config_fname": "examples/config_1.json",
    "model_type": "torch",
    "seed_id": "seed_2"
}

log_config2_seed1 = {
    "time_to_track": ["num_updates", "num_epochs"],
    "what_to_track": ["train_loss", "test_loss"],
    "experiment_dir": "experiment_dir/",
    "config_fname": "examples/config_2.json",
    "model_type": "torch",
    "seed_id": "seed_1"
}

log_config2_seed2 = {
    "time_to_track": ["num_updates", "num_epochs"],
    "what_to_track": ["train_loss", "test_loss"],
    "experiment_dir": "experiment_dir/",
    "config_fname": "examples/config_2.json",
    "model_type": "torch",
    "seed_id": "seed_2"
}

time_tic1 = {"num_updates": 10, "num_epochs": 1}
stats_tic1 = {"train_loss": 0.1234, "test_loss": 0.1235}

time_tic2 = {"num_updates": 10, "num_epochs": 1}
stats_tic2 = {"train_loss": 0.2, "test_loss": 0.1}


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


def test_load_single():
    """Test loading of single seed/config."""
    # Remove experiment dir at start of test
    if os.path.exists(log_config["experiment_dir"]) and os.path.isdir(
        log_config["experiment_dir"]
    ):
        shutil.rmtree(log_config["experiment_dir"])

    # Log some data
    log = MLELogger(**log_config)
    log.update(time_tic1, stats_tic1, model, fig, some_dict, save=True)

    # Reload log and check correctness of results
    relog = load_log(log_config["experiment_dir"])

    meta_keys = ['config_fname', 'eval_id', 'experiment_dir',
                 'extra_storage_paths', 'fig_storage_paths',
                 'log_paths', 'model_ckpt', 'init_ckpt', 'model_type']
    assert (collections.Counter(list(relog.meta.keys()))
            == collections.Counter(meta_keys))

    assert relog.stats.train_loss == 0.1234
    assert relog.time.num_updates == 10
    assert (relog.meta.fig_storage_paths
            == b'experiment_dir/figures/fig_1_no_seed_provided.png')
    assert (relog.meta.extra_storage_paths
            == b'experiment_dir/extra/extra_1_no_seed_provided.pkl')
    assert (relog.meta.model_ckpt[0].decode()
            == f'experiment_dir/models/final/final_no_seed_provided.pt')
    # Finally -- clean up
    shutil.rmtree(log_config["experiment_dir"])


def test_merge_load_seeds():
    """Test merging of multiple seeds and loading."""
    if os.path.exists(log_config1_seed1["experiment_dir"]) and os.path.isdir(
        log_config1_seed1["experiment_dir"]
    ):
        shutil.rmtree(log_config1_seed1["experiment_dir"])

    # Log some data for both seeds
    log_seed1 = MLELogger(**log_config1_seed1)
    log_seed1.update(time_tic1, stats_tic1, model, fig, some_dict, save=True)

    log_seed2 = MLELogger(**log_config1_seed2)
    log_seed2.update(time_tic2, stats_tic2, model, fig, some_dict, save=True)

    timestr = datetime.datetime.today().strftime("%Y-%m-%d")[2:]
    experiment_dir = log_config["experiment_dir"] + f"{timestr}_config_1/"
    merged_path = os.path.join(experiment_dir, "logs", "config_1.hdf5")

    # Merge different random seeds into one .hdf5 file
    merge_seed_logs(merged_path, experiment_dir)
    assert os.path.exists(os.path.join(experiment_dir,
                                       f"{timestr}_config_1.json"))

    # Load the merged log - Individual seeds can be accessed via log.seed_1, etc.
    log = load_log(experiment_dir)
    assert log.seed_1.stats.train_loss == 0.1234
    assert log.seed_2.stats.train_loss == 0.2

    # Load the merged & aggregated log
    log = load_log(experiment_dir, aggregate_seeds=True)
    assert np.isclose(log.stats.train_loss.mean[0], np.mean([0.1234, 0.2]))
    assert np.isclose(log.stats.train_loss.std[0], np.std([0.1234, 0.2]))

    # Finally -- clean up
    shutil.rmtree(log_config1_seed1["experiment_dir"])


def test_merge_load_configs():
    """Test merging of multiple configs and loading."""
    if os.path.exists(log_config1_seed1["experiment_dir"]) and os.path.isdir(
        log_config1_seed1["experiment_dir"]
    ):
        shutil.rmtree(log_config1_seed1["experiment_dir"])

    # Log some data for both seeds and both configs
    log_c1_s1 = MLELogger(**log_config1_seed1)
    log_c1_s2 = MLELogger(**log_config1_seed2)
    log_c2_s1 = MLELogger(**log_config2_seed1)
    log_c2_s2 = MLELogger(**log_config2_seed2)
    log_c1_s1.update(time_tic1, stats_tic1, model, fig, some_dict, save=True)
    log_c1_s2.update(time_tic2, stats_tic2, model, fig, some_dict, save=True)
    log_c2_s1.update(time_tic1, stats_tic1, model, fig, some_dict, save=True)
    log_c2_s2.update(time_tic2, stats_tic2, model, fig, some_dict, save=True)

    # Merge different random seeds for each config into separate .hdf5 file
    timestr = datetime.datetime.today().strftime("%Y-%m-%d")[2:]
    merge_seed_logs(f"{log_config1_seed1['experiment_dir']}/{timestr}_config_1/logs/config_1.hdf5",
                    f"{log_config1_seed1['experiment_dir']}/{timestr}_config_1/")
    merge_seed_logs(f"{log_config1_seed1['experiment_dir']}/{timestr}_config_2/logs/config_2.hdf5",
                    f"{log_config1_seed1['experiment_dir']}/{timestr}_config_2/")

    # Aggregate the different merged configuration .hdf5 files into single meta log
    eval_ids = ["config_1", "config_2"]
    seed_ids = ["seed_1", "seed_2"]
    merge_config_logs(experiment_dir = f"{log_config1_seed1['experiment_dir']}",
                      all_run_ids = eval_ids)
    meta_path = f"{log_config1_seed1['experiment_dir']}/meta_log.hdf5"
    meta_log = load_meta_log(meta_path)

    assert (collections.Counter(meta_log.eval_ids)
            == collections.Counter(eval_ids))

    aggreg_keys = ['mean', 'std', 'p50', 'p10', 'p25', 'p75', 'p90']
    assert (collections.Counter(list(meta_log.config_1.stats.test_loss.keys()))
            == collections.Counter(aggreg_keys))

    meta_log = load_meta_log(meta_path, aggregate_seeds=False)
    assert (collections.Counter(meta_log.eval_ids)
            == collections.Counter(eval_ids))
    assert (collections.Counter(meta_log.config_1.keys())
            == collections.Counter(seed_ids))
    # Finally -- clean up
    shutil.rmtree(log_config1_seed1["experiment_dir"])
