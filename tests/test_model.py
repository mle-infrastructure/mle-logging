import os
import shutil
import numpy as np
import torch.nn as nn
from sklearn.svm import SVC
from mle_logging import MLELogger, load_model, load_log


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


def create_tensorflow_model():
    import tensorflow as tf
    from tensorflow import keras
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])

    return model


def test_save_load_torch():
    """Test saving and loading of torch model."""
    # Remove experiment dir at start of test
    if os.path.exists(log_config["experiment_dir"]) and os.path.isdir(
        log_config["experiment_dir"]
    ):
        shutil.rmtree(log_config["experiment_dir"])

    # Instantiate logging to experiment_dir
    log_config["model_type"] = "torch"
    log = MLELogger(**log_config)

    # Save a torch model
    model = DummyModel()
    log.update(time_tic, stats_tic, model, save=True)
    # Assert the existence of the files
    file_to_check = os.path.join(
        log_config["experiment_dir"], "models/final",
        "final_no_seed_provided.pt"
    )
    assert os.path.exists(file_to_check)

    # Load log and afterwards the model
    relog = load_log(log_config["experiment_dir"])
    remodel = load_model(relog.meta.model_ckpt,
                         log_config["model_type"], model)
    assert type(remodel) == DummyModel
    # Finally -- clean up
    shutil.rmtree(log_config["experiment_dir"])


def test_save_load_tf():
    """Test saving and loading of tensorflow model."""
    # Remove experiment dir at start of test
    if os.path.exists(log_config["experiment_dir"]) and os.path.isdir(
        log_config["experiment_dir"]
    ):
        shutil.rmtree(log_config["experiment_dir"])

    # Instantiate logging to experiment_dir
    log_config["model_type"] = "tensorflow"
    log = MLELogger(**log_config)

    # Save a torch model
    model = create_tensorflow_model()
    log.update(time_tic, stats_tic, model, save=True)
    # Assert the existence of the files
    file_to_check = os.path.join(
        log_config["experiment_dir"], "models/final",
        "final_no_seed_provided.pt"
        + ".data-00000-of-00001"
    )
    assert os.path.exists(file_to_check)
    file_to_check = os.path.join(
        log_config["experiment_dir"], "models/final",
        "final_no_seed_provided.pt"
        + ".index"
    )
    assert os.path.exists(file_to_check)
    file_to_check = os.path.join(
        log_config["experiment_dir"], "models/final",
        "checkpoint"
    )
    assert os.path.exists(file_to_check)

    # Load log and afterwards the model
    relog = load_log(log_config["experiment_dir"])
    _ = load_model(relog.meta.model_ckpt,
                   log_config["model_type"], model)

    # Finally -- clean up
    shutil.rmtree(log_config["experiment_dir"])


def test_save_load_jax():
    """Test saving and loading of jax model."""
    # Remove experiment dir at start of test
    if os.path.exists(log_config["experiment_dir"]) and os.path.isdir(
        log_config["experiment_dir"]
    ):
        shutil.rmtree(log_config["experiment_dir"])

    # Instantiate logging to experiment_dir
    log_config["model_type"] = "jax"
    log = MLELogger(**log_config)

    # Save a torch model
    import jax
    import haiku as hk

    def lenet_fn(x):
        """Standard LeNet-300-100 MLP network."""
        mlp = hk.Sequential([
                hk.Flatten(),
                hk.Linear(300), jax.nn.relu,
                hk.Linear(100), jax.nn.relu,
                hk.Linear(10),
        ])
        return mlp(x)

    lenet = hk.without_apply_rng(hk.transform(lenet_fn))
    params = lenet.init(jax.random.PRNGKey(42), np.zeros((32, 784)))

    log.update(time_tic, stats_tic, params, save=True)
    # Assert the existence of the files
    file_to_check = os.path.join(
        log_config["experiment_dir"], "models/final",
        "final_no_seed_provided.pkl"
    )
    assert os.path.exists(file_to_check)

    # Load log and afterwards the model
    relog = load_log(log_config["experiment_dir"])
    _ = load_model(relog.meta.model_ckpt,
                   log_config["model_type"])

    # Finally -- clean up
    shutil.rmtree(log_config["experiment_dir"])


def test_save_load_sklearn():
    """Test saving and loading of sklearn model."""
    # Remove experiment dir at start of test
    if os.path.exists(log_config["experiment_dir"]) and os.path.isdir(
        log_config["experiment_dir"]
    ):
        shutil.rmtree(log_config["experiment_dir"])

    # Instantiate logging to experiment_dir
    log_config["model_type"] = "sklearn"
    log = MLELogger(**log_config)

    # Save a torch model
    model = SVC(gamma='auto')
    log.update(time_tic, stats_tic, model, save=True)

    # Assert the existence of the files
    file_to_check = os.path.join(
        log_config["experiment_dir"], "models/final",
        "final_no_seed_provided.pkl"
    )
    assert os.path.exists(file_to_check)

    # Load log and afterwards the model
    relog = load_log(log_config["experiment_dir"])
    remodel = load_model(relog.meta.model_ckpt,
                         log_config["model_type"], model)
    assert type(remodel) == SVC
    # Finally -- clean up
    shutil.rmtree(log_config["experiment_dir"])


def test_save_load_numpy():
    """Test saving and loading of numpy model/array."""
    # Remove experiment dir at start of test
    if os.path.exists(log_config["experiment_dir"]) and os.path.isdir(
        log_config["experiment_dir"]
    ):
        shutil.rmtree(log_config["experiment_dir"])

    # Instantiate logging to experiment_dir
    log_config["model_type"] = "numpy"
    log = MLELogger(**log_config)

    # Save a torch model
    model = np.array([1, 2, 3, 4])
    log.update(time_tic, stats_tic, model, save=True)

    # Assert the existence of the files
    file_to_check = os.path.join(
        log_config["experiment_dir"], "models/final",
        "final_no_seed_provided.pkl"
    )
    assert os.path.exists(file_to_check)

    # Load log and afterwards the model
    relog = load_log(log_config["experiment_dir"])
    remodel = load_model(relog.meta.model_ckpt,
                         log_config["model_type"], model)
    assert (remodel == model).all()
    # Finally -- clean up
    shutil.rmtree(log_config["experiment_dir"])
