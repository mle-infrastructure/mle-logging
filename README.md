# A Lightweight Logger for ML Experiments :book:
[![Pyversions](https://img.shields.io/pypi/pyversions/mle-logging.svg?style=flat-square)](https://pypi.python.org/pypi/mle-logging)
[![Docs Latest](https://img.shields.io/badge/docs-dev-blue.svg)](https://roberttlange.github.io/mle-toolbox/logging/mle_logging/)
[![PyPI version](https://badge.fury.io/py/mle-logging.svg)](https://badge.fury.io/py/mle-logging)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/mle-logging/blob/main/examples/getting_started.ipynb)

Simple logging of statistics, model checkpoints, plots and other objects for your Machine Learning Experiments (MLE). Furthermore, the `MLELogger` comes with smooth multi-seed result aggregation and combination of multi-configuration runs.

## The API :video_game:

```python
from mle_logging import MLELogger

# Instantiate logging to experiment_dir
log = MLELogger(time_to_track = ['num_updates', 'num_epochs'],
                what_to_track = ['train_loss', 'test_loss'],
                experiment_dir = "experiment_dir/",
                model_type = 'torch')

time_tic = {'num_updates': 10,
            'num_epochs': 1}
stats_tic = {'train_loss': 0.1234,
             'test_loss': 0.1235}

# Update the log with collected data & save it to .hdf5
log.update(time_tic, stats_tic)
log.save()
```

You can also log model checkpoints, matplotlib figures and other `.pkl` compatible objects.

```python
# Save a model (torch, sklearn, jax, numpy)
import torchvision.models as models
model = models.resnet18()
log.save_model(model)

# Save a matplotlib figure as .png
fig, ax = plt.subplots()
log.save_plot(fig)

# You can also save (somewhat) arbitrary objects .pkl
some_dict = {"hi" : "there"}
log.save_extra(some_dict)
```


Or do everything in a single line...
```python
log.update(time_tic, stats_tic, model, fig, extra, save=True)
```

### File Structure & Re-Loading :books:

For visualization and post-processing load the results via
```python
from mle_toolbox import load_log
log_out = load_log("experiment_dir/")

# The results can be accessed via meta, stats and time keys
# >>> log_out.meta.keys()
# odict_keys(['experiment_dir', 'extra_storage_paths', 'fig_storage_paths', 'log_paths', 'model_ckpt', 'model_type'])
# >>> log_out.stats.keys()
# odict_keys(['test_loss', 'train_loss'])
# >>> log_out.time.keys()
# odict_keys(['num_epochs', 'num_updates', 'time_elapsed'])
```

The `MLELogger` will create a nested directory, which looks as follows:

```
experiment_dir
├── extra: Stores saved .pkl object files
├── figures: Stores saved .png figures
├── logs: Stores .hdf5 log files (meta, stats, time)
├── models: Stores different model checkpoints
    ├── final: Stores most recent checkpoint
    ├── every_k: Stores every k-th checkpoint provided in update
    ├── top_k: Stores portfolio of top-k checkpoints based on performance
├── tboards: Stores tensorboards for model checkpointing
├── <config_name>.json: Copy of configuration file (if provided)
```


## Installation :memo:

A PyPI installation is available via:

```
pip install mle-logging
```

Alternatively, you can clone this repository and afterwards 'manually' install it:

```
git clone https://github.com/RobertTLange/mle-logging.git
cd mle-logging
pip install -e .
```


## Advanced Options :mage:

#### Merging Multiple Logs :couple:

**Merging Multiple Random Seeds** :seedling: + :seedling:

```python
```

**Merging Multiple Configurations** :bookmark: + :bookmark:

```python
```

#### Storing Checkpoint Portfolios :file_cabinet:

**Logging every k-th checkpoint update** :exclamation: :arrow_right: ... :arrow_right: :exclamation:

```python
```

**Logging top-k checkpoints based on metric** :1st_place_medal:, :2nd_place_medal:, :3rd_place_medal:

```python
```

## Development

If you find a bug or want a new feature, feel free to contact me [@RobertTLange](https://twitter.com/RobertTLange) or create an issue :hugs:


## Milestones for Next Release
- [ ] Add reloading of previous log (seamless continuation)
- [ ] Add merging utilities
- [ ] Add simple aggregation of multiple logs
- [ ] No mean/stats for time variables when aggregating multiple seeds

- [ ] Add transformations of time series
    - [ ] Running means
    - [ ] Smoothing of different degrees
- [ ] Refactor logger into smaller pieces & test
    - [ ] Save model
    - [ ] Save figure
    - [ ] Save extra
    - [ ] Save log
- [ ] Add rich print table messages for updates
- [ ] Make logger more verbose (if desired)
- [ ] Add tensorflow model saving
- [ ] Clean Up Tensorboard logging (for JAX/TF models)
- [ ] Add Weights and Biases Support
