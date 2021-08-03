# `mle-logging`: A Lightweight Logger for ML Experiments :book:
[![Pyversions](https://img.shields.io/pypi/pyversions/mle-logging.svg?style=flat-square)](https://pypi.python.org/pypi/mle-logging)
[![Docs Latest](https://img.shields.io/badge/docs-dev-blue.svg)](https://roberttlange.github.io/mle-toolbox/logging/mle_logging/)
[![PyPI version](https://badge.fury.io/py/mle-logging.svg)](https://badge.fury.io/py/mle-logging)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

```python
from mle_logging import MLELogger

# Instantiate logging to experiment_dir
log = MLELogger(time_to_track = ['num_updates', 'num_epochs'],
                what_to_track = ['train_loss', 'test_loss'],
                experiment_dir = "experiment_dir/")

time_tic = {'num_updates': 10,
            'num_epochs': 1}
stats_tic = {'train_loss': 0.1234,
             'test_loss': 0.1235}

# Update the log with collected data & save it to .hdf5
log.update(time_tic, stats_tic)
log.save()

# Save a model (torch, sklearn, jax, numpy)
model = ...
model_type = 'torch'
log.save_model(model, model_type)

# Save a matplotlib figure as .png
fig, ax = plt.subplots()
log.save_plot(fig)

# You can also save (somewhat) arbitrary objects .pkl
some_dict = {"hi" : "there"}
log.save_extra(some_dict, "some_path.pkl")

# Or do everything in one go
log.update(time_tic, stats_tic,
           model, fig, extra,
           save=True)
```

## Generated File Structure

The `MLELogger` will create a nested directory, which looks as follows:

```
experiment_dir
├── logs:
├── models:
    ├── final:
    ├── every_k:
    ├── top_k:
├── figures:
├── extra:
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

## Development

If you find a bug or want a new feature, feel free to contact me [@RobertTLange](https://twitter.com/RobertTLange) or create an issue :hugs:


## TODO until release
- [ ] No mean/stats for time variables when aggregating multiple seeds
- [ ] Rename `what_to_track` to `stats_to_track`
- [ ] Allow for different numbers of updates for different variables (e.g. test loss will be constructed less frequent)
- [ ] Add reloading utilities
- [ ] Add merging utilities
- [ ] Add simple aggregation of multiple logs
- [ ] Add tests
- [ ] Add transformations of time series
    - [ ] Running means
    - [ ] Smoothing of different degrees
- [ ] Refactor logger into smaller pieces
    - [ ] Save model
- [ ] Write simple tests
- [ ] Add rich print table messages
- [ ] Allow model_type as input to save_model
- [ ] Allow explicit plot name when saving (not only extension)
- [ ] Add figure saving to update (right now only tboard)
- [ ] Add extra object saving to update
- [ ] Add tensorflow model saving
- [ ] Clean Up Tensorboard logging
- [ ] Add Weights and Biases Support
