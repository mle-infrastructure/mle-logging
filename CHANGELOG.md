## [v0.0.5] - [Unreleased]

### Added

- Adds new case to `MetaLog` loading: Single configuration with explicit seed.
- Adds test coverage for `comms` and `utils`.
- Adds support for storage of vector-valued stats in `log.update`.
- Adds wandb backend. Based on options `use_wandb` and `wandb_config`.

### Changed

- Changes and reduces requirements

### Fixed

- Fixes different data types issues (#3)
- Fixes string decoding and merging for `MetaLog`
- Log aggregation for single seed/single configuration

## [v0.0.4] - [12/07/2021]

### Added

- Add plot detail options (title, labels) to `meta_log.plot()`

### Changed

- Get rid of time string in sub directories

### Fixed

- Makes log merging more robust
- Small fixes for `mle-monitor` release
- Fix `overwrite` and make verbose warning (delete `log.hdf5` for merged case)

## [v0.0.3] - [09/11/2021]

### Added

- Adds function to store initial model checkpoint for post-processing via `log.save_init_model(model)`.
- `MLELogger` got a new optional argument: `config_dict`, which allows you to provide a (nested) configuration of your experiment. It  will be stored as a `.yaml` file if you don't provide a path to an alternative configuration file. The file can either be a `.json` or a `.yaml`:

```python
log = MLELogger(time_to_track=['num_updates', 'num_epochs'],
                what_to_track=['train_loss', 'test_loss'],
                experiment_dir="experiment_dir/",
                config_dict={"train_config": {"lrate": 0.01}},
                model_type='torch',
                verbose=True)
```

- The `config_dict`/ loaded `config_fname` data will be stored in the `meta` data of the loaded log and can be easily retrieved:

```python
log = load_log("experiment_dir/")
log.meta.config_dict
```

### Fixed

- Fix byte decoding for strings stored as arrays in `.hdf5` log file. Previously this only worked for multi seed/config settings.

## [v0.0.2] - [08/23/2021]

### Added

- Enhances verbosity and nice rich layout printing.

## [v0.0.1] - [08/18/2021]

### Added

- Basic `mle-logging` API:

```python
from mle_logging import MLELogger

# Instantiate logging to experiment_dir
log = MLELogger(time_to_track=['num_updates', 'num_epochs'],
                what_to_track=['train_loss', 'test_loss'],
                experiment_dir="experiment_dir/",
                model_type='torch')

time_tic = {'num_updates': 10, 'num_epochs': 1}
stats_tic = {'train_loss': 0.1234, 'test_loss': 0.1235}

# Update the log with collected data & save it to .hdf5
log.update(time_tic, stats_tic)
log.save()
```
