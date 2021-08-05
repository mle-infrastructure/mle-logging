import os
import h5py
from dotmap import DotMap
from ..meta_log import MetaLog


def load_meta_log(log_fname: str, aggregate_seeds: bool = True) -> MetaLog:
    """Load in logging results & mean the results over different runs"""
    assert os.path.exists(log_fname), f"File {log_fname} does not exist."
    # Open File & Get array names to load in
    h5f = h5py.File(log_fname, mode="r", swmr=True)
    # Get all ids of all runs (b_1_eval_0_seed_0)
    run_names = list(h5f.keys())
    # Get only stem of ids (b_1_eval_0)
    # run_ids = list(set([r_n.split("_seed")[0] for r_n in run_names]))
    # Get all main data source keys ("meta", "stats", "time")
    data_sources = list(h5f[run_names[0]].keys())
    # Get all variables within the data sources
    data_items = {
        data_sources[i]: list(h5f[run_names[0]][data_sources[i]].keys())
        for i in range(len(data_sources))
    }

    # Create a group for each runs (eval and seed)
    # Out: {'b_1_eval_0_seed_0': {'meta': {}, 'stats': {}, 'time': {}}, ...}
    result_dict = {key: {} for key in run_names}
    for rn in run_names:
        run = h5f[rn]
        source_to_store = {key: {} for key in data_sources}
        for ds in data_sources:
            data_to_store = {key: {} for key in data_items[ds]}
            for i, o_name in enumerate(data_items[ds]):
                data_to_store[o_name] = run[ds][o_name][:]
            source_to_store[ds] = data_to_store
        result_dict[rn] = source_to_store
    # h5f.close()

    # Return as dot-callable dictionary
    if aggregate_seeds:
        # Important aggregation helper & compute mean/median/10p/50p/etc.
        from ..merge.aggregate import aggregate_over_seeds
        result_dict = aggregate_over_seeds(result_dict)
    return MetaLog(DotMap(result_dict, _dynamic=False))


def load_log(experiment_dir: str, aggregate_seeds: bool = False) -> MetaLog:
    """Load a single .hdf5 log from <experiment_dir>/logs."""
    if experiment_dir.endswith(".hdf5"):
        log_path = experiment_dir
    else:
        log_dir = os.path.join(experiment_dir, "logs/")
        log_paths = []
        for file in os.listdir(log_dir):
            if file.endswith(".hdf5"):
                log_paths.append(os.path.join(log_dir, file))
        if len(log_paths) > 1:
            print(f"Multiple .hdf5 files available: {log_paths}")
            print(f"Continue using: {log_paths[0]}")
        log_path = log_paths[0]
    run_log = load_meta_log(log_path, aggregate_seeds)
    return run_log
