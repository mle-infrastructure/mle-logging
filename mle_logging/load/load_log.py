import os
import h5py
from dotmap import DotMap
import collections
from ..meta_log import MetaLog


def load_meta_log(log_fname: str, aggregate_seeds: bool = True) -> MetaLog:
    """Load in logging results & mean the results over different runs"""
    assert os.path.exists(log_fname), f"File {log_fname} does not exist."
    # Open File & Get array names to load in
    h5f = h5py.File(log_fname, mode="r", swmr=True)
    # Get all ids of all runs (b_1_eval_0, b_1_eval_1, ...)
    run_names = list(h5f.keys())
    # Get all main data source keys (single vs multi-seed)
    data_sources = list(h5f[run_names[0]].keys())
    data_types = ["meta", "stats", "time"]

    """
    3 Possible Cases:
    1. Single config - single seed = no aggregation - 'no_seed_provided'
    2. Single config - multi seed = aggregation - seed_id -> meta, stats, time
    3. Multi config - multi seed = aggregation - config_id -> seed_id -> ...
    """
    case_1 = len(run_names) == 1 and collections.Counter(
        h5f[run_names[0]].keys()
    ) == collections.Counter(data_types)
    case_2 = len(run_names) > 1 and collections.Counter(
        h5f[run_names[0]].keys()
    ) == collections.Counter(data_types)
    case_3 = len(run_names) > 1 and collections.Counter(
        h5f[run_names[0]].keys()
    ) != collections.Counter(data_types)

    result_dict = {key: {} for key in run_names}
    # Shallow versus deep aggregation
    if case_1 or case_2:
        data_items = {
            data_types[i]: list(h5f[run_names[0]][data_types[i]].keys())
            for i in range(len(data_types))
        }
        for rn in run_names:
            run = h5f[rn]
            source_to_store = {key: {} for key in data_types}
            for ds in data_items:
                data_to_store = {key: {} for key in data_items[ds]}
                for i, o_name in enumerate(data_items[ds]):
                    data_to_store[o_name] = run[ds][o_name][:]
                source_to_store[ds] = data_to_store
            result_dict[rn] = source_to_store
    else:
        data_items = {
            data_types[i]: list(
                h5f[run_names[0]][data_sources[0]][data_types[i]].keys()
            )
            for i in range(len(data_types))
        }
        for rn in run_names:
            run = h5f[rn]
            result_dict[rn] = {}
            for seed_id in data_sources:
                source_to_store = {key: {} for key in data_types}
                for ds in data_items:
                    data_to_store = {key: {} for key in data_items[ds]}
                    for i, o_name in enumerate(data_items[ds]):
                        data_to_store[o_name] = run[seed_id][ds][o_name][:]
                    source_to_store[ds] = data_to_store
                result_dict[rn][seed_id] = source_to_store

    # Return as dot-callable dictionary
    if aggregate_seeds and (case_2 or case_3):
        # Important aggregation helper & compute mean/median/10p/50p/etc.
        from ..merge.aggregate import aggregate_over_seeds

        result_dict = aggregate_over_seeds(result_dict, batch_case=case_3)
    return MetaLog(
        DotMap(result_dict, _dynamic=False),
        non_aggregated=(not aggregate_seeds and case_3),
    )


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
