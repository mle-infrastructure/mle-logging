from dotmap import DotMap
import numpy as np
from typing import List, Tuple, Any


def aggregate_over_seeds(result_dict: DotMap) -> DotMap:
    """Mean all individual runs over their respective seeds.
    BATCH EVAL CASE:
    IN: {'b_1_eval_0_seed_0': {'meta': {}, 'stats': {}, 'time': {}},
         'b_1_eval_0_seed_1': {'meta': {}, 'stats': {}, 'time': {}},
          ...}
    OUT: {'b_1_eval_0': {'meta': {}, 'stats': {}, 'time': {},
          'b_1_eval_1': {'meta': {}, 'stats': {}, 'time': {}}
    SINGLE EVAL CASE:
    IN: {'seed_0': {'meta': {}, 'stats': {}, 'time': {}},
         'seed_1': {'meta': {}, 'stats': {}, 'time': {}},
          ...}
    OUT: {'eval': {'meta': {}, 'stats': {}, 'time': {}}
    """
    all_runs = list(result_dict.keys())
    eval_runs = []
    split_by = "_seed_"

    # Get the different unique runs (without their seeds)
    for run in all_runs:
        split = run.split(split_by)
        if len(split) > 1:
            eval_runs.append(split[0])
        else:
            # Break if there are only different seeds - single config!
            break
    unique_evals = list(set(eval_runs))

    if len(unique_evals) > 0:
        # Get seeds specific to each eval/run - append later on to meta data
        evals_and_seeds = {key: [] for key in unique_evals}
        for run in all_runs:
            split = run.split(split_by)
            evals_and_seeds[split[0]].append(int(split[1]))
        # Perform seed aggregation for all evaluations
        new_results_dict = aggregate_batch_evals(
            result_dict, unique_evals, evals_and_seeds, all_runs
        )
    else:
        new_results_dict = aggregate_single_eval(result_dict, all_runs, "eval")
    return DotMap(new_results_dict, _dynamic=False)


def aggregate_single_eval(  # noqa: C901
    result_dict: dict, all_seeds_for_run: list, eval_name: str
) -> dict:
    """Mean over seeds of single config run."""
    new_results_dict = {}
    data_temp = result_dict[all_seeds_for_run[0]]
    # Get all main data source keys ("meta", "stats", "time")
    data_sources = list(data_temp.keys())
    # Get all variables within the data sources
    data_items = {
        data_sources[i]: list(data_temp[data_sources[i]].keys())
        for i in range(len(data_sources))
    }
    # Collect all runs together - data at this point is not modified
    source_to_store = {key: {} for key in data_sources}
    for ds in data_sources:
        data_to_store = {key: [] for key in data_items[ds]}
        for i, o_name in enumerate(data_items[ds]):
            for i, seed_id in enumerate(all_seeds_for_run):
                seed_run = result_dict[seed_id]
                data_to_store[o_name].append(seed_run[ds][o_name][:])
        source_to_store[ds] = data_to_store
    new_results_dict[eval_name] = source_to_store

    # Aggregate over the collected runs
    aggregate_sources = {key: {} for key in data_sources}
    for ds in data_sources:
        # Mean over time and stats data
        if ds in ["time", "stats"]:
            aggregate_dict = {key: {} for key in data_items[ds]}
            for i, o_name in enumerate(data_items[ds]):
                if type(new_results_dict[eval_name][ds][o_name][0][0]) not in [
                    str,
                    bytes,
                    np.bytes_,
                    np.str_,
                ]:
                    # Compute mean and standard deviation over seeds
                    mean_tol, std_tol = tolerant_mean(
                        new_results_dict[eval_name][ds][o_name]
                    )
                    aggregate_dict[o_name]["mean"] = mean_tol
                    aggregate_dict[o_name]["std"] = std_tol

                    # Compute 10, 25, 50, 75, 90 percentiles over seeds
                    p50, p10, p25, p75, p90 = tolerant_median(
                        new_results_dict[eval_name][ds][o_name]
                    )
                    aggregate_dict[o_name]["p50"] = p50
                    aggregate_dict[o_name]["p10"] = p10
                    aggregate_dict[o_name]["p25"] = p25
                    aggregate_dict[o_name]["p75"] = p75
                    aggregate_dict[o_name]["p90"] = p90
                else:
                    aggregate_dict[o_name] = new_results_dict[eval_name][ds][o_name]
        # Append over all meta data (strings, seeds nothing to mean)
        elif ds == "meta":
            aggregate_dict = {}
            for i, o_name in enumerate(data_items[ds]):
                temp = (
                    np.array(new_results_dict[eval_name][ds][o_name])
                    .squeeze()
                    .astype("U200")
                )
                # Get rid of duplicate experiment dir strings
                if o_name in [
                    "experiment_dir",
                    "eval_id",
                    "config_fname",
                    "model_type",
                ]:
                    aggregate_dict[o_name] = str(np.unique(temp)[0])
                else:
                    aggregate_dict[o_name] = temp

            # Add seeds as clean array of integers to dict
            aggregate_dict["seeds"] = [int(s.split("_")[1]) for s in all_seeds_for_run]
        else:
            raise ValueError
        aggregate_sources[ds] = aggregate_dict
    new_results_dict[eval_name] = aggregate_sources
    return new_results_dict


def aggregate_batch_evals(
    result_dict: dict, unique_evals: list, evals_and_seeds: list, all_runs: list
) -> dict:
    """Mean over seeds for all batches and evals."""
    # Loop over all evals (e.g. b_1_eval_0) and merge + aggregate data
    new_results_dict = {}
    for eval in unique_evals:
        all_seeds_for_run = [i for i in all_runs if i.startswith(eval + "_")]
        eval_dict = aggregate_single_eval(result_dict, all_seeds_for_run, eval)
        new_results_dict[eval] = eval_dict[eval]
    return new_results_dict


def tolerant_mean(arrs: List[Any]) -> Tuple[Any]:
    """Helper function for case where data to mean has different lengths."""
    lens = [len(i) for i in arrs]
    if len(arrs[0].shape) == 1:
        arr = np.ma.empty((np.max(lens), len(arrs)))
        arr.mask = True
        for idx, l in enumerate(arrs):
            arr[: len(l), idx] = l
    else:
        arr = np.ma.empty((np.max(lens), arrs[0].shape[1], len(arrs)))
        arr.mask = True
        for idx, l in enumerate(arrs):
            arr[: len(l), :, idx] = l
    return arr.mean(axis=-1), arr.std(axis=-1)


def tolerant_median(arrs: List[Any]) -> Tuple[Any]:
    """Helper function for case data to median has different lengths."""
    lens = [len(i) for i in arrs]
    if len(arrs[0].shape) == 1:
        arr = np.ma.empty((np.max(lens), len(arrs)))
        arr.mask = True
        for idx, l in enumerate(arrs):
            arr[: len(l), idx] = l
    else:
        arr = np.ma.empty((np.max(lens), arrs[0].shape[1], len(arrs)))
        arr.mask = True
        for idx, l in enumerate(arrs):
            arr[: len(l), :, idx] = l
    return (
        np.percentile(arr, 50, axis=-1),
        np.percentile(arr, 10, axis=-1),
        np.percentile(arr, 25, axis=-1),
        np.percentile(arr, 75, axis=-1),
        np.percentile(arr, 90, axis=-1),
    )
