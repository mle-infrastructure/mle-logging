import os
import time
from typing import Union
from .merge_hdf5 import merge_hdf5_files


def merge_seed_logs(
    merged_path: str,
    experiment_dir: str,
    num_logs: Union[int, None] = None,
    delete_files: bool = True,
) -> None:
    """Merge all .hdf5 files for different seeds into single log."""
    # Collect paths in log dir until the num_logs is found
    log_dir = os.path.join(experiment_dir, "logs")
    while True:
        log_paths = [os.path.join(log_dir, log) for log in os.listdir(log_dir)]
        if num_logs is not None:
            if len(log_paths) == num_logs:
                # Delete joined log if at some point over-eagerly merged
                if merged_path in log_paths:
                    os.remove(merged_path)
                break
            else:
                time.sleep(1)
        else:
            break
    merge_hdf5_files(merged_path, log_paths, delete_files=delete_files)


def merge_config_logs(experiment_dir: str, all_run_ids: list) -> None:
    """Scavenge the experiment dictonaries & load in logs."""
    all_folders = [x[0] for x in os.walk(experiment_dir)][1:]
    # Get rid of timestring in beginning & collect all folders/hdf5 files
    hyperp_results_folder = []
    # Need to make sure that run_ids & experiment folder match!
    for run_id in all_run_ids:
        for f in all_folders:
            if f[len(experiment_dir) + 9 :] == run_id:
                hyperp_results_folder.append(f)
                continue

    # Collect all paths to the .hdf5 file
    log_paths = []
    for i in range(len(hyperp_results_folder)):
        log_d_t = os.path.join(hyperp_results_folder[i], "logs/")
        for file in os.listdir(log_d_t):
            fname, fext = os.path.splitext(file)
            if file.endswith(".hdf5") and fname in all_run_ids:
                log_paths.append(os.path.join(log_d_t, file))

    # Merge individual run results into a single hdf5 file
    assert len(log_paths) == len(all_run_ids)
    meta_log_fname = os.path.join(experiment_dir, "meta_log.hdf5")
    merge_hdf5_files(meta_log_fname, log_paths, file_ids=all_run_ids)
