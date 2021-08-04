import os
from .merge_hdf5 import merge_hdf5_files
from ..load import load_meta_log


def merge_meta_log(experiment_dir: str, all_run_ids: list):
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
    meta_log_fname = os.path.join(experiment_dir, "meta_log.hdf5")

    assert len(log_paths) == len(all_run_ids)

    merge_hdf5_files(meta_log_fname, log_paths, file_ids=all_run_ids)

    # Load in meta-results log with values meaned over seeds
    meta_eval_logs = load_meta_log(meta_log_fname, aggregate_seeds=True)
    return meta_eval_logs
