import os
from typing import List, Union
import h5py
import numpy as np


def merge_hdf5_files(
    new_filename: str,
    log_paths: List[str],
    file_ids: Union[None, List[str]] = None,
    delete_files: bool = False,
) -> None:
    """Merges a set of hdf5 files into a new hdf5 file with more groups."""
    file_to = h5py.File(new_filename, "w")
    for i, log_p in enumerate(log_paths):
        file_from = h5py.File(log_p, "r")
        datasets = get_datasets("/", file_from)
        if file_ids is None:
            write_data_to_file(file_to, file_from, datasets)
        else:
            # Maintain unique config id even if they have same random seed
            write_data_to_file(file_to, file_from, datasets, file_ids[i])
        file_from.close()

        # Delete individual log file if desired
        if delete_files:
            os.remove(log_p)
    file_to.close()


def get_datasets(key: str, archive: h5py.File):
    """Collects different paths to datasets in recursive fashion."""
    if key[-1] != "/":
        key += "/"
    out = []
    for name in archive[key]:
        path = key + name
        if isinstance(archive[path], h5py.Dataset):
            out += [path]
        else:
            out += get_datasets(path, archive)
    return out


def write_data_to_file(
    file_to: h5py.File,
    file_from: h5py.File,
    datasets: List[str],
    file_id: Union[str, None] = None,
):
    """Writes the datasets from-to file."""
    # get the group-names from the lists of datasets
    groups = list(set([i[::-1].split("/", 1)[1][::-1] for i in datasets]))
    if file_id is None:
        groups = [i for i in groups if len(i) > 0]
    else:
        groups = [i[0] + file_id + "/" + i[1:] for i in groups if len(i) > 0]

    # sort groups based on depth
    idx = np.argsort(np.array([len(i.split("/")) for i in groups]))
    groups = [groups[i] for i in idx]

    # create all groups that contain dataset that will be copied
    for group in groups:
        file_to.create_group(group)

    # copy datasets
    for path in datasets:
        # - get group name // - minimum group name // - copy data
        group = path[::-1].split("/", 1)[1][::-1]
        if len(group) == 0:
            group = "/"
        if file_id is not None:
            group_to_index = group[0] + file_id + "/" + group[1:]
        else:
            group_to_index = group
        file_from.copy(path, file_to[group_to_index])

    file_from.close()
