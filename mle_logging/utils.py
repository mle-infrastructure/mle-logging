import pickle
import pickle5
from typing import Any
import h5py
import numpy as np


def save_pkl_object(obj, filename: str) -> None:
    """Helper to store pickle objects."""
    with open(filename, "wb") as output:
        # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_pkl_object(filename: str) -> Any:
    """Helper to reload pickle objects."""
    with open(filename, "rb") as input:
        obj = pickle5.load(input)
    return obj


def write_to_hdf5(
    log_fname: str, log_path: str, data_to_log: Any, dtype: str = "S200"
) -> None:
    # Store figure paths if any where created
    if dtype == "S200":
        data_to_store = [t.encode("ascii", "ignore") for t in data_to_log]
    else:
        data_to_store = np.array(data_to_log)

    h5f = h5py.File(log_fname, "a")
    if h5f.get(log_path):
        del h5f[log_path]
    h5f.create_dataset(
        name=log_path,
        data=data_to_store,
        compression="gzip",
        compression_opts=4,
        dtype=dtype,
    )
    h5f.flush()
    h5f.close()
