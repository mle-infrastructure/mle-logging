import os
from os.path import isfile, join
from typing import Union, List
from ..utils import save_pkl_object


class ExtraLog(object):
    """Extra .pkl Object Logger Class Instance."""

    def __init__(
        self,
        experiment_dir: str = "/",
        seed_id: str = "no_seed_provided",
        reload: bool = False,
    ):
        # Setup extra logging directories
        self.experiment_dir = experiment_dir
        self.extra_dir = os.path.join(self.experiment_dir, "extra/")
        self.seed_id = seed_id

        # Reload filenames and counter from previous execution
        if reload:
            self.reload()
        else:
            self.extra_save_counter = 0
            self.extra_storage_paths: List[str] = []

    def save(self, obj, obj_fname: Union[str, None] = None):
        """Store a .pkl object."""
        # Create new directory to store objects - if it doesn't exist yet
        self.extra_save_counter += 1
        if self.extra_save_counter == 1:
            os.makedirs(self.extra_dir, exist_ok=True)

        # Tick up counter, save figure, store new path to figure
        if obj_fname is None:
            obj_fname = os.path.join(
                self.extra_dir,
                "extra_"
                + str(self.extra_save_counter)
                + "_"
                + str(self.seed_id)
                + ".pkl",
            )
        else:
            self.extra_save_counter -= 1
            obj_fname = os.path.join(
                self.extra_dir,
                obj_fname,
            )

        save_pkl_object(obj, obj_fname)
        self.extra_storage_paths.append(obj_fname)

    def reload(self):
        """Reload results from previous experiment run."""
        # Go into extra directory, get list of files and set counter
        try:
            extra_paths = [
                join(self.extra_dir, f)
                for f in os.listdir(self.extra_dir)
                if isfile(join(self.extra_dir, f))
            ]
            self.extra_storage_paths = [
                f for f in extra_paths if f.endswith(str(self.seed_id) + ".pkl")
            ]
            self.extra_save_counter = len(self.extra_storage_paths)
        except FileNotFoundError:
            self.extra_save_counter = 0
            self.extra_storage_paths: List[str] = []
