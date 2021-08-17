import os
from os.path import isfile, join
from typing import Union, List


class FigureLog(object):
    """Figure Logger Class Instance."""

    def __init__(
        self,
        experiment_dir: str = "/",
        seed_id: str = "no_seed_provided",
        reload: bool = False,
    ):
        # Setup figure logging directories
        self.experiment_dir = experiment_dir
        self.figures_dir = os.path.join(self.experiment_dir, "figures/")
        self.seed_id = seed_id

        # Reload filenames and counter from previous execution
        if reload:
            self.reload()
        else:
            self.fig_save_counter = 0
            self.fig_storage_paths: List[str] = []

    def save(self, fig, fig_fname: Union[str, None] = None) -> None:
        """Store a matplotlib figure."""
        # Create new directory to store figures - if it doesn't exist yet
        self.fig_save_counter += 1
        if self.fig_save_counter == 1:
            os.makedirs(self.figures_dir, exist_ok=True)

        # Tick up counter, save figure, store new path to figure
        if fig_fname is None:
            figure_fname = os.path.join(
                self.figures_dir,
                "fig_" + str(self.fig_save_counter) + "_" + str(self.seed_id) + ".png",
            )
        else:
            self.fig_save_counter -= 1
            figure_fname = os.path.join(
                self.figures_dir,
                fig_fname,
            )

        fig.savefig(figure_fname, dpi=300)
        self.fig_storage_paths.append(figure_fname)

    def reload(self):
        """Reload results from previous experiment run."""
        # Go into figures directory, get list of figure files and set counter
        try:
            fig_paths = [
                join(self.figures_dir, f)
                for f in os.listdir(self.figures_dir)
                if isfile(join(self.figures_dir, f))
            ]
            self.fig_storage_paths = [
                f for f in fig_paths if f.endswith(str(self.seed_id) + ".png")
            ]
            self.fig_save_counter = len(self.fig_storage_paths)
        except FileNotFoundError:
            self.fig_save_counter = 0
            self.fig_storage_paths: List[str] = []
