import os
from typing import Union, List


class FigureLog(object):
    """Figure Logger Class Instance."""

    def __init__(
        self,
        experiment_dir: str = "/",
        seed_id: str = "no_seed_provided",
    ):
        # Setup figure logging directories
        self.experiment_dir = experiment_dir
        self.figures_dir = os.path.join(self.experiment_dir, "figures/")
        self.fig_save_counter = 0
        self.fig_storage_paths: List[str] = []
        self.seed_id = seed_id

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
