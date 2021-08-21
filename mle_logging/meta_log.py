from dotmap import DotMap
from typing import Union, List
from .utils import visualize_1D_lcurves


class MetaLog(object):
    """Class wrapper for meta_log dictionary w. additional functionality."""

    meta_vars: List[str]
    stats_vars: List[str]
    time_vars: List[str]
    num_configs: int

    def __init__(self, meta_log: DotMap, non_aggregated: bool = False):
        self.meta_log = meta_log

        # Return shallow log if there is only a single experiment stored
        self.num_configs = len(list(meta_log.keys()))
        ph_run = list(meta_log.keys())[0]
        # Extract different variable names from meta log
        if not non_aggregated:
            self.meta_vars = list(meta_log[ph_run].meta.keys())
            self.stats_vars = list(meta_log[ph_run].stats.keys())
            self.time_vars = list(meta_log[ph_run].time.keys())
        else:
            ph_seed = list(meta_log[ph_run].keys())[0]
            self.meta_vars = list(meta_log[ph_run][ph_seed].meta.keys())
            self.stats_vars = list(meta_log[ph_run][ph_seed].stats.keys())
            self.time_vars = list(meta_log[ph_run][ph_seed].time.keys())

        # Make log shallow if there is only a single experiment stored
        if self.num_configs == 1:
            self.meta_log = self.meta_log[ph_run]

        # Make possible that all runs are accessible via attribute as in pd
        for key in self.meta_log:
            setattr(self, key, self.meta_log[key])

    def filter(self, run_ids: List[str]):
        """Subselect the meta log dict based on a list of run ids."""
        sub_dict = subselect_meta_log(self.meta_log, run_ids)
        return MetaLog(sub_dict)

    def plot(
        self,
        target_to_plot: str,
        iter_to_plot: Union[str, None] = None,
        fig=None,
        ax=None,
    ) -> None:
        """Plot all runs in meta-log for variable 'target_to_plot'."""
        if iter_to_plot is None:
            iter_to_plot = self.time_vars[0]
        assert iter_to_plot in self.time_vars
        fig, ax = visualize_1D_lcurves(
            self.meta_log,
            iter_to_plot,
            target_to_plot,
            smooth_window=1,
            every_nth_tick=None,
            num_legend_cols=2,
            run_ids=self.eval_ids,
            fig=fig,
            ax=ax,
        )
        return fig, ax

    @property
    def eval_ids(self) -> Union[int, None]:
        """Get ids of runs stored in meta_log instance."""
        if self.num_configs > 1:
            return list(self.meta_log.keys())
        # else:
        #     print("Only single aggregated configuration or random seed loaded.")

    def __len__(self) -> int:
        """Return number of runs stored in meta_log."""
        return len(self.eval_ids)

    def __getitem__(self, item):
        """Get run log via string subscription."""
        return self.meta_log[item]


def subselect_meta_log(meta_log: DotMap, run_ids: List[str]) -> DotMap:
    """Subselect the meta log dict based on a list of run ids."""
    sub_log = DotMap()
    for run_id in run_ids:
        sub_log[run_id] = meta_log[run_id]
    return sub_log
