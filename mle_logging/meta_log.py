import ast
from dotmap import DotMap
from typing import Union, List
from .utils import visualize_1D_lcurves


class MetaLog(object):
    meta_vars: List[str]
    stats_vars: List[str]
    time_vars: List[str]
    num_configs: int

    def __init__(self, meta_log: DotMap, non_aggregated: bool = False):
        """Class wrapper for meta_log dictionary w. additional functionality.

        Args:
            meta_log (DotMap): Raw reloaded meta-log dotmap dictionary.
            non_aggregated (bool, optional):
                Whether the meta-log has previously been aggregated across
                seeds. Defaults to False.
        """
        self.meta_log = meta_log

        # Return shallow log if there is only a single experiment stored
        self.num_configs = len(list(meta_log.keys()))
        ph_run = list(meta_log.keys())[0]
        ph_seed = list(meta_log[ph_run].keys())[0]

        # Extract different variable names from meta log
        if not non_aggregated and ph_seed in ["meta", "stats", "time"]:
            self.meta_vars = list(meta_log[ph_run].meta.keys())
            self.stats_vars = list(meta_log[ph_run].stats.keys())
            self.time_vars = list(meta_log[ph_run].time.keys())
        else:
            self.meta_vars = list(meta_log[ph_run][ph_seed].meta.keys())
            self.stats_vars = list(meta_log[ph_run][ph_seed].stats.keys())
            self.time_vars = list(meta_log[ph_run][ph_seed].time.keys())

        # Decode all byte strings in meta data
        for run_id in self.meta_log.keys():
            if "meta" in self.meta_log[run_id].keys():
                try:
                    self.meta_log[run_id] = decode_meta_strings(
                        self.meta_log[run_id]
                    )
                except Exception:
                    pass
            else:
                for seed_id in self.meta_log[run_id].keys():
                    self.meta_log[run_id][seed_id] = decode_meta_strings(
                        self.meta_log[run_id][seed_id]
                    )

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
        smooth_window: int = 1,
        plot_title: Union[str, None] = None,
        xy_labels: Union[list, None] = None,
        base_label: str = "{}",
        run_ids: Union[list, None] = None,
        curve_labels: list = [],
        every_nth_tick: Union[int, None] = None,
        plot_std_bar: bool = False,
        fname: Union[None, str] = None,
        num_legend_cols: Union[int, None] = 1,
        fig=None,
        ax=None,
        figsize: tuple = (9, 6),
        plot_labels: bool = True,
        legend_title: Union[None, str] = None,
        ax_lims: Union[None, list] = None,
    ):
        """Plot all runs in meta-log for variable 'target_to_plot'."""
        if iter_to_plot is None:
            iter_to_plot = self.time_vars[0]
        assert iter_to_plot in self.time_vars
        if run_ids is None:
            run_ids = self.eval_ids
        fig, ax = visualize_1D_lcurves(
            self.meta_log,
            iter_to_plot,
            target_to_plot,
            smooth_window=smooth_window,
            every_nth_tick=every_nth_tick,
            num_legend_cols=num_legend_cols,
            run_ids=run_ids,
            plot_title=plot_title,
            xy_labels=xy_labels,
            base_label=base_label,
            curve_labels=curve_labels,
            plot_std_bar=plot_std_bar,
            fig=fig,
            ax=ax,
            figsize=figsize,
            plot_labels=plot_labels,
            legend_title=legend_title,
            ax_lims=ax_lims,
        )
        # Save the figure if a filename was provided
        if fname is not None:
            fig.savefig(fname, dpi=300)
        else:
            return fig, ax

    @property
    def eval_ids(self) -> Union[int, None]:
        """Get ids of runs stored in meta_log instance."""
        return list(self.meta_log.keys())

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


def decode_meta_strings(log: DotMap):
    """Decode all bytes encoded strings."""
    for k in log.meta.keys():
        temp_list = []
        if type(log.meta[k]) != str and type(log.meta[k]) != dict:
            list_to_loop = (
                log.meta[k].tolist()
                if type(log.meta[k]) != list
                else log.meta[k]
            )

            if type(list_to_loop) in [str, bytes]:
                list_to_loop = [list_to_loop]
            for i in list_to_loop:
                if type(i) == bytes:
                    if len(i) > 0:
                        temp_list.append(i.decode())
                else:
                    temp_list.append(i)
        else:
            temp_list.append(log.meta[k])

        if len(temp_list) == 1:
            if k == "config_dict":
                # Convert config into dict
                config_dict = ast.literal_eval(str(temp_list[0]))
                log.meta[k] = config_dict
            else:
                log.meta[k] = temp_list[0]
        else:
            log.meta[k] = temp_list

    return log
