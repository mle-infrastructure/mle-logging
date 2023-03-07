import time
from datetime import datetime
from typing import List, Dict, Union
from ..load import load_log


class StatsLog(object):
    """Time-Series Statistics Logger Class Instance."""

    def __init__(
        self,
        experiment_dir: str,
        seed_id: str,
        time_to_track: List[str] = [],
        what_to_track: List[str] = [],
        reload: bool = False,
        freeze_keys: bool = False,  # Freeze keys that are stored in time/stats
    ):
        self.experiment_dir = experiment_dir
        self.seed_id = seed_id
        # Create empty dataframes to log statistics in
        self.time_to_track = ["time", "time_elapsed", "num_updates"] + time_to_track
        self.what_to_track = what_to_track
        self.clock_tracked = {k: [] for k in self.time_to_track}
        self.stats_tracked = {k: [] for k in self.what_to_track}
        self.freeze_keys = freeze_keys
        # Set update counter & start stop-watch/clock of experiment
        if reload:
            self.reload()
        else:
            self.stats_update_counter = 0
        # Regardless of reloading - start time counter at 0
        self.start_time = time.time()

    def extend_tracking(
        self,
        stats_keys: Union[List[str], None] = None,
        time_keys: Union[List[str], None] = None,
    ) -> None:
        """Add string names of variables to track."""
        if stats_keys is not None:
            self.what_to_track += stats_keys
            for k in stats_keys:
                self.stats_tracked[k] = []
        if time_keys is not None:
            self.time_to_track += time_keys
            for k in time_keys:
                self.clock_tracked[k] = []

    def update(self, clock_tick: Dict[str, int], stats_tick: Dict[str, float]) -> None:
        # Check all keys do exist in data dicts to log [exclude time time_elapsed num_updates]
        if self.freeze_keys:
            for k in self.time_to_track[3:]:
                assert k in clock_tick.keys(), f"{k} not in clock_tick keys."
            for k in self.what_to_track:
                assert k in stats_tick.keys(), f"{k} not in stats_tick keys."
        else:
            # Update time logged first
            self.stats_update_counter += 1
            clock_tick["time"] = datetime.today().strftime("%y-%m-%d/%H:%M")
            clock_tick["time_elapsed"] = time.time() - self.start_time
            clock_tick["num_updates"] = self.stats_update_counter

            for k in clock_tick.keys():
                if k in self.time_to_track:
                    self.clock_tracked[k].append(clock_tick[k])
                else:
                    self.time_to_track.append(k)
                    self.clock_tracked[k] = [clock_tick[k]]

            # Update stats logged next
            for k in stats_tick.keys():
                if k in self.what_to_track:
                    self.stats_tracked[k].append(stats_tick[k])
                else:
                    self.what_to_track.append(k)
                    self.stats_tracked[k] = [stats_tick[k]]

            return clock_tick, stats_tick

    def reload(self):
        """Reload results from previous experiment run."""
        reloaded_log = load_log(self.experiment_dir,
                                aggregate_seeds=False,
                                reload_log=True)
        self.clock_tracked, self.stats_tracked = {}, {}
        self.what_to_track, self.time_to_track = [], []
        # Make sure to reload in results for correct seed
        if reloaded_log.eval_ids[0] == "no_seed_provided":
            for k in reloaded_log["no_seed_provided"].time.keys():
                self.time_to_track.append(k)
                self.clock_tracked[k] = reloaded_log["no_seed_provided"].time[k].tolist()
            for k in reloaded_log["no_seed_provided"].stats.keys():
                self.what_to_track.append(k)
                self.stats_tracked[k] = reloaded_log["no_seed_provided"].stats[k].tolist()
        else:
            for k in reloaded_log[self.seed_id].time.keys():
                self.time_to_track.append(k)
                self.clock_tracked[k] = reloaded_log[self.seed_id].time[k].tolist()
            for k in reloaded_log[self.seed_id].stats.keys():
                self.what_to_track.append(k)
                self.stats_tracked[k] = reloaded_log[self.seed_id].stats[k].tolist()
        self.stats_update_counter = self.clock_tracked["num_updates"][-1]
