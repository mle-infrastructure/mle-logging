import time
import datetime
import pandas as pd
import collections
from typing import List, Dict
from ..load import load_log


class StatsLog(object):
    """Time-Series Statistics Logger Class Instance."""

    def __init__(
        self,
        experiment_dir: str,
        seed_id: str,
        time_to_track: List[str],
        what_to_track: List[str],
        reload: bool = False,
    ):
        self.experiment_dir = experiment_dir
        self.seed_id = seed_id
        # Create empty dataframes to log statistics in
        self.time_to_track = ["time"] + time_to_track + ["time_elapsed"]
        self.what_to_track = what_to_track
        self.clock_to_track = pd.DataFrame(columns=self.time_to_track)
        self.stats_to_track = pd.DataFrame(columns=self.what_to_track)

        # Set update counter & start stop-watch/clock of experiment
        if reload:
            self.reload()
        else:
            self.stats_update_counter = 0
        # Regardless of reloading - start time counter at 0
        self.start_time = time.time()

    def extend_tracking(self, add_track_vars: List[str]) -> None:
        """Add string names of variables to track."""
        assert self.stats_update_counter == 0
        self.what_to_track += add_track_vars
        self.stats_to_track = pd.DataFrame(columns=self.what_to_track)

    def update(self, clock_tick: Dict[str, int], stats_tick: Dict[str, float]) -> None:
        # Check all keys do exist in data dicts to log [exclude time_elapsed]
        for k in self.time_to_track[1:-1]:
            assert k in clock_tick.keys(), f"{k} not in clock_tick keys."
        for k in self.what_to_track:
            assert k in stats_tick.keys(), f"{k} not in stats_tick keys."

        # Transform clock_tick, stats_tick lists into pd arrays
        timestr = datetime.datetime.today().strftime("%y-%m-%d/%H:%M")
        c_tick = pd.DataFrame(columns=self.time_to_track)
        c_tick.loc[0] = (
            [timestr]
            + [clock_tick[k] for k in self.time_to_track[1:-1]]
            + [time.time() - self.start_time]
        )
        s_tick = pd.DataFrame(columns=self.what_to_track)
        s_tick.loc[0] = [stats_tick[k] for k in self.what_to_track]

        # Append time tick & results to pandas dataframes
        self.clock_to_track = pd.concat(
            [self.clock_to_track, c_tick], axis=0
        ).reset_index(drop=True)
        self.stats_to_track = pd.concat(
            [self.stats_to_track, s_tick], axis=0
        ).reset_index(drop=True)

        # Tick up the update counter
        self.stats_update_counter += 1
        return c_tick, s_tick

    def reload(self):
        """Reload results from previous experiment run."""
        reloaded_log = load_log(self.experiment_dir)
        # Make sure to reload in results for correct seed
        if reloaded_log.eval_ids is None:
            self.clock_to_track = pd.DataFrame(reloaded_log.time)
            self.stats_to_track = pd.DataFrame(reloaded_log.stats)
        else:
            self.clock_to_track = pd.DataFrame(reloaded_log[self.seed_id].time)
            self.stats_to_track = pd.DataFrame(reloaded_log[self.seed_id].stats)
        # Check that all required stats keys/column names are present
        assert collections.Counter(self.clock_to_track.columns) == collections.Counter(
            self.time_to_track
        )
        assert collections.Counter(self.stats_to_track.columns) == collections.Counter(
            self.what_to_track
        )
        self.stats_update_counter = self.stats_to_track.shape[0]
