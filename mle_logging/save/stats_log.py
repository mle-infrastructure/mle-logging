import time
import datetime
import pandas as pd
from typing import List, Dict


class StatsLog(object):
    """Time-Series Statistics Logger Class Instance."""

    def __init__(
        self,
        experiment_dir: str,
        base_str: str,
        seed_id: str,
        time_to_track: List[str],
        what_to_track: List[str],
    ):
        # Create empty dataframes to log statistics in
        self.time_to_track = ["time"] + time_to_track + ["time_elapsed"]
        self.what_to_track = what_to_track
        self.clock_to_track = pd.DataFrame(columns=self.time_to_track)
        self.stats_to_track = pd.DataFrame(columns=self.what_to_track)

        # Set update counter
        self.stats_update_counter = 0

        # Start stop-watch/clock of experiment
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
        timestr = datetime.datetime.today().strftime("%m-%d|%H:%M:%S")
        c_tick = pd.DataFrame(columns=self.time_to_track)
        c_tick.loc[0] = (
            [timestr]
            + [clock_tick[k] for k in self.time_to_track[1:-1]]
            + [time.time() - self.start_time]
        )
        s_tick = pd.DataFrame(columns=self.what_to_track)
        s_tick.loc[0] = [stats_tick[k] for k in self.stats_to_track]

        # Append time tick & results to pandas dataframes
        self.clock_to_track = pd.concat([self.clock_to_track, c_tick], axis=0)
        self.stats_to_track = pd.concat([self.stats_to_track, s_tick], axis=0)

        # Tick up the update counter
        self.stats_update_counter += 1
        return c_tick, s_tick
