import pickle
import pickle5
from typing import Any, Union
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns


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


def print_startup(
    experiment_dir: str,
    time_to_track: list,
    what_to_track: list,
    model_type: str,
    ckpt_time_to_track: Union[str, None],
    save_every_k_ckpt: Union[int, None],
    save_top_k_ckpt: Union[int, None],
    top_k_metric_name: Union[str, None],
    top_k_minimize_metric: Union[bool, None],
):
    """Sample print statement at logger startup."""
    console = Console()

    def format_content(title, value, color):
        if type(value) == list:
            base = f"[b]{title}[/b]"
            for v in value:
                base += f"\n[{color}]{v}"
            return base
        else:
            return f"[b]{title}[/b]\n[{color}]{value}"

    renderables = [
        Panel(format_content("Log Directory", experiment_dir, "black"), expand=True),
        Panel(format_content("Time Tracked", time_to_track, "red"), expand=True),
        Panel(format_content("Stats Tracked", what_to_track, "blue"), expand=True),
        Panel(format_content("Models Tracked", model_type, "green"), expand=True),
    ]
    console.print(Columns(renderables))
