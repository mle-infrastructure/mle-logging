import pandas as pd
from typing import Union
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns
from rich.table import Table
from rich import box
import datetime


def print_welcome() -> None:
    """Display header with clock and general toolbox configurations."""
    welcome_ascii = """███╗   ███╗██╗     ███████╗      ██╗      ██████╗  ██████╗
████╗ ████║██║     ██╔════╝      ██║     ██╔═══██╗██╔════╝
██╔████╔██║██║     █████╗  █████╗██║     ██║   ██║██║  ███╗
██║╚██╔╝██║██║     ██╔══╝  ╚════╝██║     ██║   ██║██║   ██║
██║ ╚═╝ ██║███████╗███████╗      ███████╗╚██████╔╝╚██████╔╝
╚═╝     ╚═╝╚══════╝╚══════╝      ╚══════╝ ╚═════╝  ╚═════╝
    """.splitlines()

    grid = Table.grid(expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="right")
    grid.add_row(
        welcome_ascii[0],
        datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
    )
    grid.add_row(
        welcome_ascii[1],
        "  [link=https://tinyurl.com/srpy4nrp]You are awesome![/link] [not italic]:hugging_face:[/]",  # noqa: E501
    )

    grid.add_row(
        welcome_ascii[2],
        "  [link=https://twitter.com/RobertTLange]@RobertTLange[/link] :bird:",
    )
    grid.add_row(
        welcome_ascii[3],
        "  [link=https://roberttlange.github.io/mle-toolbox/logging/mle_logging/]MLE-Logging Docs[/link] [not italic]:notebook:[/]",  # noqa: E501
    )
    grid.add_row(
        welcome_ascii[4],
        "  [link=https://github.com/RobertTLange/mle-logging/]MLE-Logging Repo[/link] [not italic]:pencil:[/]",  # noqa: E501
    )
    panel = Panel(grid, style="white on red", expand=False)
    console = Console()
    console.print(panel)


def print_startup(
    experiment_dir: str,
    time_to_track: list,
    what_to_track: list,
    model_type: str,
    seed_id: str,
    ckpt_time_to_track: Union[str, None],
    save_every_k_ckpt: Union[int, None],
    save_top_k_ckpt: Union[int, None],
    top_k_metric_name: Union[str, None],
    top_k_minimize_metric: Union[bool, None],
):
    """Rich print statement at logger startup."""
    console = Console()

    def format_content(title, value, color):
        if type(value) == list:
            base = f"[b]{title}[/b]"
            for v in value:
                base += f"\n[{color}]{v}"
            return base
        else:
            return f"[b]{title}[/b]\n[{color}]{value}"

    time_to_print = [t for t in time_to_track if t not in ["time", "time_elapsed"]]
    renderables = [
        Panel(format_content("Log Directory", experiment_dir, "grey"), expand=True),
        Panel(format_content("Time Tracked", time_to_print, "red"), expand=True),
        Panel(format_content("Stats Tracked", what_to_track, "blue"), expand=True),
        Panel(format_content("Models Tracked", model_type, "green"), expand=True),
        Panel(format_content("Seed ID", seed_id, "orange"), expand=True),
    ]
    console.print(Columns(renderables))


def print_update(time_to_print, what_to_print, c_tick, s_tick):
    """Rich print statement for logger update."""
    console = Console()
    table = Table(
        show_header=True,
        row_styles=["none"],
        border_style="white",
        box=box.SIMPLE,
    )
    # Add watch and book emoji
    for i, c_label in enumerate(time_to_print):
        if i == 0:
            table.add_column(
                ":watch: [red]" + c_label + "[/red]",
                style="red",
                width=14,
                justify="left",
            )
        else:
            table.add_column(
                "[red]" + c_label + "[/red]",
                style="red",
                width=12,
                justify="center",
            )
    for i, c_label in enumerate(what_to_print):
        if i == 0:
            table.add_column(
                ":open_book: [blue]" + c_label + "[/blue]",
                style="blue",
                width=14,
                justify="center",
            )
        else:
            table.add_column(
                "[blue]" + c_label + "[/blue]",
                style="blue",
                width=12,
                justify="center",
            )
    row_list = pd.concat(
        [c_tick[time_to_print], s_tick[what_to_print].round(4)], axis=1
    ).values.tolist()[0]
    row_str_list = [str(v) for v in row_list]
    table.add_row(*row_str_list)
    console.print(table, justify="center")


if __name__ == "__main__":
    print_welcome()
    print_startup(
        experiment_dir="experiment_dir",
        time_to_track=["meta_loss"],
        what_to_track=["num_updates"],
        model_type="torch",
        ckpt_time_to_track="num_updates",
        save_every_k_ckpt=None,
        save_top_k_ckpt=2,
        top_k_metric_name="meta_loss",
        top_k_minimize_metric=True,
    )

    c_tick = pd.DataFrame(columns=["num_updates"])
    c_tick.loc[0] = [10]
    s_tick = pd.DataFrame(columns=["meta_loss", "train_loss", "test_loss"])
    s_tick.loc[0] = [0.124456356436, 0.13451345135, 0.1345531513]
    print_update(
        time_to_print=["num_updates"],
        what_to_print=["meta_loss", "train_loss", "test_loss"],
        c_tick=c_tick,
        s_tick=s_tick,
    )
