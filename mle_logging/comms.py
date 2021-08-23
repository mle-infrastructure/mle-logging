import pandas as pd
from typing import Union
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns
from rich.table import Table
from rich import box
import datetime
from ._version import __version__


console_width = 80


def print_welcome() -> None:
    """Display header with clock and general toolbox configurations."""
    welcome_ascii = r"""
 __    __  __      ______  __      ______  ______
/\ "-./  \/\ \    /\  ___\/\ \    /\  __ \/\  ___\
\ \ \-./\ \ \ \___\ \  __\  \ \___\ \ \/\ \ \ \__ \
 \ \_\ \ \_\ \_____\ \_____\ \_____\ \_____\ \_____\
  \/_/  \/_/\/_____/\/_____/\/_____/\/_____/\/_____/
    """.splitlines()
    grid = Table.grid(expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="right")
    grid.add_row(
        welcome_ascii[1],
        datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"),
    )
    grid.add_row(welcome_ascii[2], f"Logger v{__version__} :lock_with_ink_pen:")
    # grid.add_row(
    #     welcome_ascii[3],
    #     "  [link=https://live.staticflickr.com/2061/2306127707_2607857c2d_z.jpg]U r awesome![/link] :hugging_face:",  # noqa: E501
    # )

    grid.add_row(
        welcome_ascii[3],
        "  [link=https://twitter.com/RobertTLange]@RobertTLange[/link] :bird:",
    )
    grid.add_row(
        welcome_ascii[4],
        "  [link=https://github.com/RobertTLange/mle-logging/blob/main/examples/getting_started.ipynb]MLE-Log Docs[/link] [not italic]:notebook:[/]",  # noqa: E501
    )
    grid.add_row(
        welcome_ascii[5],
        "  [link=https://github.com/RobertTLange/mle-logging/]MLE-Log Repo[/link] [not italic]:pencil:[/]",  # noqa: E501
    )
    panel = Panel(grid, style="white on blue", expand=True)
    Console(width=console_width).print(panel)


def print_startup(
    experiment_dir: str,
    config_fname: Union[str, None],
    time_to_track: list,
    what_to_track: list,
    model_type: str,
    seed_id: str,
    use_tboard: bool,
    reload: bool,
    print_every_k_updates: Union[int, None],
    ckpt_time_to_track: Union[str, None],
    save_every_k_ckpt: Union[int, None],
    save_top_k_ckpt: Union[int, None],
    top_k_metric_name: Union[str, None],
    top_k_minimize_metric: Union[bool, None],
):
    """Rich print statement at logger startup."""
    grid = Table.grid(expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="left")

    def format_content(title, value):
        if type(value) == list:
            base = f"[b]{title}[/b]: "
            for i, v in enumerate(value):
                base += f"{v}"
                if i < len(value) - 1:
                    base += ", "
            return base
        else:
            return f"[b]{title}[/b]: {value}"

    time_to_print = [t for t in time_to_track if t not in ["time", "time_elapsed"]]
    renderables = [
        Panel(format_content(":book: Log Dir", experiment_dir), expand=True),
        Panel(format_content(":page_facing_up: Config", config_fname), expand=True),
        Panel(format_content(":watch: Time", time_to_print), expand=True),
        Panel(
            format_content(":chart_with_downwards_trend: Stats", what_to_track),
            expand=True,
        ),
        Panel(format_content(":seedling: Seed ID", seed_id), expand=True),
        Panel(
            format_content(":chart_with_upwards_trend: Tensorboard", use_tboard),
            expand=True,
        ),
        Panel(format_content(":rocket: Model", model_type), expand=True),
        Panel(format_content("Tracked ckpt Time", ckpt_time_to_track), expand=True),
        Panel(
            format_content(":clock1130: Every k-th ckpt", save_every_k_ckpt),
            expand=True,
        ),
        Panel(format_content(":trident: Top k ckpt", save_top_k_ckpt), expand=True),
        Panel(format_content("Top k-th metric", top_k_metric_name), expand=True),
        Panel(
            format_content("Top k-th minimization", top_k_minimize_metric), expand=True
        ),
    ]

    grid.add_row(renderables[0], renderables[1])
    grid.add_row(renderables[2], renderables[3])
    grid.add_row(renderables[4], renderables[6])
    if save_every_k_ckpt is None and save_top_k_ckpt is not None:
        grid.add_row(renderables[8],)
    elif save_every_k_ckpt is not None and save_top_k_ckpt is None:
        grid.add_row(renderables[9],)
    elif save_every_k_ckpt is not None and save_top_k_ckpt is not None:
        grid.add_row(renderables[8], renderables[9])
    # grid.add_row(renderables[10], renderables[11])
    panel = Panel(grid, expand=True)
    Console(width=console_width).print(panel)


def print_update(time_to_print, what_to_print, c_tick, s_tick, print_header):
    """Rich print statement for logger update."""
    table = Table(
        show_header=print_header,
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
                ":chart_with_downwards_trend: [blue]" + c_label + "[/blue]",
                width=14,
                justify="center",
            )
        else:
            table.add_column(
                "[blue]" + c_label + "[/blue]",
                width=12,
                justify="center",
            )
    row_list = pd.concat(
        [c_tick[time_to_print], s_tick[what_to_print].round(4)], axis=1
    ).values.tolist()[0]
    row_str_list = [str(v) for v in row_list]
    table.add_row(*row_str_list)

    # Print statistics update
    Console(width=console_width).print(table, justify="center")


def print_reload(experiment_dir: str):
    """Rich print statement for logger reloading."""
    Console().log(f"Reloaded log from {experiment_dir}")


def print_storage(fig_path: Union[str, None] = None,
                  extra_path: Union[str, None] = None,
                  final_model_path: Union[str, None] = None,
                  every_k_model_path: Union[str, None] = None,
                  top_k_model_path: Union[str, None] = None):
    """Rich print statement for object saving log."""
    table = Table(
        show_header=False,
        row_styles=["none"],
        border_style="white",
        box=box.SIMPLE,
    )

    table.add_column(
        "---",
        style="red",
        width=16,
        justify="left",
    )

    table.add_column(
        "---",
        style="red",
        width=64,
        justify="left",
    )

    if fig_path is not None:
        table.add_row(":envelope_with_arrow: - Figure",
                      f"{fig_path}")
    if extra_path is not None:
        table.add_row(":envelope_with_arrow: - Extra",
                      f"{extra_path}")
    if final_model_path is not None:
        table.add_row(":envelope_with_arrow: - Model",
                      f"{final_model_path}")
    if every_k_model_path is not None:
        table.add_row(":envelope_with_arrow: - Every-K",
                      f"{every_k_model_path}")
    if top_k_model_path is not None:
        table.add_row(":envelope_with_arrow: - Top-K",
                      f"{top_k_model_path}")

    to_print = ((final_model_path is not None) +
                (fig_path is not None) +
                (extra_path is not None) +
                (final_model_path is not None) +
                (every_k_model_path is not None) +
                (top_k_model_path is not None)) > 0
    # Print storage update
    if to_print:
        Console(width=console_width).print(table, justify="left")


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
