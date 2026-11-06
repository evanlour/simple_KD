from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

PALETTE = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
"""Default color cycle (plotly-like hex colors) for multi-line figures."""

STYLE_RC = {
    "axes.edgecolor": "#D0D5DD",
    "axes.labelcolor": "#344054",
    "text.color": "#101828",
    "xtick.color": "#475467",
    "ytick.color": "#475467",
    "grid.color": "#EAECF0",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def plot_cm(
    cm: np.ndarray,
    figsize: tuple[float, float] = (10.8, 8.6),
    cmap: str = "Blues",
    classes: int = 10,
    labels: list["str"] | None = None,
    tick_step: int = 10,
    dpi: int = 200,
    savename: str | Path | None = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
):
    """Plots a given confusion matrix heatmap over the class grid. Uses seaborn and supports cosmetics settings. More info below.

    Args:
        cm (np.ndarray): The confusion matrix that will be plotted.
        figsize (tuple[float, float], optional): Figure size in inches. Defaults to (10.8, 8.6).
        cmap (str, optional): Any matplotlib/seaborn colormap name. Defaults to "Blues".
        classes (int, optional): Number of classes in the matrix, used to place axis ticks.
            Defaults to 10.
        labels (list["str"] | None, optional): Optional list of class names to label the
            ticks and pass to `confusion_matrix` as its `labels` argument. Defaults to None.
        tick_step (int, optional): Labeled tick every N classes. Defaults to 10.
        dpi (int, optional): PNG resolution when saving. Defaults to 200.
        savename (str | Path | None, optional): Path to save the PNG to; parent directories
            are created automatically. If None, the figure is not saved. Defaults to None.
        title (str, optional): Figure title. Defaults to "".
        xlabel (str, optional): X axis label (predicted labels). Defaults to "".
        ylabel (str, optional): Y axis label (true labels). Defaults to "".
    """
    fig = Figure(figsize=figsize, facecolor="white")
    ax = fig.add_subplot(111)
    with sns.axes_style("whitegrid", rc=STYLE_RC):
        sns.heatmap(
            cm,
            ax=ax,
            cmap=cmap,
            vmin=0,
            vmax=float(cm.max()) or 1.0,
            xticklabels=False,
            yticklabels=False,
            cbar_kws={"label": "count", "shrink": 0.8},
            linewidths=0,
        )
        ticks = np.arange(0, cm.shape[0], tick_step) + 0.5
        tick_labels = np.asarray(labels) if labels is not None else np.arange(classes)
        tick_labels = tick_labels[::tick_step]
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=0)  # type: ignore
        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels, rotation=0)  # type: ignore
        ax.set_title(title, loc="left", fontsize=15, pad=12, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(labelsize=8)
        ax.grid(False)
        fig.tight_layout()

    if savename is not None:
        if isinstance(savename, str):
            savename = Path(savename)
        savename.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            savename,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.12,
            facecolor="white",
        )


def plot_lines(
    values: list[np.ndarray],
    colors: list["str"] | None = None,
    labels: list[str] | None = None,
    figsize: tuple[float, float] = (11, 4.4),
    savename: str | Path | None = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    dpi: int = 200,
):
    """Plots a series of lines with seaborn. Supports some cosmetics options. More info below.

    Args:
        values (list[np.ndarray]): One 1D array per line; all must be equal length.
        colors (list["str"] | None, optional): One valid matplotlib color per line.
            Defaults to the a pre-defined color palette if None. Defaults to None.
        labels (list[str] | None, optional): Legend label per line. When None, no
            legend is drawn. Defaults to None.
        figsize (tuple[float, float], optional): Figure size in inches. Defaults to (11, 4.4).
        savename (str | Path | None, optional): Path to save the PNG to; parent directories
            are created automatically. If None, the figure is not saved. Defaults to None.
        title (str, optional): Figure title. Defaults to "".
        xlabel (str, optional): X axis label. Defaults to "".
        ylabel (str, optional): Y axis label. Defaults to "".
        dpi (int, optional): PNG resolution when saving. Defaults to 200.
    """
    colors = (
        list(colors)
        if colors is not None
        else [PALETTE[i % len(PALETTE)] for i in range(len(values))]
    )
    fig = Figure(figsize=figsize, facecolor="white")
    ax = fig.add_subplot(111)

    with sns.axes_style("whitegrid", rc=STYLE_RC):
        x_values = np.arange(1, values[0].shape[0] + 1)
        for i, (line, color) in enumerate(zip(values, colors)):
            sns.lineplot(
                x=x_values,
                y=line,
                ax=ax,
                label=None if labels is None else labels[i],
                color=color,
                linewidth=2.2,
                legend=False,
            )
        ax.set_title(
            title,
            loc="left",
            fontsize=15,
            pad=12,
            fontweight="bold",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if labels is not None:
            ax.legend(frameon=False, loc="best")
        ax.set_axisbelow(True)
        sns.despine(ax=ax)
        fig.tight_layout()

    if savename is not None:
        if isinstance(savename, str):
            savename = Path(savename)
        savename.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            savename,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.12,
            facecolor="white",
        )
