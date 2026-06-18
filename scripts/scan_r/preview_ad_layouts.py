from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from comparison_figure import (
    DATA_DIR,
    METRICS,
    MODEL_COLORS,
    MODEL_MARKERS,
    configure_nature_style,
    format_axis,
    load_summary,
    metric_ylim,
    model_label,
    parameter_values_for_facets,
    parameter_column,
    plot_curve,
)


LEFT_MODEL = "A"
RIGHT_MODEL = "D"
METRIC = METRICS[0]
OUTPUT_STEM = "cooperation_A_D_comparison"


def save_preview(fig: plt.Figure, stem: str) -> list[str]:
    paths: list[str] = []
    for suffix, dpi in [(".svg", 600), (".pdf", 600), (".png", 600), (".tiff", 600)]:
        path = DATA_DIR / f"{stem}{suffix}"
        try:
            fig.savefig(path, dpi=dpi, facecolor="white")
        except PermissionError:
            path = DATA_DIR / f"{stem}_new{suffix}"
            fig.savefig(path, dpi=dpi, facecolor="white")
        paths.append(str(path))
    return paths


def plot_comparison_panel(
    ax: plt.Axes,
    left_df,
    right_df,
    value: float,
    ylim: tuple[float, float],
    *,
    show_xlabel: bool = False,
) -> tuple[list, list]:
    left_col = parameter_column(left_df)
    right_col = parameter_column(right_df)
    left_group = left_df[np.isclose(left_df[left_col], value)]
    right_group = right_df[np.isclose(right_df[right_col], value)]

    plot_curve(
        ax,
        left_group,
        METRIC,
        color=MODEL_COLORS[LEFT_MODEL],
        marker=MODEL_MARKERS[LEFT_MODEL],
        label=model_label(LEFT_MODEL),
        linewidth=1.15,
        zorder=4,
    )
    plot_curve(
        ax,
        right_group,
        METRIC,
        color=MODEL_COLORS[RIGHT_MODEL],
        marker=MODEL_MARKERS[RIGHT_MODEL],
        label=model_label(RIGHT_MODEL),
        linewidth=1.15,
        zorder=5,
    )
    format_axis(ax, METRIC, ylim)
    ax.set_title(rf"$\chi={value:g}$", pad=2, fontsize=7.5)
    ax.set_xlabel(r"$r$", labelpad=6 if show_xlabel else 0)
    if not show_xlabel:
        ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([1, 2, 3, 4, 5])
    return ax.get_legend_handles_labels()


def draw_layout_a(left_df, right_df, values: list[float]) -> plt.Figure:
    ylim = metric_ylim(METRIC, left_df, right_df)
    fig, axes = plt.subplots(
        1,
        len(values),
        figsize=(7.2, 1.65),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    handles, labels = [], []
    for idx, (ax, value) in enumerate(zip(np.atleast_1d(axes), values)):
        panel_handles, panel_labels = plot_comparison_panel(ax, left_df, right_df, value, ylim)
        if idx == 0:
            handles, labels = panel_handles, panel_labels
        ax.label_outer()

    fig.supxlabel(r"$r$", y=-0.02, fontsize=8)
    fig.supylabel(METRIC.ylabel, x=-0.01, fontsize=8)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.2,
    )
    return fig


def draw_layout_b(left_df, right_df, values: list[float]) -> plt.Figure:
    ylim = metric_ylim(METRIC, left_df, right_df)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(6.6, 3.65),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    flat_axes = axes.ravel()

    handles, labels = [], []
    for idx, value in enumerate(values):
        panel_handles, panel_labels = plot_comparison_panel(
            flat_axes[idx],
            left_df,
            right_df,
            value,
            ylim,
            show_xlabel=True,
        )
        if idx == 0:
            handles, labels = panel_handles, panel_labels
        flat_axes[idx].label_outer()
        flat_axes[idx].set_xlabel(r"$r$", labelpad=6)

    legend_ax = flat_axes[len(values)]
    legend_ax.axis("off")
    legend_ax.legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        handlelength=2.2,
        handletextpad=0.5,
    )

    fig.supylabel(METRIC.ylabel, x=-0.012, fontsize=8)
    return fig


def main() -> None:
    configure_nature_style()
    plt.rcParams.update(
        {
            "legend.frameon": False,
            "legend.fontsize": 6.8,
            "lines.markersize": 2.1,
        }
    )

    left_df = load_summary(LEFT_MODEL)
    right_df = load_summary(RIGHT_MODEL)
    values = parameter_values_for_facets(left_df, right_df)
    if not values:
        raise ValueError("No common parameter values found for A-D comparison.")

    outputs: list[str] = []
    fig_a = draw_layout_a(left_df, right_df, values)
    outputs.extend(save_preview(fig_a, f"{OUTPUT_STEM}_layout_A"))
    plt.close(fig_a)

    fig_b = draw_layout_b(left_df, right_df, values)
    outputs.extend(save_preview(fig_b, f"{OUTPUT_STEM}_layout_B"))
    plt.close(fig_b)

    print("Saved A-D layout previews:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
