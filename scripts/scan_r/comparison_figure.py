from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "scan_r"
PARAMETER_COLUMNS = ("fixed_punishment", "chi", "χ")

# 统一折线粗细
LINEWIDTH = 1.5

MODEL_COLORS = {
    "A": "#F66B00",
    "B": "#0072B2",
    "C": "#EDAC07",
    "D": "#0E0008",
    "E": "#111111",
}

PARAMETER_COLORS = ["#EE1606", "#0B8C1F", "#0D2F97", "#960BE0", "#EE771D"]
PARAMETER_MARKERS = ["o", "s", "^", "D", "v"]
MODEL_MARKERS = {
    "A": "o",
    "B": "s",
    "C": "^",
    "D": "D",
    "E": "P",
}


@dataclass(frozen=True)
class ComparisonSpec:
    left: str
    right: str
    basename: str


@dataclass(frozen=True)
class MetricSpec:
    column: str
    basename_key: str
    title: str
    ylabel: str
    fixed_ylim: tuple[float, float] | None = None
    bottom_zero: bool = False
    include_zero: bool = False


METRICS = (
    MetricSpec(
        "mean_cooperation",
        "cooperation",
        "Cooperation",
        r"$\rho_{C+P}$",
        fixed_ylim=(-0.03, 1.03),
    ),
    MetricSpec(
        "mean_net_payoff_avg",
        "net_payoff_avg",
        "Net payoff",
        "net payoff per agent per round",
        include_zero=True,
    ),
    MetricSpec(
        "mean_punisher_cost_avg",
        "punisher_cost_avg",
        "Punisher cost",
        "cost per punisher per D-active round",
        bottom_zero=True,
    ),
    MetricSpec(
        "mean_defector_fine_avg",
        "defector_fine_avg",
        "Defector fine",
        "fine per defector per D-active round",
        bottom_zero=True,
    ),
    MetricSpec(
        "mean_punishment_burden_avg",
        "punishment_burden_avg",
        "Punishment burden",
        "punisher cost + defector fine",
        bottom_zero=True,
    ),
)

REQUIRED_COLUMNS = {"r", *(metric.column for metric in METRICS)}


def configure_nature_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "mathtext.fontset": "stixsans",
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "axes.linewidth": 0.7,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.edgecolor": "black",
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.55,
            "ytick.minor.width": 0.55,
            "lines.linewidth": LINEWIDTH,
            "lines.markersize": 2.4,
            "legend.fontsize": 6.8,
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.edgecolor": "0.65",
            "legend.facecolor": "white",
            "legend.framealpha": 0.88,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def summary_path(model: str) -> Path:
    model = model.upper()
    return DATA_DIR / model / f"{model}_summary.csv"


def require_columns(df: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = columns.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")


def parameter_column(df: pd.DataFrame) -> str | None:
    for column in PARAMETER_COLUMNS:
        if column in df.columns:
            return column
    return None


def load_summary(model: str) -> pd.DataFrame:
    path = summary_path(model)
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")

    df = pd.read_csv(path)
    require_columns(df, REQUIRED_COLUMNS, path)

    numeric_columns = ["r", "runs"]
    for metric in METRICS:
        numeric_columns.append(metric.column)
        std_column = metric.column.replace("mean_", "std_", 1)
        if std_column in df.columns:
            numeric_columns.append(std_column)
    if "mean_defector_active_steps" in df.columns:
        numeric_columns.append("mean_defector_active_steps")
    if "std_defector_active_steps" in df.columns:
        numeric_columns.append("std_defector_active_steps")

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="raise")

    param_col = parameter_column(df)
    if param_col is not None:
        df[param_col] = pd.to_numeric(df[param_col], errors="raise")
        df = df.sort_values([param_col, "r"]).reset_index(drop=True)
    else:
        df = df.sort_values("r").reset_index(drop=True)
    return df


def metric_ylim(metric: MetricSpec, *frames: pd.DataFrame) -> tuple[float, float]:
    if metric.fixed_ylim is not None:
        return metric.fixed_ylim

    arrays = [
        frame[metric.column].to_numpy(dtype=float)
        for frame in frames
        if metric.column in frame.columns and not frame.empty
    ]
    if not arrays:
        return (0.0, 1.0)

    values = np.concatenate(arrays)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0.0, 1.0)

    ymin = float(values.min())
    ymax = float(values.max())
    if metric.include_zero:
        ymin = min(ymin, 0.0)
        ymax = max(ymax, 0.0)
    if metric.bottom_zero:
        ymin = 0.0

    if np.isclose(ymin, ymax):
        span = abs(ymax) if not np.isclose(ymax, 0.0) else 1.0
        pad = 0.08 * span
        ymin -= pad
        ymax += pad
        if metric.bottom_zero:
            ymin = 0.0
    else:
        pad = 0.08 * (ymax - ymin)
        if metric.bottom_zero:
            ymax += pad
        else:
            ymin -= pad
            ymax += pad

    return ymin, ymax


def format_axis(ax: plt.Axes, metric: MetricSpec, ylim: tuple[float, float]) -> None:
    ax.set_xlim(0.95, 5.05)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    if metric.column == "mean_cooperation":
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(metric.ylabel)


def model_label(model: str) -> str:
    return f"Model {model.upper()}"


def param_label(model: str, value: float) -> str:
    return rf"Model {model.upper()}, $\chi={value:g}$"


def plot_curve(
    ax: plt.Axes,
    group: pd.DataFrame,
    metric: MetricSpec,
    *,
    color: str,
    marker: str,
    label: str,
    linewidth: float = LINEWIDTH,
    zorder: int = 3,
) -> Line2D:
    line = ax.plot(
        group["r"].to_numpy(dtype=float),
        group[metric.column].to_numpy(dtype=float),
        color=color,
        linestyle="-",
        marker=marker,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=0.55,
        linewidth=linewidth,
        label=label,
        zorder=zorder,
    )[0]
    return line


def legend_rect(loc: str, width: float, height: float) -> tuple[float, float, float, float]:
    if loc == "upper left":
        return 0.02, 0.98 - height, 0.02 + width, 0.98
    if loc == "upper right":
        return 0.98 - width, 0.98 - height, 0.98, 0.98
    if loc == "lower left":
        return 0.02, 0.02, 0.02 + width, 0.02 + height
    if loc == "lower right":
        return 0.98 - width, 0.02, 0.98, 0.02 + height
    if loc == "center left":
        return 0.02, 0.5 - height / 2.0, 0.02 + width, 0.5 + height / 2.0
    return 0.98 - width, 0.5 - height / 2.0, 0.98, 0.5 + height / 2.0


def line_occupancy_score(ax: plt.Axes, loc: str, label_count: int) -> float:
    width = 0.34 if label_count <= 2 else 0.48
    height = min(0.18 + 0.07 * label_count, 0.58)
    x0, y0, x1, y1 = legend_rect(loc, width, height)
    margin = 0.035
    x0 -= margin
    y0 -= margin
    x1 += margin
    y1 += margin

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if np.isclose(xmin, xmax) or np.isclose(ymin, ymax):
        return 0.0

    score = 0.0
    for line in ax.lines:
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        if x.size == 0:
            continue
        if x.size > 1:
            x = np.concatenate((x, 0.5 * (x[:-1] + x[1:])))
            y = np.concatenate((y, 0.5 * (y[:-1] + y[1:])))

        nx = (x - xmin) / (xmax - xmin)
        ny = (y - ymin) / (ymax - ymin)
        inside = (nx >= x0) & (nx <= x1) & (ny >= y0) & (ny <= y1)
        score += float(np.count_nonzero(inside))

    return score


def choose_legend_location(ax: plt.Axes, label_count: int) -> tuple[str, bool]:
    if label_count > 4:
        return "center left", True

    candidates = ("upper left", "upper right", "lower left", "lower right", "center left", "center right")
    scores = [(line_occupancy_score(ax, loc, label_count), idx, loc) for idx, loc in enumerate(candidates)]
    best_score, _, best_loc = min(scores, key=lambda item: (item[0], item[1]))
    if best_score > 3.0:
        return "center left", True
    return best_loc, False


def add_adaptive_legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    kwargs = {
        "borderaxespad": 0.35,
        "handlelength": 1.8,
        "handletextpad": 0.4,
    }
    ax.legend(handles, labels, loc="lower right", **kwargs)


def parameter_values_for_facets(left_df: pd.DataFrame, right_df: pd.DataFrame) -> list[float]:
    left_col = parameter_column(left_df)
    right_col = parameter_column(right_df)
    if left_col is None and right_col is None:
        return []

    if left_col is None:
        return sorted({round(float(value), 10) for value in right_df[right_col].dropna().unique()})
    if right_col is None:
        return sorted({round(float(value), 10) for value in left_df[left_col].dropna().unique()})

    left_values = {round(float(value), 10) for value in left_df[left_col].dropna().unique()}
    right_values = {round(float(value), 10) for value in right_df[right_col].dropna().unique()}
    return sorted(left_values.intersection(right_values))


def facet_group(df: pd.DataFrame, param_col: str | None, value: float) -> pd.DataFrame:
    if param_col is None:
        return df
    return df[np.isclose(df[param_col], value)]


def parameter_panel_label(value: float) -> str:
    return rf"$x={value:g}$"


def make_facet_axes(n_values: int) -> tuple[plt.Figure, list[plt.Axes]]:
    if n_values == 5:
        fig = plt.figure(figsize=(6.9, 4.45), constrained_layout=True)
        gridspec = fig.add_gridspec(2, 6)
        positions = (
            gridspec[0, 0:2],
            gridspec[0, 2:4],
            gridspec[0, 4:6],
            gridspec[1, 1:3],
            gridspec[1, 3:5],
        )
        axes: list[plt.Axes] = []
        shared_ax: plt.Axes | None = None
        for position in positions:
            ax = fig.add_subplot(position, sharex=shared_ax, sharey=shared_ax)
            if shared_ax is None:
                shared_ax = ax
            axes.append(ax)
        return fig, axes

    ncols = 3 if n_values > 2 else n_values
    nrows = int(np.ceil(n_values / ncols))
    fig, axes_array = plt.subplots(
        nrows,
        ncols,
        figsize=(6.9, 2.35 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    flat_axes = np.atleast_1d(axes_array).ravel()
    for ax in flat_axes[n_values:]:
        ax.axis("off")
    return fig, list(flat_axes[:n_values])


def draw_parameter_facets(
    left: str,
    left_df: pd.DataFrame,
    right: str,
    right_df: pd.DataFrame,
    metric: MetricSpec,
) -> plt.Figure:
    left_col = parameter_column(left_df)
    right_col = parameter_column(right_df)
    values = parameter_values_for_facets(left_df, right_df)
    if not values:
        raise ValueError("Facet comparison requires at least one parameterized summary.")

    n_values = len(values)
    fig, flat_axes = make_facet_axes(n_values)
    ylim = metric_ylim(metric, left_df, right_df)

    for idx, value in enumerate(values):
        ax = flat_axes[idx]
        left_group = facet_group(left_df, left_col, value)
        right_group = facet_group(right_df, right_col, value)

        plot_curve(
            ax,
            left_group,
            metric,
            color=MODEL_COLORS[left],
            marker=MODEL_MARKERS[left],
            label=model_label(left),
            linewidth=LINEWIDTH,
            zorder=4,
        )
        plot_curve(
            ax,
            right_group,
            metric,
            color=MODEL_COLORS[right],
            marker=MODEL_MARKERS[right],
            label=model_label(right),
            linewidth=LINEWIDTH,
            zorder=5,
        )

        format_axis(ax, metric, ylim)
        ax.set_title(parameter_panel_label(value), pad=2, fontsize=7.5)
        add_adaptive_legend(ax)

    return fig


def draw_single_axis_comparison(
    left: str,
    left_df: pd.DataFrame,
    right: str,
    right_df: pd.DataFrame,
    metric: MetricSpec,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.8, 3.35), constrained_layout=True)
    ylim = metric_ylim(metric, left_df, right_df)

    for model, df, _ in [(left, left_df, False), (right, right_df, True)]:
        param_col = parameter_column(df)

        if param_col is None:
            color = MODEL_COLORS[model]
            marker = MODEL_MARKERS[model]
            plot_curve(
                ax,
                df,
                metric,
                color=color,
                marker=marker,
                label=model_label(model),
                linewidth=LINEWIDTH,
                zorder=5,
            )
            continue

        for idx, (value, group) in enumerate(df.groupby(param_col, sort=True)):
            color = PARAMETER_COLORS[idx % len(PARAMETER_COLORS)]
            marker = PARAMETER_MARKERS[idx % len(PARAMETER_MARKERS)]
            plot_curve(
                ax,
                group,
                metric,
                color=color,
                marker=marker,
                label=param_label(model, float(value)),
                linewidth=LINEWIDTH,
                zorder=3,
            )

    format_axis(ax, metric, ylim)
    add_adaptive_legend(ax)
    return fig


def save_figure(fig: plt.Figure, basename: str) -> list[Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stem = DATA_DIR / basename
    outputs: list[Path] = []
    for suffix, dpi in [(".svg", 1200), (".pdf", 1200), (".png", 1200), (".tiff", 1200)]:
        path = stem.with_suffix(suffix)
        try:
            fig.savefig(path, dpi=dpi, facecolor="white")
            outputs.append(path)
        except PermissionError:
            fallback = stem.with_name(f"{stem.name}_new").with_suffix(suffix)
            fig.savefig(fallback, dpi=dpi, facecolor="white")
            print(f"Warning: could not overwrite {path}; saved {fallback} instead.")
            outputs.append(fallback)
    return outputs


def metric_basename(base: str, metric: MetricSpec) -> str:
    for existing in sorted((item.basename_key for item in METRICS), key=len, reverse=True):
        prefix = f"{existing}_"
        if base.startswith(prefix):
            return base.replace(prefix, f"{metric.basename_key}_", 1)
    return f"{base}_{metric.basename_key}"


def make_comparison(spec: ComparisonSpec) -> list[Path]:
    configure_nature_style()
    left = spec.left.upper()
    right = spec.right.upper()
    left_df = load_summary(left)
    right_df = load_summary(right)

    outputs: list[Path] = []
    for metric in METRICS:
        if parameter_column(left_df) is not None or parameter_column(right_df) is not None:
            fig = draw_parameter_facets(left, left_df, right, right_df, metric)
        else:
            fig = draw_single_axis_comparison(left, left_df, right, right_df, metric)

        outputs.extend(save_figure(fig, metric_basename(spec.basename, metric)))
        plt.close(fig)
    return outputs


def main(spec: ComparisonSpec) -> None:
    outputs = make_comparison(spec)
    print("Saved scan-r model comparison:")
    for path in outputs:
        print(path)
