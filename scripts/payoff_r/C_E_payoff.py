from __future__ import annotations

# Standalone script: comparison helpers are inlined here intentionally.
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


SCRIPT_DIR = Path(__file__).resolve().parent
CWD = Path.cwd().resolve()

REQUIRED_COLUMNS = {"model", "r", "runs", "system_payoff_mean", "system_payoff_std"}
NON_PARAMETER_COLUMNS = REQUIRED_COLUMNS.union({"seed", "system_payoff", "seconds"})

COOPERATION_CURVE_COLORS = [
    "#F2BE00",
    "#FF7043",
    "#3498DB",
    "#8A2BE2",
    "#E6A39A",
]

COOPERATION_CURVE_MARKERS = ["<", "D", "v", "^", "o"]

A_D_MODEL_COLORS = {
    "A": "#F78409F9",
    "D": "#0A0A0A",
}

A_D_MODEL_MARKERS = {
    "A": "o",
    "D": "s",
}

SINGLE_CURVE_COLOR = "#FF7043"
REFERENCE_CURVE_COLOR = "#000000"
REFERENCE_MARKER = "X"


@dataclass(frozen=True)
class PayoffCurve:
    model: str
    label: str
    data: pd.DataFrame
    color: str
    marker: str
    linestyle: str = "-"


def payoff_summary_path(root: Path, model: str) -> Path:
    return root / "data" / "payoff_r" / model / f"{model}_payoff_r_summary.csv"


def find_project_root(models: tuple[str, ...]) -> Path:
    candidates = [
        CWD,
        SCRIPT_DIR,
        CWD.parent,
        SCRIPT_DIR.parent,
    ]

    for root in candidates:
        if all(payoff_summary_path(root, model).exists() for model in models):
            return root

    required = "\n".join(str(payoff_summary_path(candidates[0], model)) for model in models)
    searched = "\n".join(str(root) for root in candidates)
    raise FileNotFoundError(
        "Could not find the required payoff summary CSV files.\n"
        f"Required under project root:\n{required}\n"
        "Please run the corresponding *_payoff_r.py scripts first.\n"
        f"Searched these folders:\n{searched}"
    )


def configure_publication_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.edgecolor": "black",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 3.6,
            "ytick.major.size": 3.6,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.7,
            "ytick.minor.width": 0.7,
            "lines.linewidth": 1.55,
            "lines.markersize": 3.8,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
        }
    )


def require_columns(df: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{name} is missing required columns: {missing_text}")


def load_payoff_summary(root: Path, model: str) -> pd.DataFrame:
    path = payoff_summary_path(root, model)
    df = pd.read_csv(path)
    require_columns(df, REQUIRED_COLUMNS, str(path))

    for column in df.columns:
        if column != "model":
            df[column] = pd.to_numeric(df[column], errors="raise")
    df["model"] = df["model"].astype(str)
    return df.sort_values("r").reset_index(drop=True)


def parameter_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in NON_PARAMETER_COLUMNS]


def format_value(value: float | int) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):g}"


def parameter_label(params: dict[str, float]) -> str:
    if not params:
        return ""
    labels = []
    for key, value in params.items():
        if key == "chi":
            labels.append(rf"$\chi={format_value(value)}$")
        else:
            labels.append(rf"${key}={format_value(value)}$")
    return ", ".join(labels)


def curve_label(model: str, params: dict[str, float], include_single_params: bool) -> str:
    label = f"Model_{model}"
    params_text = parameter_label(params)
    if include_single_params and params_text:
        return f"{label}  {params_text}"
    return label


def single_curve_color(model: str, first_model: str, second_model: str) -> str:
    if model == "E":
        return REFERENCE_CURVE_COLOR
    if first_model in {"A", "D"} and second_model in {"C", "E"} and model == second_model:
        return REFERENCE_CURVE_COLOR
    if (first_model, second_model) in {("B", "E"), ("C", "E")} and model == first_model:
        return SINGLE_CURVE_COLOR
    return REFERENCE_CURVE_COLOR


def single_curve_marker(model: str, first_model: str, second_model: str) -> str:
    if model == "E":
        return REFERENCE_MARKER
    if first_model in {"A", "D"} and second_model in {"C", "E"} and model == second_model:
        return REFERENCE_MARKER
    if (first_model, second_model) in {("B", "E"), ("C", "E")} and model == first_model:
        return "o"
    return "o"


def curves_for_model(
    df: pd.DataFrame,
    model: str,
    first_model: str,
    second_model: str,
    *,
    include_single_params: bool = True,
) -> list[PayoffCurve]:
    param_columns = parameter_columns(df)
    color = single_curve_color(model, first_model, second_model)
    marker = single_curve_marker(model, first_model, second_model)

    if not param_columns:
        return [PayoffCurve(model=model, label=f"Model_{model}", data=df.sort_values("r"), color=color, marker=marker)]

    curves: list[PayoffCurve] = []
    grouped = df.groupby(param_columns, sort=True, dropna=False)
    total_groups = grouped.ngroups
    for idx, (key, group) in enumerate(grouped):
        if not isinstance(key, tuple):
            key = (key,)
        params = {column: float(value) for column, value in zip(param_columns, key)}
        group_color = color if total_groups == 1 else COOPERATION_CURVE_COLORS[idx % len(COOPERATION_CURVE_COLORS)]
        group_marker = marker if total_groups == 1 else COOPERATION_CURVE_MARKERS[idx % len(COOPERATION_CURVE_MARKERS)]
        label = curve_label(model, params, include_single_params or total_groups > 1)
        curves.append(
            PayoffCurve(
                model=model,
                label=label,
                data=group.sort_values("r").reset_index(drop=True),
                color=group_color,
                marker=group_marker,
            )
        )
    return curves


def plot_curve(ax: plt.Axes, curve: PayoffCurve, *, zorder: int = 4) -> None:
    x = curve.data["r"].to_numpy(dtype=float)
    y = curve.data["system_payoff_mean"].to_numpy(dtype=float)

    ax.plot(
        x,
        y,
        color=curve.color,
        linestyle=curve.linestyle,
        marker=curve.marker,
        markevery=max(1, len(x) // 10),
        markerfacecolor="white",
        markeredgecolor=curve.color,
        markeredgewidth=0.75,
        linewidth=1.55,
        label=curve.label,
        zorder=zorder,
    )


def payoff_ylim(curves: list[PayoffCurve]) -> tuple[float, float]:
    values: list[float] = []
    for curve in curves:
        y = curve.data["system_payoff_mean"].to_numpy(dtype=float)
        values.extend(y.tolist())
    finite = np.array([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return -1.0, 1.0
    ymin = float(np.min(finite))
    ymax = float(np.max(finite))
    if np.isclose(ymin, ymax):
        pad = max(0.2, abs(ymin) * 0.12)
    else:
        pad = 0.1 * (ymax - ymin)
    ymin -= pad
    ymax += pad
    if ymin > 0:
        ymin = 0.0
    if ymax < 0:
        ymax = 0.0
    return ymin, ymax


def format_payoff_axis(ax: plt.Axes, curves: list[PayoffCurve]) -> None:
    r_values = np.concatenate([curve.data["r"].to_numpy(dtype=float) for curve in curves])
    ax.set_xlim(float(np.nanmin(r_values)) - 0.05, float(np.nanmax(r_values)) + 0.05)
    ax.set_ylim(*payoff_ylim(curves))
    ax.axhline(0.0, color="0.35", linewidth=0.7, linestyle="--", alpha=0.55)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel("System average payoff")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))


def save_figure(fig: plt.Figure, basename: Path) -> list[Path]:
    basename.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix, dpi in [(".pdf", 600), (".svg", 600), (".png", 600), (".tiff", 600)]:
        path = basename.with_suffix(suffix)
        fig.savefig(path, dpi=dpi, facecolor="white")
        outputs.append(path)
    return outputs


def draw_pair_comparison(first_df: pd.DataFrame, second_df: pd.DataFrame, first_model: str, second_model: str) -> plt.Figure:
    first_curves = curves_for_model(
        first_df,
        first_model,
        first_model,
        second_model,
        include_single_params=first_model in {"A", "D"},
    )
    second_curves = curves_for_model(
        second_df,
        second_model,
        first_model,
        second_model,
        include_single_params=second_model in {"A", "D"},
    )
    all_curves = first_curves + second_curves

    fig, ax = plt.subplots(figsize=(5.1, 3.8), constrained_layout=True)
    for idx, curve in enumerate(all_curves):
        plot_curve(ax, curve, zorder=4 + idx)

    format_payoff_axis(ax, all_curves)
    ax.set_title(f"Model {first_model} vs {second_model}: payoff", fontsize=12, pad=4)
    ax.legend(loc="best", handlelength=2.2, handletextpad=0.45)
    return fig


def common_chi_values(first_df: pd.DataFrame, second_df: pd.DataFrame) -> list[float]:
    require_columns(first_df, {"chi"}, "first payoff summary")
    require_columns(second_df, {"chi"}, "second payoff summary")
    first_values = {round(float(value), 10) for value in first_df["chi"].unique()}
    second_values = {round(float(value), 10) for value in second_df["chi"].unique()}
    values = sorted(first_values.intersection(second_values))
    if not values:
        raise ValueError("No common chi values were found in the two payoff summaries.")
    return values


def draw_chi_facets(first_df: pd.DataFrame, second_df: pd.DataFrame, first_model: str, second_model: str) -> plt.Figure:
    chi_values = common_chi_values(first_df, second_df)
    n_panels = min(len(chi_values), 5)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.45), sharex=True, sharey=True)
    flat_axes = axes.ravel()
    legend_handles: list[Line2D] = []
    legend_labels: list[str] = []
    all_curves: list[PayoffCurve] = []

    for idx, chi in enumerate(chi_values[:n_panels]):
        ax = flat_axes[idx]
        first_group = first_df[np.isclose(first_df["chi"], chi)].sort_values("r")
        second_group = second_df[np.isclose(second_df["chi"], chi)].sort_values("r")
        curves = [
            PayoffCurve(
                first_model,
                f"Model_{first_model}",
                first_group,
                A_D_MODEL_COLORS.get(first_model, REFERENCE_CURVE_COLOR),
                A_D_MODEL_MARKERS.get(first_model, "o"),
            ),
            PayoffCurve(
                second_model,
                f"Model_{second_model}",
                second_group,
                A_D_MODEL_COLORS.get(second_model, REFERENCE_CURVE_COLOR),
                A_D_MODEL_MARKERS.get(second_model, "s"),
            ),
        ]
        all_curves.extend(curves)
        for curve_idx, curve in enumerate(curves):
            plot_curve(ax, curve, zorder=4 + curve_idx)
        ax.set_title(rf"$\chi={chi:g}$", fontsize=10, pad=3)
        if idx == 0:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    for ax in flat_axes[n_panels:]:
        ax.axis("off")

    legend_ax = flat_axes[-1]
    legend_ax.axis("off")
    legend_ax.legend(legend_handles, legend_labels, loc="center", handlelength=2.4, handletextpad=0.5, borderaxespad=0.0)

    for ax in flat_axes[:n_panels]:
        format_payoff_axis(ax, all_curves)
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle(f"Model {first_model} vs {second_model}: payoff", fontsize=12, y=0.995)
    fig.supxlabel(r"$r$", fontsize=12, y=0.012)
    fig.supylabel("System average payoff", fontsize=12, x=0.012)
    fig.tight_layout(rect=(0.035, 0.055, 1.0, 0.94), pad=0.45, w_pad=0.55, h_pad=0.65)
    return fig


def main_pair(first_model: str, second_model: str) -> None:
    configure_publication_style()
    root = find_project_root((first_model, second_model))
    first_df = load_payoff_summary(root, first_model)
    second_df = load_payoff_summary(root, second_model)
    fig = draw_pair_comparison(first_df, second_df, first_model, second_model)
    outputs = save_figure(fig, root / "figures" / f"payoff_{first_model}_{second_model}_comparison")
    plt.close(fig)

    print("Saved publication-ready payoff comparison:")
    for path in outputs:
        print(path)


def main_chi_pair(first_model: str, second_model: str) -> None:
    configure_publication_style()
    root = find_project_root((first_model, second_model))
    first_df = load_payoff_summary(root, first_model)
    second_df = load_payoff_summary(root, second_model)
    fig = draw_chi_facets(first_df, second_df, first_model, second_model)
    outputs = save_figure(fig, root / "figures" / f"payoff_{first_model}_{second_model}_by_chi")
    plt.close(fig)

    print("Saved publication-ready payoff comparison:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main_pair("C", "E")

