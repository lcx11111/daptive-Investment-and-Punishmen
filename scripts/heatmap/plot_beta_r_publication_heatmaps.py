from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


METRIC_LABELS = {
    "avg_payoff": "Average payoff",
    "rho_CP": r"Cooperation level, $\rho_{C+P}$",
    "rho_D": r"Defector density, $\rho_D$",
    "rho_C": r"Cooperator density, $\rho_C$",
    "rho_P": r"Punisher density, $\rho_P$",
    "avg_contribution_active": r"Mean contribution cost, $\bar{c}$",
    "avg_punishment_punishers": r"Global mean punishment cost, $\bar{a}_{D>0}$",
    "avg_punishment_punishers_conditional": r"Conditional mean punishment, $\bar{a}\mid n_P>0$",
    "avg_punishment_punishers_weighted_d_positive": r"Weighted mean punishment, $\bar{a}_{D>0}^{\mathrm{w}}$",
}

METRIC_SYMBOLS = {
    "avg_payoff": r"$\bar{\Pi}$",
    "rho_CP": r"$\rho_{C+P}$",
    "rho_D": r"$\rho_D$",
    "rho_C": r"$\rho_C$",
    "rho_P": r"$\rho_P$",
    "avg_contribution_active": r"$\bar{c}$",
    "avg_punishment_punishers": r"$\bar{a}$",
    "avg_punishment_punishers_conditional": r"$\bar{a}\mid n_P>0$",
    "avg_punishment_punishers_weighted_d_positive": r"$\bar{a}_{D>0}^{\mathrm{w}}$",
}

DEFAULT_METRICS = (
    "rho_CP",
    "avg_contribution_active",
    "avg_punishment_punishers",
)


@dataclass(frozen=True)
class HeatmapData:
    source: Path
    tag: str
    r_values: np.ndarray
    beta_values: np.ndarray
    means: dict[str, np.ndarray]
    sems: dict[str, np.ndarray]


def configure_publication_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.size": 10.5,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "black",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.top": False,
            "ytick.right": False,
            "xtick.major.size": 4.2,
            "ytick.major.size": 4.2,
            "xtick.major.width": 0.85,
            "ytick.major.width": 0.85,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def safe_stem(path: Path) -> str:
    stem = path.stem
    stem = stem.replace("doubleQ_modelE_betaF_r_heatmap_data_", "")
    stem = stem.replace("doubleQ_modelE_betaF_r_heatmap_summary_", "")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")


def choose_input_files(input_dir: Path, explicit_input: Path | None) -> list[Path]:
    if explicit_input is not None:
        if not explicit_input.exists():
            raise FileNotFoundError(f"Input file does not exist: {explicit_input}")
        return [explicit_input]

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = sorted(input_dir.glob("*.csv")) + sorted(input_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .csv or .npz files found in {input_dir}")
    return files


def _decode_metric_names(names: np.ndarray) -> list[str]:
    out: list[str] = []
    for item in names.tolist():
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


def load_npz_heatmap(path: Path) -> HeatmapData:
    with np.load(path, allow_pickle=True) as data:
        required = {"r_values", "beta_values", "mean", "metric_names"}
        missing = sorted(required - set(data.keys()))
        if missing:
            raise KeyError(f"{path} is missing keys: {missing}. Available keys: {list(data.keys())}")

        r_values = np.asarray(data["r_values"], dtype=float)
        beta_values = np.asarray(data["beta_values"], dtype=float)
        mean = np.asarray(data["mean"], dtype=float)
        metric_names = _decode_metric_names(np.asarray(data["metric_names"]))
        sem = np.asarray(data["sem"], dtype=float) if "sem" in data else None

    expected = (r_values.size, beta_values.size, len(metric_names))
    if mean.shape != expected:
        raise ValueError(f"{path} mean shape is {mean.shape}, expected {expected}")
    if sem is not None and sem.shape != expected:
        raise ValueError(f"{path} sem shape is {sem.shape}, expected {expected}")

    means = {name: mean[:, :, idx] for idx, name in enumerate(metric_names)}
    sems = {name: sem[:, :, idx] for idx, name in enumerate(metric_names)} if sem is not None else {}
    return HeatmapData(path, safe_stem(path), r_values, beta_values, means, sems)


def load_csv_heatmap(path: Path) -> HeatmapData:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"{path} is empty")

    columns = set(rows[0].keys())
    if "r" not in columns or "beta_F" not in columns:
        raise KeyError(f"{path} must contain columns 'r' and 'beta_F'")

    mean_metrics = sorted(col.removeprefix("mean_") for col in columns if col.startswith("mean_"))
    if not mean_metrics:
        raise KeyError(f"{path} must contain at least one mean_<metric> column")

    r_values = np.array(sorted({float(row["r"]) for row in rows}), dtype=float)
    beta_values = np.array(sorted({float(row["beta_F"]) for row in rows}), dtype=float)
    r_index = {value: idx for idx, value in enumerate(r_values)}
    beta_index = {value: idx for idx, value in enumerate(beta_values)}

    means = {metric: np.full((r_values.size, beta_values.size), np.nan, dtype=float) for metric in mean_metrics}
    sems = {
        metric: np.full((r_values.size, beta_values.size), np.nan, dtype=float)
        for metric in mean_metrics
        if f"sem_{metric}" in columns
    }

    for row in rows:
        i = r_index[float(row["r"])]
        j = beta_index[float(row["beta_F"])]
        for metric in mean_metrics:
            value = row.get(f"mean_{metric}", "")
            if value != "":
                means[metric][i, j] = float(value)
            sem_value = row.get(f"sem_{metric}", "")
            if metric in sems and sem_value != "":
                sems[metric][i, j] = float(sem_value)

    return HeatmapData(path, safe_stem(path), r_values, beta_values, means, sems)


def load_heatmap(path: Path) -> HeatmapData:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return load_npz_heatmap(path)
    if suffix == ".csv":
        return load_csv_heatmap(path)
    raise ValueError(f"Unsupported input type: {path}")


def metric_limits(metric: str, matrix: np.ndarray) -> tuple[float, float]:
    if metric.startswith("rho_"):
        return 0.0, 1.0
    if metric == "avg_contribution_active":
        return 0.1, 1.0
    if metric.startswith("avg_punishment"):
        return 0.0, 0.5

    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(finite, 2.0))
    vmax = float(np.nanpercentile(finite, 98.0))
    if np.isclose(vmin, vmax):
        delta = 0.5 if np.isclose(vmin, 0.0) else abs(vmin) * 0.1
        return vmin - delta, vmax + delta
    return vmin, vmax


def nice_ticks(vmin: float, vmax: float, count: int = 5) -> np.ndarray:
    return np.linspace(float(vmin), float(vmax), count)


def sparse_value_ticks(values: np.ndarray, target_ticks: int = 6) -> np.ndarray:
    step = max(1, values.size // max(1, target_ticks - 1))
    ticks = values[::step]
    if ticks.size == 0 or not np.isclose(ticks[-1], values[-1]):
        ticks = np.r_[ticks, values[-1]]
    return ticks


def selected_metrics(data: HeatmapData, metrics_arg: str, all_metrics: bool) -> list[str]:
    available = list(data.means.keys())
    if all_metrics:
        return available

    requested = [name.strip() for name in metrics_arg.split(",") if name.strip()]
    out = [name for name in requested if name in data.means]
    if out:
        return out

    out = [name for name in DEFAULT_METRICS if name in data.means]
    return out if out else available[: min(4, len(available))]


def _smooth_display_matrix(matrix: np.ndarray, smooth_sigma: float) -> np.ndarray:
    if smooth_sigma <= 0:
        return matrix
    try:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(matrix, sigma=smooth_sigma, mode="nearest")
    except Exception:
        return matrix


def draw_one_heatmap(
    ax: plt.Axes,
    data: HeatmapData,
    metric: str,
    cmap: str,
    smooth_sigma: float,
    interpolation: str,
) -> mpl.image.AxesImage:
    matrix = _smooth_display_matrix(np.asarray(data.means[metric], dtype=float), smooth_sigma)
    vmin, vmax = metric_limits(metric, matrix)

    image = ax.imshow(
        matrix.T,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[
            float(data.r_values[0]),
            float(data.r_values[-1]),
            float(data.beta_values[0]),
            float(data.beta_values[-1]),
        ],
        origin="lower",
        aspect="auto",
        interpolation=interpolation,
    )

    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\beta_F$")
    ax.set_xlim(float(data.r_values.min()), float(data.r_values.max()))
    ax.set_ylim(float(data.beta_values.min()), float(data.beta_values.max()))
    x_ticks = sparse_value_ticks(data.r_values)
    y_ticks = sparse_value_ticks(data.beta_values)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([f"{value:.1f}" for value in x_ticks], rotation=45, ha="right")
    ax.set_yticklabels([f"{value:.1f}" for value in y_ticks])
    return image


def save_figure(fig: plt.Figure, base: Path, dpi: int) -> list[Path]:
    paths: list[Path] = []
    fig.tight_layout(pad=1.1)
    for suffix in (".pdf", ".svg", ".png", ".tiff"):
        path = base.with_suffix(suffix)
        fig.savefig(path, dpi=dpi, facecolor="white")
        paths.append(path)
    return paths


def plot_multipanel(
    data: HeatmapData,
    metrics: list[str],
    output_dir: Path,
    cmap: str,
    smooth_sigma: float,
    interpolation: str,
    dpi: int,
) -> list[Path]:
    n = len(metrics)
    fig_width = max(8.0, 4.6 * n)
    fig, axes = plt.subplots(1, n, figsize=(fig_width, 4.2), squeeze=False)

    for idx, metric in enumerate(metrics):
        ax = axes[0, idx]
        image = draw_one_heatmap(ax, data, metric, cmap, smooth_sigma, interpolation)
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
        cbar.ax.set_title(METRIC_SYMBOLS.get(metric, metric), pad=6)
        cbar.ax.tick_params(length=3.0, width=0.75, labelsize=9)

    base = output_dir / f"{data.tag}_publication_heatmaps"
    paths = save_figure(fig, base, dpi)
    plt.close(fig)
    return paths


def plot_individual(
    data: HeatmapData,
    metrics: list[str],
    output_dir: Path,
    cmap: str,
    smooth_sigma: float,
    interpolation: str,
    dpi: int,
) -> list[Path]:
    paths: list[Path] = []
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8.0, 7.0))
        image = draw_one_heatmap(ax, data, metric, cmap, smooth_sigma, interpolation)
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_title(METRIC_SYMBOLS.get(metric, metric), pad=7)
        cbar.ax.tick_params(length=3.4, width=0.85, labelsize=11)
        base = output_dir / f"{data.tag}_{metric}_heatmap"
        paths.extend(save_figure(fig, base, dpi))
        plt.close(fig)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read beta_F-r heatmap files from data/beta_r and redraw publication-quality heatmaps. "
            "CSV files with r, beta_F, mean_<metric> columns and NPZ files with r_values, beta_values, "
            "mean, metric_names are supported."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data") / "beta_r")
    parser.add_argument("--input", type=Path, default=None, help="Optional single .csv or .npz file.")
    parser.add_argument("--output-dir", type=Path, default=Path("figures") / "beta_r_publication")
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metric names. Missing metrics are skipped.",
    )
    parser.add_argument("--all-metrics", action="store_true", help="Plot every metric available in the input file.")
    parser.add_argument("--cmap", type=str, default="jet", help="Colormap, e.g. jet, turbo, viridis, plasma, magma.")
    parser.add_argument(
        "--interpolation",
        type=str,
        default="bilinear",
        help="imshow interpolation, e.g. nearest, bilinear, bicubic.",
    )
    parser.add_argument("--levels", type=int, default=256, help="Kept for compatibility; imshow mode ignores it.")
    parser.add_argument("--smooth-sigma", type=float, default=0.5, help="Display-only Gaussian smoothing.")
    parser.add_argument("--dpi", type=int, default=1500)
    parser.add_argument("--no-individual", action="store_true", help="Only save the combined multi-panel figure.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_publication_style()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_files = choose_input_files(args.input_dir, args.input)

    saved: list[Path] = []
    for input_file in input_files:
        data = load_heatmap(input_file)
        metrics = selected_metrics(data, args.metrics, args.all_metrics)
        if not metrics:
            print(f"Skipped {input_file}: no matching metrics.")
            continue

        saved.extend(
            plot_multipanel(
                data=data,
                metrics=metrics,
                output_dir=args.output_dir,
                cmap=args.cmap,
                smooth_sigma=max(0.0, args.smooth_sigma),
                interpolation=args.interpolation,
                dpi=args.dpi,
            )
        )
        if not args.no_individual:
            saved.extend(
                plot_individual(
                    data=data,
                    metrics=metrics,
                    output_dir=args.output_dir,
                    cmap=args.cmap,
                    smooth_sigma=max(0.0, args.smooth_sigma),
                    interpolation=args.interpolation,
                    dpi=args.dpi,
                )
            )

    print("Saved beta_F-r heatmap figures:")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
