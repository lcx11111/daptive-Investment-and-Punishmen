from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit

import evo


MM = 1 / 25.4
DEFAULT_OUTPUT_DIR = Path("data/Q_max")
DEFAULT_R = 2.1


@njit(cache=True)
def simulate_q_tables_numba(
    L: int,
    T: int,
    r_value: float,
    beta_F: float,
    K: float,
    alpha_c: float,
    gamma_c: float,
    alpha_a: float,
    gamma_a: float,
    epsilon_c0: float,
    epsilon_c_min: float,
    epsilon_c_dcy: float,
    epsilon_a0: float,
    epsilon_a_min: float,
    epsilon_a_dcy: float,
    init_D: float,
    init_C: float,
    init_P: float,
    seed: int,
    contribution_actions: np.ndarray,
    punishment_actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)

    z = np.empty((L, L), dtype=np.int8)
    z_next = np.empty((L, L), dtype=np.int8)
    q_c = np.zeros((L, L, evo.NUM_STATES, contribution_actions.shape[0]), dtype=np.float64)
    q_a = np.zeros(
        (L, L, evo.NUM_STATES, evo.NUM_PUNISHER_STATES, punishment_actions.shape[0]),
        dtype=np.float64,
    )

    state_d_old = np.empty((L, L), dtype=np.int64)
    state_p_old = np.empty((L, L), dtype=np.int64)
    state_cp_old = np.empty((L, L), dtype=np.int64)
    state_d_next = np.empty((L, L), dtype=np.int64)
    state_p_next = np.empty((L, L), dtype=np.int64)
    state_cp_next = np.empty((L, L), dtype=np.int64)

    contribution_idx = np.zeros((L, L), dtype=np.int64)
    punishment_idx = np.zeros((L, L), dtype=np.int64)
    contribution_value = np.zeros((L, L), dtype=np.float64)
    punishment_value = np.zeros((L, L), dtype=np.float64)
    payoff = np.zeros((L, L), dtype=np.float64)

    threshold_D = init_D
    threshold_C = init_D + init_C
    _ = init_P
    for i in range(L):
        for j in range(L):
            u = np.random.random()
            if u < threshold_D:
                z[i, j] = evo.D
            elif u < threshold_C:
                z[i, j] = evo.C
            else:
                z[i, j] = evo.P

    epsilon_c = epsilon_c0
    epsilon_a = epsilon_a0
    for _step in range(1, T + 1):
        evo.compute_neighbor_states(z, state_d_old, state_p_old, state_cp_old)
        evo.choose_actions(
            z,
            state_d_old,
            state_p_old,
            state_cp_old,
            q_c,
            q_a,
            contribution_idx,
            contribution_value,
            punishment_idx,
            punishment_value,
            epsilon_c,
            epsilon_a,
            contribution_actions,
            punishment_actions,
        )
        evo.compute_payoff_numba(z, contribution_value, punishment_value, payoff, r_value, beta_F)
        evo.strategy_update_numba(z, payoff, z_next, K)
        evo.compute_neighbor_states(z_next, state_d_next, state_p_next, state_cp_next)
        evo.update_q_tables_numba(
            z,
            state_d_old,
            state_p_old,
            state_cp_old,
            state_d_next,
            state_p_next,
            state_cp_next,
            q_c,
            q_a,
            contribution_idx,
            punishment_idx,
            payoff,
            alpha_c,
            gamma_c,
            alpha_a,
            gamma_a,
            contribution_actions,
            punishment_actions,
        )

        epsilon_c = max(epsilon_c * epsilon_c_dcy, epsilon_c_min)
        epsilon_a = max(epsilon_a * epsilon_a_dcy, epsilon_a_min)
        evo.copy_strategy(z_next, z)

    q_c_mean = np.zeros((evo.NUM_STATES, contribution_actions.shape[0]), dtype=np.float64)
    q_a_mean = np.zeros(
        (evo.NUM_STATES, evo.NUM_PUNISHER_STATES, punishment_actions.shape[0]),
        dtype=np.float64,
    )
    total_sites = L * L
    for i in range(L):
        for j in range(L):
            for state_cp in range(evo.NUM_STATES):
                for action_idx in range(contribution_actions.shape[0]):
                    q_c_mean[state_cp, action_idx] += q_c[i, j, state_cp, action_idx] / total_sites
            for state_d in range(evo.NUM_STATES):
                for state_p in range(evo.NUM_PUNISHER_STATES):
                    for action_idx in range(punishment_actions.shape[0]):
                        q_a_mean[state_d, state_p, action_idx] += (
                            q_a[i, j, state_d, state_p, action_idx] / total_sites
                        )

    return q_c_mean, q_a_mean


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "mathtext.fontset": "dejavusans",
            "font.size": 7.0,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.70,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.top": False,
            "ytick.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def fmt_action(value: float) -> str:
    return "1.0" if np.isclose(float(value), 1.0) else f"{float(value):.1f}"


def tag_for(params: evo.EvolutionParams) -> str:
    return (
        f"model_E_Qmax"
        f"_r{evo.format_value(params.r)}"
        f"_L{params.L}_T{params.T}"
        f"_runs{params.runs}"
    )


def valid_punishment_pairs() -> list[tuple[int, int]]:
    pairs = []
    neighbor_count = evo.GROUP_SIZE - 1
    for state_d in range(evo.NUM_STATES):
        for state_p in range(evo.NUM_PUNISHER_STATES):
            if state_d + state_p <= neighbor_count:
                pairs.append((state_d, state_p))
    return pairs


def q_value_frames(
    q_c: np.ndarray,
    q_a: np.ndarray,
    contribution_actions: np.ndarray,
    punishment_actions: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    investment_rows = []
    for state_cp in range(q_c.shape[0]):
        for action_index, action_value in enumerate(contribution_actions):
            investment_rows.append(
                {
                    "table": "investment",
                    "state_cp": state_cp,
                    "state_d": evo.GROUP_SIZE - 1 - state_cp,
                    "action_index": action_index,
                    "action_value": float(action_value),
                    "mean_q": float(q_c[state_cp, action_index]),
                }
            )

    punishment_rows = []
    for state_d, state_p in valid_punishment_pairs():
        for action_index, action_value in enumerate(punishment_actions):
            punishment_rows.append(
                {
                    "table": "punishment",
                    "state_d": state_d,
                    "state_p": state_p,
                    "action_index": action_index,
                    "action_value": float(action_value),
                    "mean_q": float(q_a[state_d, state_p, action_index]),
                }
            )

    return pd.DataFrame(investment_rows), pd.DataFrame(punishment_rows)


def best_actions(
    investment_q: pd.DataFrame,
    punishment_q: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inv_rows = []
    for state_cp, group in investment_q.groupby("state_cp", sort=True):
        ranked = group.sort_values("mean_q", ascending=False).reset_index(drop=True)
        inv_rows.append(
            {
                "state_cp": int(state_cp),
                "state_d": int(evo.GROUP_SIZE - 1 - state_cp),
                "q_maximizing_c": float(ranked.loc[0, "action_value"]),
                "max_mean_q": float(ranked.loc[0, "mean_q"]),
                "q_gap": float(ranked.loc[0, "mean_q"] - ranked.loc[1, "mean_q"])
                if len(ranked) > 1
                else np.nan,
            }
        )

    pun_rows = []
    for (state_d, state_p), group in punishment_q.groupby(["state_d", "state_p"], sort=True):
        ranked = group.sort_values("mean_q", ascending=False).reset_index(drop=True)
        pun_rows.append(
            {
                "state_d": int(state_d),
                "state_p": int(state_p),
                "q_maximizing_a": float(ranked.loc[0, "action_value"]),
                "max_mean_q": float(ranked.loc[0, "mean_q"]),
                "q_gap": float(ranked.loc[0, "mean_q"] - ranked.loc[1, "mean_q"])
                if len(ranked) > 1
                else np.nan,
            }
        )

    inv_best = pd.DataFrame(inv_rows).sort_values("state_d").reset_index(drop=True)
    pun_best = pd.DataFrame(pun_rows).sort_values(["state_d", "state_p"]).reset_index(drop=True)
    return inv_best, pun_best


def add_state_table(
    ax: plt.Axes,
    rows: list[list[str]],
    row_labels: list[str],
    *,
    bbox: list[float],
) -> None:
    table = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        cellLoc="center",
        rowLoc="center",
        loc="bottom",
        bbox=bbox,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.2)
    for cell in table.get_celld().values():
        cell.set_edgecolor("#666666")
        cell.set_linewidth(0.55)
        cell.set_facecolor("white")


def draw_table_cell(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    fontsize: float = 6.2,
) -> None:
    ax.add_patch(
        plt.Rectangle(
            (x, y),
            width,
            height,
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="#666666",
            linewidth=0.55,
            clip_on=False,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#202124",
        clip_on=False,
    )


def add_punishment_state_table(ax: plt.Axes, summary: pd.DataFrame, *, bbox: list[float]) -> None:
    x0, y0, width, height = bbox
    n_cols = len(summary)
    cell_w = width / n_cols
    row_h = height / 2
    label_w = min(0.055, cell_w * 0.72)

    draw_table_cell(ax, x0 - label_w, y0 + row_h, label_w, row_h, r"$n_P$")
    draw_table_cell(ax, x0 - label_w, y0, label_w, row_h, r"$n_D$")

    for col, value in enumerate(summary["state_p"]):
        draw_table_cell(ax, x0 + col * cell_w, y0 + row_h, cell_w, row_h, str(int(value)))

    start = 0
    for state_d, group in summary.groupby("state_d", sort=True):
        span = len(group)
        draw_table_cell(
            ax,
            x0 + start * cell_w,
            y0,
            span * cell_w,
            row_h,
            str(int(state_d)),
        )
        start += span


def style_axis(ax: plt.Axes, y_max: float, y_ticks: list[float]) -> None:
    ax.set_ylim(0, y_max)
    ax.set_yticks(y_ticks)
    ax.grid(axis="y", color="#E3E7EB", linewidth=0.55, zorder=0)
    ax.tick_params(axis="both", direction="out", length=2.8, width=0.65, pad=1.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.70)


def annotate_bars(ax: plt.Axes, x: np.ndarray, values: np.ndarray, offset: float) -> None:
    for xpos, value in zip(x, values):
        ax.text(
            xpos,
            float(value) + offset,
            fmt_action(float(value)),
            ha="center",
            va="bottom",
            fontsize=6.2,
            color="#202124",
        )


def draw_qmax_action_figure(inv_best: pd.DataFrame, pun_best: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(178 * MM, 64 * MM),
        dpi=300,
        gridspec_kw={"width_ratios": [0.85, 1.75]},
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.28, top=0.94, wspace=0.30)

    inv = inv_best.sort_values("state_d").reset_index(drop=True)
    x_inv = np.arange(len(inv))
    c_values = inv["q_maximizing_c"].to_numpy(dtype=float)
    axes[0].bar(
        x_inv,
        c_values,
        width=0.68,
        color="#F2A000",
        edgecolor="#4A3A16",
        linewidth=0.55,
        zorder=3,
    )
    annotate_bars(axes[0], x_inv, c_values, 0.025)
    style_axis(axes[0], 1.12, [0, 0.5, 1.0])
    axes[0].set_xlim(-0.60, len(inv) - 0.40)
    axes[0].set_xticks([])
    axes[0].set_ylabel(r"Q-maximizing investment action, $c_Q^*$")
    add_state_table(
        axes[0],
        [[str(int(value)) for value in inv["state_cp"]]],
        [r"$n_{CP}$"],
        bbox=[0.0, -0.26, 1.0, 0.18],
    )

    pun = pun_best.sort_values(["state_d", "state_p"]).reset_index(drop=True)
    x_pun = np.arange(len(pun))
    a_values = pun["q_maximizing_a"].to_numpy(dtype=float)
    axes[1].bar(
        x_pun,
        a_values,
        width=0.68,
        color="#2E86C1",
        edgecolor="#1F4E66",
        linewidth=0.52,
        zorder=3,
    )
    annotate_bars(axes[1], x_pun, a_values, 0.012)
    style_axis(axes[1], 0.58, [0, 0.25, 0.5])
    axes[1].set_xlim(-0.60, len(pun) - 0.40)
    axes[1].set_xticks([])
    axes[1].set_ylabel(r"Q-maximizing punishment action, $a_Q^*$")

    boundaries = np.cumsum(pun.groupby("state_d", sort=True).size().to_numpy())[:-1]
    for boundary in boundaries:
        axes[1].axvline(boundary - 0.5, color="#8A8A8A", linewidth=0.55, zorder=2)

    add_punishment_state_table(axes[1], pun, bbox=[0.0, -0.34, 1.0, 0.26])

    return fig


def save_figure(fig: plt.Figure, out_base: Path, dpi: int) -> list[Path]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in [".svg", ".pdf", ".png", ".tiff"]:
        path = out_base.with_suffix(suffix)
        if suffix == ".tiff":
            fig.savefig(path, dpi=dpi, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
        else:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def collect_q_tables(
    params: evo.EvolutionParams,
    contribution_actions: np.ndarray,
    punishment_actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    init_D, init_C, init_P = evo.validate_params(params)
    q_c_sum = np.zeros((evo.NUM_STATES, contribution_actions.shape[0]), dtype=np.float64)
    q_a_sum = np.zeros(
        (evo.NUM_STATES, evo.NUM_PUNISHER_STATES, punishment_actions.shape[0]),
        dtype=np.float64,
    )

    for run_idx in range(int(params.runs)):
        seed = int(params.seed_start) + run_idx
        print(f"running Q-max run {run_idx + 1}/{params.runs}, seed={seed}...")
        start = time.perf_counter()
        q_c_run, q_a_run = simulate_q_tables_numba(
            int(params.L),
            int(params.T),
            float(params.r),
            float(params.beta_F),
            float(params.K),
            float(params.alpha_c),
            float(params.gamma_c),
            float(params.alpha_a),
            float(params.gamma_a),
            float(params.epsilon_c0),
            float(params.epsilon_c_min),
            float(params.epsilon_c_dcy),
            float(params.epsilon_a0),
            float(params.epsilon_a_min),
            float(params.epsilon_a_dcy),
            init_D,
            init_C,
            init_P,
            seed,
            contribution_actions,
            punishment_actions,
        )
        q_c_sum += q_c_run
        q_a_sum += q_a_run
        print(f"run {run_idx + 1}/{params.runs} done: {time.perf_counter() - start:.2f}s")

    return q_c_sum / params.runs, q_a_sum / params.runs


def warmup_qmax() -> None:
    simulate_q_tables_numba(
        4,
        2,
        2.5,
        1.0,
        0.5,
        0.8,
        0.8,
        0.8,
        0.8,
        0.5,
        0.05,
        0.99,
        0.5,
        0.05,
        0.99,
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 3.0,
        123,
        evo.CONTRIBUTION_ACTIONS,
        evo.PUNISHMENT_ACTIONS,
    )


def run_qmax(params: evo.EvolutionParams) -> tuple[list[Path], list[Path]]:
    configure_style()
    contribution_actions = evo.parse_float_list(params.contribution_actions)
    punishment_actions = evo.parse_float_list(params.punishment_actions)

    if params.warmup:
        print("compiling numba functions...")
        start = time.perf_counter()
        warmup_qmax()
        print(f"numba warmup done: {time.perf_counter() - start:.2f}s")

    q_c, q_a = collect_q_tables(params, contribution_actions, punishment_actions)
    investment_q, punishment_q = q_value_frames(q_c, q_a, contribution_actions, punishment_actions)
    inv_best, pun_best = best_actions(investment_q, punishment_q)

    params.output_dir.mkdir(parents=True, exist_ok=True)
    tag = tag_for(params)
    csv_paths = [
        params.output_dir / f"{tag}_investment_q_values.csv",
        params.output_dir / f"{tag}_punishment_q_values.csv",
        params.output_dir / f"{tag}_investment_qmax_actions.csv",
        params.output_dir / f"{tag}_punishment_qmax_actions.csv",
    ]
    investment_q.to_csv(csv_paths[0], index=False)
    punishment_q.to_csv(csv_paths[1], index=False)
    inv_best.to_csv(csv_paths[2], index=False)
    pun_best.to_csv(csv_paths[3], index=False)

    figure = draw_qmax_action_figure(inv_best, pun_best)
    figure_paths = save_figure(figure, params.output_dir / f"{tag}_qmax_action_bars", params.dpi)

    print("Saved Q-max tables and figures:")
    for path in csv_paths + figure_paths:
        print(path)
    return csv_paths, figure_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Save Model E mean Q tables and Q-maximizing action bars.")
    parser.add_argument("--L", type=int, default=evo.CONFIG.L)
    parser.add_argument("--T", type=int, default=evo.CONFIG.T)
    parser.add_argument("--r", type=float, default=DEFAULT_R)
    parser.add_argument("--seed", "--seed-start", dest="seed_start", type=int, default=evo.CONFIG.seed_start)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--beta-F", type=float, default=evo.CONFIG.beta_F)
    parser.add_argument("--K", type=float, default=evo.CONFIG.K)
    parser.add_argument("--eta-c", "--alpha-c", dest="alpha_c", type=float, default=evo.CONFIG.alpha_c)
    parser.add_argument("--gamma-c", type=float, default=evo.CONFIG.gamma_c)
    parser.add_argument("--eta-a", "--alpha-a", dest="alpha_a", type=float, default=evo.CONFIG.alpha_a)
    parser.add_argument("--gamma-a", type=float, default=evo.CONFIG.gamma_a)
    parser.add_argument("--epsilon-c0", type=float, default=evo.CONFIG.epsilon_c0)
    parser.add_argument("--epsilon-c-min", type=float, default=evo.CONFIG.epsilon_c_min)
    parser.add_argument("--epsilon-c-dcy", type=float, default=evo.CONFIG.epsilon_c_dcy)
    parser.add_argument("--epsilon-a0", type=float, default=evo.CONFIG.epsilon_a0)
    parser.add_argument("--epsilon-a-min", type=float, default=evo.CONFIG.epsilon_a_min)
    parser.add_argument("--epsilon-a-dcy", type=float, default=evo.CONFIG.epsilon_a_dcy)
    parser.add_argument("--init-D", type=float, default=evo.CONFIG.init_D)
    parser.add_argument("--init-C", type=float, default=evo.CONFIG.init_C)
    parser.add_argument("--init-P", type=float, default=evo.CONFIG.init_P)
    parser.add_argument("--contribution-actions", type=str, default=evo.CONFIG.contribution_actions)
    parser.add_argument("--punishment-actions", type=str, default=evo.CONFIG.punishment_actions)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=evo.CONFIG.dpi)
    parser.add_argument("--no-warmup", action="store_true")
    return parser


def params_from_args(args: argparse.Namespace) -> evo.EvolutionParams:
    return replace(
        evo.CONFIG,
        L=args.L,
        T=args.T,
        r=args.r,
        seed=args.seed_start,
        seed_start=args.seed_start,
        runs=args.runs,
        beta_F=args.beta_F,
        K=args.K,
        alpha_c=args.alpha_c,
        gamma_c=args.gamma_c,
        alpha_a=args.alpha_a,
        gamma_a=args.gamma_a,
        epsilon_c0=args.epsilon_c0,
        epsilon_c_min=args.epsilon_c_min,
        epsilon_c_dcy=args.epsilon_c_dcy,
        epsilon_a0=args.epsilon_a0,
        epsilon_a_min=args.epsilon_a_min,
        epsilon_a_dcy=args.epsilon_a_dcy,
        init_D=args.init_D,
        init_C=args.init_C,
        init_P=args.init_P,
        contribution_actions=args.contribution_actions,
        punishment_actions=args.punishment_actions,
        output_dir=args.output_dir,
        dpi=args.dpi,
        warmup=not args.no_warmup,
    )


def main() -> None:
    params = params_from_args(build_parser().parse_args())
    run_qmax(params)


if __name__ == "__main__":
    main()
