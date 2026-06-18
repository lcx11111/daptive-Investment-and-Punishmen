from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from numba import njit

from evo_cost import (
    C,
    CONFIG as COST_CONFIG,
    D,
    P,
    EvolutionParams,
    compute_neighbor_states,
    compute_payoff_numba,
    configure_publication_style,
    copy_strategy,
    choose_actions,
    format_value,
    parse_float_list,
    save_one_figure,
    strategy_update_numba,
    update_q_tables_numba,
    validate_params,
    warmup_cost_numba,
)


CONFIG = replace(COST_CONFIG, output_dir=Path("data/evolution_fig4_action_distribution"))

# Keep action-distribution panels visually comparable across action spaces.
# Investment has 10 actions, punishment has 5; scaling the categorical bar width
# makes the bars have similar physical thickness in the final subplot.
REFERENCE_ACTION_COUNT = 10
REFERENCE_BAR_WIDTH = 0.78
INVESTMENT_BAR_COLOR = "#E3A12F"
PUNISHMENT_BAR_COLOR = "#D39A3A"
DISTRIBUTION_LINE_COLOR = "#D65F5F"


def action_bar_width(actions: np.ndarray) -> float:
    return REFERENCE_BAR_WIDTH * min(1.0, actions.size / REFERENCE_ACTION_COUNT)


@njit(cache=True)
def simulate_fig4_cp_investment_distribution_numba(
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
    snapshot_steps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fig.4-style single-time snapshot statistics.

    Statistical object:
        Current cooperators C and punishers P are merged into one group C+P.
    Statistical variable:
        Their current investment action e, i.e. contribution_idx / contribution_actions.
    Output:
        investment_counts_cp[s, a] = number of C+P agents whose investment action is action a
                                     at snapshot_steps[s].
        punishment_counts_p[s, a] = number of P agents whose punishment action is action a
                                    at snapshot_steps[s].
        strategy_counts[s, 0/1/2] = D/C/P headcounts at snapshot_steps[s].
    """
    np.random.seed(seed)

    z = np.empty((L, L), dtype=np.int8)
    z_next = np.empty((L, L), dtype=np.int8)

    q_c = np.zeros((L, L, 5, contribution_actions.shape[0]), dtype=np.float64)
    q_a = np.zeros((L, L, 5, 5, punishment_actions.shape[0]), dtype=np.float64)

    state_d_old = np.empty((L, L), dtype=np.int64)
    state_p_old = np.empty((L, L), dtype=np.int64)
    state_cp_old = np.empty((L, L), dtype=np.int64)
    state_d_next = np.empty((L, L), dtype=np.int64)
    state_p_next = np.empty((L, L), dtype=np.int64)
    state_cp_next = np.empty((L, L), dtype=np.int64)

    contribution_idx = np.zeros((L, L), dtype=np.int64)
    contribution_value = np.zeros((L, L), dtype=np.float64)
    punishment_idx = np.zeros((L, L), dtype=np.int64)
    punishment_value = np.zeros((L, L), dtype=np.float64)
    payoff = np.zeros((L, L), dtype=np.float64)

    n_snapshots = snapshot_steps.shape[0]
    investment_counts_cp = np.zeros((n_snapshots, contribution_actions.shape[0]), dtype=np.float64)
    punishment_counts_p = np.zeros((n_snapshots, punishment_actions.shape[0]), dtype=np.float64)
    strategy_counts = np.zeros((n_snapshots, 3), dtype=np.float64)

    threshold_D = init_D
    threshold_C = init_D + init_C
    _ = init_P

    for i in range(L):
        for j in range(L):
            u = np.random.random()
            if u < threshold_D:
                z[i, j] = D
            elif u < threshold_C:
                z[i, j] = C
            else:
                z[i, j] = P

    epsilon_c = epsilon_c0
    epsilon_a = epsilon_a0
    snapshot_pos = 0

    for step in range(1, T + 1):
        compute_neighbor_states(z, state_d_old, state_p_old, state_cp_old)
        choose_actions(
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

        # 与论文 Fig.4 的“给定 T 时刻分布”口径一致：
        # 在当前 step 的策略状态和动作选择确定后，统计该时刻的 e 分布。
        while snapshot_pos < n_snapshots and step == snapshot_steps[snapshot_pos]:
            for i in range(L):
                for j in range(L):
                    strategy = z[i, j]
                    strategy_counts[snapshot_pos, strategy] += 1.0

                    # 惩罚者 P 与合作者 C 合并，统一统计投资动作 e。
                    if strategy == C or strategy == P:
                        action_id = contribution_idx[i, j]
                        investment_counts_cp[snapshot_pos, action_id] += 1.0
                    if strategy == P:
                        action_id = punishment_idx[i, j]
                        punishment_counts_p[snapshot_pos, action_id] += 1.0

            snapshot_pos += 1

        compute_payoff_numba(z, contribution_value, punishment_value, payoff, r_value, beta_F)
        strategy_update_numba(z, payoff, z_next, K)
        compute_neighbor_states(z_next, state_d_next, state_p_next, state_cp_next)
        update_q_tables_numba(
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
        copy_strategy(z_next, z)

    return investment_counts_cp, punishment_counts_p, strategy_counts


def parse_int_list(raw: str) -> np.ndarray:
    values: list[int] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        values.append(int(item))

    if not values:
        raise ValueError("snapshot_steps must contain at least one positive integer.")

    unique_sorted = sorted(set(values))
    if unique_sorted[0] <= 0:
        raise ValueError("snapshot_steps must be positive integers, e.g. 10,100,1000,10000.")

    return np.array(unique_sorted, dtype=np.int64)


def action_tag(actions: np.ndarray) -> str:
    return "-".join(format_value(float(value)) for value in actions)


def snapshot_tag(snapshot_steps: np.ndarray) -> str:
    return "-".join(str(int(step)) for step in snapshot_steps)


def build_tag(
    params: EvolutionParams,
    contribution_actions: np.ndarray,
    punishment_actions: np.ndarray,
    snapshot_steps: np.ndarray,
) -> str:
    return (
        f"modelE_fig4_action_distribution_L{params.L}_T{params.T}_r{format_value(params.r)}"
        f"_runs{params.runs}_seed{params.seed_start}_bF{format_value(params.beta_F)}"
        f"_snap{snapshot_tag(snapshot_steps)}"
        f"_Ac{action_tag(contribution_actions)}_Aa{action_tag(punishment_actions)}"
    )


def proportions(counts: np.ndarray) -> np.ndarray:
    total = float(np.sum(counts))
    if total <= 0.0:
        return np.zeros_like(counts, dtype=np.float64)
    return counts / total


def save_distribution_csv(
    path: Path,
    snapshot_steps: np.ndarray,
    contribution_actions: np.ndarray,
    investment_counts_cp_mean: np.ndarray,
    strategy_counts_mean: np.ndarray,
) -> None:
    rows: list[tuple[int, float, float, float, float, float, float, float]] = []

    for snap_idx, step in enumerate(snapshot_steps):
        counts = investment_counts_cp_mean[snap_idx]
        props = proportions(counts)

        d_mean = float(strategy_counts_mean[snap_idx, D])
        c_mean = float(strategy_counts_mean[snap_idx, C])
        p_mean = float(strategy_counts_mean[snap_idx, P])
        cp_mean = c_mean + p_mean

        for action, count, prop in zip(contribution_actions, counts, props):
            rows.append(
                (
                    int(step),
                    float(action),
                    float(count),
                    float(prop),
                    d_mean,
                    c_mean,
                    p_mean,
                    cp_mean,
                )
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "T,investment_e,mean_count_CplusP,within_CplusP_proportion,"
        "mean_D_count,mean_C_count,mean_P_count,mean_CplusP_count"
    )
    np.savetxt(
        path,
        np.array(rows, dtype=object),
        delimiter=",",
        header=header,
        comments="",
        fmt=["%d", "%.10f", "%.10f", "%.10f", "%.10f", "%.10f", "%.10f", "%.10f"],
    )


def save_punishment_distribution_csv(
    path: Path,
    snapshot_steps: np.ndarray,
    punishment_actions: np.ndarray,
    punishment_counts_p_mean: np.ndarray,
    strategy_counts_mean: np.ndarray,
) -> None:
    rows: list[tuple[int, float, float, float, float, float, float]] = []

    for snap_idx, step in enumerate(snapshot_steps):
        counts = punishment_counts_p_mean[snap_idx]
        props = proportions(counts)

        d_mean = float(strategy_counts_mean[snap_idx, D])
        c_mean = float(strategy_counts_mean[snap_idx, C])
        p_mean = float(strategy_counts_mean[snap_idx, P])

        for action, count, prop in zip(punishment_actions, counts, props):
            rows.append(
                (
                    int(step),
                    float(action),
                    float(count),
                    float(prop),
                    d_mean,
                    c_mean,
                    p_mean,
                )
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "T,punishment_a,mean_count_P,within_P_proportion,"
        "mean_D_count,mean_C_count,mean_P_count"
    )
    np.savetxt(
        path,
        np.array(rows, dtype=object),
        delimiter=",",
        header=header,
        comments="",
        fmt=["%d", "%.10f", "%.10f", "%.10f", "%.10f", "%.10f", "%.10f"],
    )


def _grid_shape(n_panels: int) -> tuple[int, int]:
    if n_panels <= 1:
        return 1, 1
    if n_panels <= 4:
        return 1, n_panels
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols
    return nrows, ncols


def plot_fig4_panel(
    ax: plt.Axes,
    actions: np.ndarray,
    counts: np.ndarray,
    step: int,
    x_label: str,
    bar_color: str = "#F4A300",
    show_left_axis: bool = True,
    show_right_axis: bool = True,
) -> None:
    x = np.arange(actions.size)
    fractions = proportions(counts)
    props_pct = fractions * 100.0

    # Fig.4-style bar distribution with the percentage overlaid as a dashed line.
    ax.bar(
        x,
        fractions,
        width=action_bar_width(actions),
        color=bar_color,
        edgecolor="#4A3824",
        linewidth=0.5,
        zorder=2,
    )
    ax.set_title(rf"$T={int(step)}$", pad=4)
    ax.set_xlabel(x_label)
    ax.set_ylabel("fraction" if show_left_axis else "")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{float(value):g}" for value in actions])
    ax.grid(axis="y", linewidth=0.35, alpha=0.35, zorder=0)
    ax.tick_params(axis="y", labelleft=show_left_axis)

    ax_prop = ax.twinx()
    ax_prop.plot(
        x,
        props_pct,
        color=DISTRIBUTION_LINE_COLOR,
        marker="o",
        markersize=3.1,
        linewidth=1.05,
        linestyle="--",
        zorder=3,
    )
    ax_prop.set_ylabel("Percentage" if show_right_axis else "")
    ax_prop.set_ylim(0.0, 100.0)
    ax_prop.yaxis.set_major_formatter(PercentFormatter(xmax=100.0))
    ax_prop.tick_params(axis="y", right=show_right_axis, labelright=show_right_axis)
    ax_prop.spines["right"].set_visible(show_right_axis)


def save_distribution_figure(
    output_dir: Path,
    tag: str,
    dpi: int,
    snapshot_steps: np.ndarray,
    actions: np.ndarray,
    counts_by_snapshot: np.ndarray,
    file_suffix: str,
    x_label: str,
    bar_color: str = "#F4A300",
) -> list[Path]:
    configure_publication_style()

    n_panels = int(snapshot_steps.size)
    nrows, ncols = _grid_shape(n_panels)
    fig_width = 3.35 * ncols if nrows == 1 else (8.3 if ncols == 2 else 4.25 * ncols)
    fig_height = 2.95 * nrows
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        sharey=True,
        constrained_layout=True,
    )

    axes_flat = np.atleast_1d(axes).ravel()
    for idx, ax in enumerate(axes_flat):
        if idx < n_panels:
            plot_fig4_panel(
                ax,
                actions,
                counts_by_snapshot[idx],
                int(snapshot_steps[idx]),
                x_label,
                bar_color,
                show_left_axis=(idx == 0),
                show_right_axis=(idx == n_panels - 1),
            )
        else:
            ax.axis("off")

    paths = save_one_figure(fig, output_dir / f"{tag}_{file_suffix}", dpi)
    plt.close(fig)
    return paths


def run_distribution(
    params: EvolutionParams,
    snapshot_steps: np.ndarray,
) -> tuple[list[Path], list[Path]]:
    if snapshot_steps.size <= 0:
        raise ValueError("snapshot_steps must not be empty.")
    if int(snapshot_steps[-1]) > int(params.T):
        raise ValueError(
            f"The largest snapshot T={int(snapshot_steps[-1])} exceeds simulation T={int(params.T)}. "
            "Please increase --T or reduce --snapshot-steps."
        )

    init_D, init_C, init_P = validate_params(params)
    contribution_actions = parse_float_list(params.contribution_actions)
    punishment_actions = parse_float_list(params.punishment_actions)
    params.output_dir.mkdir(parents=True, exist_ok=True)

    if params.warmup:
        print("compiling numba functions...")
        start = time.perf_counter()
        warmup_cost_numba()
        print(f"numba warmup done: {time.perf_counter() - start:.2f}s")

    investment_counts_cp_sum = np.zeros(
        (snapshot_steps.size, contribution_actions.size),
        dtype=np.float64,
    )
    punishment_counts_p_sum = np.zeros(
        (snapshot_steps.size, punishment_actions.size),
        dtype=np.float64,
    )
    strategy_counts_sum = np.zeros((snapshot_steps.size, 3), dtype=np.float64)

    for run_idx in range(int(params.runs)):
        seed = int(params.seed_start) + run_idx
        print(f"running Fig.4-style distribution run {run_idx + 1}/{params.runs}, seed={seed}...")
        start = time.perf_counter()

        counts_cp, punishment_counts_p, strategy_counts = simulate_fig4_cp_investment_distribution_numba(
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
            snapshot_steps,
        )
        investment_counts_cp_sum += counts_cp
        punishment_counts_p_sum += punishment_counts_p
        strategy_counts_sum += strategy_counts

        print(f"run {run_idx + 1}/{params.runs} done: {time.perf_counter() - start:.2f}s")

    runs = float(params.runs)
    investment_counts_cp_mean = investment_counts_cp_sum / runs
    punishment_counts_p_mean = punishment_counts_p_sum / runs
    strategy_counts_mean = strategy_counts_sum / runs

    tag = build_tag(params, contribution_actions, punishment_actions, snapshot_steps)
    investment_csv_path = params.output_dir / f"{tag}.csv"
    punishment_csv_path = params.output_dir / f"{tag}_punishment.csv"
    save_distribution_csv(
        investment_csv_path,
        snapshot_steps,
        contribution_actions,
        investment_counts_cp_mean,
        strategy_counts_mean,
    )
    save_punishment_distribution_csv(
        punishment_csv_path,
        snapshot_steps,
        punishment_actions,
        punishment_counts_p_mean,
        strategy_counts_mean,
    )
    figure_paths = save_distribution_figure(
        params.output_dir,
        tag,
        params.dpi,
        snapshot_steps,
        contribution_actions,
        investment_counts_cp_mean,
        "fig4_CP_investment_distribution",
        "Investment $e$",
        INVESTMENT_BAR_COLOR,
    )
    figure_paths += save_distribution_figure(
        params.output_dir,
        tag,
        params.dpi,
        snapshot_steps,
        punishment_actions,
        punishment_counts_p_mean,
        "fig4_P_punishment_distribution",
        "Punishment $a$",
        PUNISHMENT_BAR_COLOR,
    )

    print("Fig.4-style C+P investment distribution summary:")
    for idx, step in enumerate(snapshot_steps):
        d_mean = strategy_counts_mean[idx, D]
        c_mean = strategy_counts_mean[idx, C]
        p_mean = strategy_counts_mean[idx, P]
        cp_mean = c_mean + p_mean
        print(
            f"  T={int(step)} | mean D/C/P/C+P = "
            f"{d_mean:.2f}/{c_mean:.2f}/{p_mean:.2f}/{cp_mean:.2f}"
        )

    return [investment_csv_path, punishment_csv_path], figure_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Fig.4-style investment-e distributions for Model E. "
            "Cooperators C and punishers P are merged and counted together as C+P."
        )
    )
    parser.add_argument("--L", type=int, default=CONFIG.L)
    parser.add_argument("--T", type=int, default=CONFIG.T)
    parser.add_argument("--r", type=float, default=CONFIG.r)
    parser.add_argument("--seed", "--seed-start", dest="seed_start", type=int, default=CONFIG.seed_start)
    parser.add_argument("--runs", type=int, default=CONFIG.runs)
    parser.add_argument("--beta-F", type=float, default=CONFIG.beta_F)
    parser.add_argument("--K", type=float, default=CONFIG.K)
    parser.add_argument("--eta-c", "--alpha-c", dest="alpha_c", type=float, default=CONFIG.alpha_c)
    parser.add_argument("--gamma-c", type=float, default=CONFIG.gamma_c)
    parser.add_argument("--eta-a", "--alpha-a", dest="alpha_a", type=float, default=CONFIG.alpha_a)
    parser.add_argument("--gamma-a", type=float, default=CONFIG.gamma_a)
    parser.add_argument("--epsilon-c0", type=float, default=CONFIG.epsilon_c0)
    parser.add_argument("--epsilon-c-min", type=float, default=CONFIG.epsilon_c_min)
    parser.add_argument("--epsilon-c-dcy", type=float, default=CONFIG.epsilon_c_dcy)
    parser.add_argument("--epsilon-a0", type=float, default=CONFIG.epsilon_a0)
    parser.add_argument("--epsilon-a-min", type=float, default=CONFIG.epsilon_a_min)
    parser.add_argument("--epsilon-a-dcy", type=float, default=CONFIG.epsilon_a_dcy)
    parser.add_argument("--init-D", type=float, default=CONFIG.init_D)
    parser.add_argument("--init-C", type=float, default=CONFIG.init_C)
    parser.add_argument("--init-P", type=float, default=CONFIG.init_P)
    parser.add_argument("--contribution-actions", type=str, default=CONFIG.contribution_actions)
    parser.add_argument("--punishment-actions", type=str, default=CONFIG.punishment_actions)
    parser.add_argument(
        "--snapshot-steps",
        type=str,
        default="10,100,1000,10000",
        help="Comma-separated T values to plot, e.g. 10,100,1000,10000.",
    )
    parser.add_argument("--output-dir", type=Path, default=CONFIG.output_dir)
    parser.add_argument("--dpi", type=int, default=CONFIG.dpi)
    parser.add_argument("--no-warmup", action="store_true")
    return parser


def params_from_args(args: argparse.Namespace) -> EvolutionParams:
    return replace(
        CONFIG,
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
    args = build_parser().parse_args()
    snapshot_steps = parse_int_list(args.snapshot_steps)
    csv_paths, figure_paths = run_distribution(params_from_args(args), snapshot_steps)
    for path in csv_paths:
        print("distribution data:", path)
    for path in figure_paths:
        print("distribution plot:", path)


if __name__ == "__main__":
    main()
