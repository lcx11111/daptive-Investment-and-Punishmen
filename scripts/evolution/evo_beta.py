from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evo import (
    CONFIG as BASE_CONFIG,
    EvolutionParams,
    configure_publication_style,
    format_value,
    parse_float_list,
    save_one_figure,
    simulate_evolution_numba,
    summarize_runs,
    validate_params,
    warmup_numba,
)


# ============================================================
# Model E evolution curves under different beta_F values.
#
# This script reuses the simulation kernel in evo.py, runs the
# same seed range for each beta_F, averages repeated runs, and saves:
#   1. one long-format CSV containing all beta_F trajectories;
#   2. an overlay figure for rho_C + rho_P;
#   3. a three-panel overlay figure for rho_D, rho_C, rho_P.
#
# Example:
#   python evo_beta.py --betas 1.0,1.25,1.5,1.75,2.0
# ============================================================


DEFAULT_BETAS = "1.0,1.5,2.0"
CONFIG = replace(BASE_CONFIG, output_dir=Path("data/evolution_beta"))


def beta_label(beta_value: float) -> str:
    return rf"$\beta_F={beta_value:g}$"


def beta_tag(beta_values: np.ndarray) -> str:
    return "-".join(format_value(float(value)) for value in beta_values)


def action_tag(actions: np.ndarray) -> str:
    return f"{actions.size}x{format_value(float(actions.min()))}-{format_value(float(actions.max()))}"


def build_tag(
    params: EvolutionParams,
    beta_values: np.ndarray,
    contribution_actions: np.ndarray,
    punishment_actions: np.ndarray,
) -> str:
    return (
        f"modelE_evolution_logt_betaF_L{params.L}_T{params.T}_r{format_value(params.r)}"
        f"_runs{params.runs}_seed{params.seed_start}"
        f"_betaF{beta_tag(beta_values)}_K{format_value(params.K)}"
        f"_aC{format_value(params.alpha_c)}_gC{format_value(params.gamma_c)}"
        f"_aA{format_value(params.alpha_a)}_gA{format_value(params.gamma_a)}"
        f"_eC{format_value(params.epsilon_c0)}-{format_value(params.epsilon_c_min)}-{format_value(params.epsilon_c_dcy)}"
        f"_eP{format_value(params.epsilon_a0)}-{format_value(params.epsilon_a_min)}-{format_value(params.epsilon_a_dcy)}"
        f"_Ac{action_tag(contribution_actions)}_Aa{action_tag(punishment_actions)}"
    )


def validate_beta_values(beta_values: np.ndarray) -> np.ndarray:
    if beta_values.size == 0 or np.any(beta_values <= 0.0):
        raise ValueError("beta values must be positive.")
    return beta_values.astype(np.float64)


def simulate_for_beta(
    params: EvolutionParams,
    beta_value: float,
    init_D: float,
    init_C: float,
    init_P: float,
    contribution_actions: np.ndarray,
    punishment_actions: np.ndarray,
) -> np.ndarray:
    beta_params = replace(params, beta_F=float(beta_value))
    validate_params(beta_params)
    return simulate_evolution_numba(
        int(beta_params.L),
        int(beta_params.T),
        float(beta_params.r),
        float(beta_params.beta_F),
        float(beta_params.K),
        float(beta_params.alpha_c),
        float(beta_params.gamma_c),
        float(beta_params.alpha_a),
        float(beta_params.gamma_a),
        float(beta_params.epsilon_c0),
        float(beta_params.epsilon_c_min),
        float(beta_params.epsilon_c_dcy),
        float(beta_params.epsilon_a0),
        float(beta_params.epsilon_a_min),
        float(beta_params.epsilon_a_dcy),
        init_D,
        init_C,
        init_P,
        int(beta_params.seed),
        contribution_actions,
        punishment_actions,
    )


def save_beta_csv(
    csv_path: Path,
    beta_values: np.ndarray,
    trajectories: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]],
    T: int,
) -> None:
    t = np.arange(T + 1, dtype=np.int64)
    t_plus_1 = t + 1
    rows = []
    for beta_value in beta_values:
        densities_mean, densities_std, densities_sem = trajectories[float(beta_value)]
        beta_column = np.full(T + 1, float(beta_value), dtype=np.float64)
        rows.append(np.column_stack((beta_column, t, t_plus_1, densities_mean, densities_std, densities_sem)))

    header = (
        "beta_F,t,t_plus_1,"
        "mean_rho_D,mean_rho_C,mean_rho_P,mean_rho_C_plus_P,"
        "std_rho_D,std_rho_C,std_rho_P,std_rho_C_plus_P,"
        "sem_rho_D,sem_rho_C,sem_rho_P,sem_rho_C_plus_P"
    )
    data = np.vstack(rows)
    np.savetxt(
        csv_path,
        data,
        delimiter=",",
        header=header,
        comments="",
        fmt=["%.10f", "%d", "%d"] + ["%.10f"] * 12,
    )


def format_log_axis(ax: plt.Axes, params: EvolutionParams, ylabel: str) -> None:
    ax.set_xscale("log")
    ax.set_xlim(1, params.T + 1)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"$t+1$")
    ax.set_ylabel(ylabel)


def legend_columns(count: int) -> int:
    if count <= 4:
        return 1
    if count <= 8:
        return 2
    return 3


def save_beta_figures(
    output_dir: Path,
    tag: str,
    params: EvolutionParams,
    beta_values: np.ndarray,
    trajectories: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> list[Path]:
    t_plus_1 = np.arange(params.T + 1, dtype=np.int64) + 1
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, beta_values.size))
    figure_paths: list[Path] = []

    fig_total, ax_total = plt.subplots(figsize=(4.25, 3.05), constrained_layout=True)
    for color, beta_value in zip(colors, beta_values):
        densities, _, _ = trajectories[float(beta_value)]
        ax_total.plot(
            t_plus_1,
            densities[:, 3],
            color=color,
            linewidth=1.65,
            label=beta_label(float(beta_value)),
        )
    format_log_axis(ax_total, params, r"$\rho_C+\rho_P$")
    ax_total.set_title(rf"Model E, $r={params.r:g}$", fontsize=11, pad=4)
    ax_total.legend(loc="best", handlelength=2.2, ncol=legend_columns(beta_values.size))
    figure_paths.extend(save_one_figure(fig_total, output_dir / f"{tag}_C_plus_P_by_beta", params.dpi))
    plt.close(fig_total)

    fig_three, axes = plt.subplots(3, 1, figsize=(4.35, 5.45), sharex=True, constrained_layout=True)
    strategy_columns = (
        (0, r"$\rho_D$"),
        (1, r"$\rho_C$"),
        (2, r"$\rho_P$"),
    )
    for ax, (column, ylabel) in zip(axes, strategy_columns):
        for color, beta_value in zip(colors, beta_values):
            densities, _, _ = trajectories[float(beta_value)]
            ax.plot(
                t_plus_1,
                densities[:, column],
                color=color,
                linewidth=1.45,
                label=beta_label(float(beta_value)),
            )
        format_log_axis(ax, params, ylabel)
        ax.set_xlabel("")
    axes[-1].set_xlabel(r"$t+1$")
    axes[0].set_title(rf"Model E, $r={params.r:g}$", fontsize=11, pad=4)
    axes[0].legend(loc="best", handlelength=2.2, ncol=legend_columns(beta_values.size))
    figure_paths.extend(save_one_figure(fig_three, output_dir / f"{tag}_three_strategies_by_beta", params.dpi))
    plt.close(fig_three)

    return figure_paths


def run_beta_evolution(params: EvolutionParams, beta_values: np.ndarray) -> tuple[Path, list[Path]]:
    configure_publication_style()
    beta_values = validate_beta_values(beta_values)
    init_D, init_C, init_P = validate_params(params)
    contribution_actions = parse_float_list(params.contribution_actions)
    punishment_actions = parse_float_list(params.punishment_actions)
    params.output_dir.mkdir(parents=True, exist_ok=True)

    if params.warmup:
        print("compiling numba functions...")
        start = time.perf_counter()
        warmup_numba()
        print(f"numba warmup done: {time.perf_counter() - start:.2f}s")

    trajectories: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for beta_value in beta_values:
        print(f"running beta_F={beta_value:g}...")
        start = time.perf_counter()
        curves = np.empty((int(params.runs), int(params.T) + 1, 4), dtype=np.float64)
        for run_idx in range(int(params.runs)):
            seed = int(params.seed_start) + run_idx
            print(f"  beta_F={beta_value:g}, run {run_idx + 1}/{params.runs}, seed={seed}...")
            curves[run_idx] = simulate_for_beta(
                replace(params, seed=seed, seed_start=seed),
                float(beta_value),
                init_D,
                init_C,
                init_P,
                contribution_actions,
                punishment_actions,
            )
        trajectories[float(beta_value)] = summarize_runs(curves)
        print(f"beta_F={beta_value:g} done: {time.perf_counter() - start:.2f}s")

    tag = build_tag(params, beta_values, contribution_actions, punishment_actions)
    csv_path = params.output_dir / f"{tag}.csv"
    save_beta_csv(csv_path, beta_values, trajectories, int(params.T))
    figure_paths = save_beta_figures(params.output_dir, tag, params, beta_values, trajectories)
    return csv_path, figure_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Save Model E time-evolution curves for multiple beta_F values.")
    parser.add_argument("--L", type=int, default=CONFIG.L)
    parser.add_argument("--T", type=int, default=CONFIG.T)
    parser.add_argument("--r", type=float, default=CONFIG.r)
    parser.add_argument("--seed", "--seed-start", dest="seed_start", type=int, default=CONFIG.seed_start)
    parser.add_argument("--runs", type=int, default=CONFIG.runs)
    parser.add_argument("--betas", "--beta-values", "--beta-F-values", dest="betas", type=str, default=DEFAULT_BETAS)
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
    params = params_from_args(args)
    beta_values = parse_float_list(args.betas)
    csv_path, figure_paths = run_beta_evolution(params, beta_values)
    print("beta evolution data:", csv_path)
    for path in figure_paths:
        print("beta evolution plot:", path)


if __name__ == "__main__":
    main()
