from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from numba import njit
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing numba. Please install it first: pip install numba") from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

# ----------------------------
# Inlined Model C helpers from scripts/scan_r/C.py
# ----------------------------
D, C, P = 0, 1, 2

GROUP_SIZE = 5

NUM_STATES = 5

NUM_PUNISHER_STATES = 5

PUNISHMENT_ACTIONS = np.array([0.1,0.2, 0.3, 0.4,0.5], dtype=np.float64)

@njit(cache=True, inline="always")
def prev_index(x: int, L: int) -> int:
    return L - 1 if x == 0 else x - 1

@njit(cache=True, inline="always")
def next_index(x: int, L: int) -> int:
    return 0 if x == L - 1 else x + 1

@njit(cache=True, inline="always")
def is_close_numba(a: float, b: float) -> bool:
    return abs(a - b) <= 1.0e-08 + 1.0e-05 * abs(b)

@njit(cache=True, inline="always")
def random_argmax_punishment(
    q_table: np.ndarray,
    i: int,
    j: int,
    s_d: int,
    s_p: int,
    n_actions: int,
) -> int:
    max_value = q_table[i, j, s_d, s_p, 0]
    for a in range(1, n_actions):
        value = q_table[i, j, s_d, s_p, a]
        if value > max_value:
            max_value = value

    count = 0
    for a in range(n_actions):
        if is_close_numba(q_table[i, j, s_d, s_p, a], max_value):
            count += 1

    chosen_rank = np.random.randint(0, count)
    seen = 0
    for a in range(n_actions):
        if is_close_numba(q_table[i, j, s_d, s_p, a], max_value):
            if seen == chosen_rank:
                return a
            seen += 1
    return 0

@njit(cache=True)
def initialize_strategy(L: int, init_D: float, init_C: float, init_P: float) -> tuple[np.ndarray, np.ndarray]:
    z = np.empty((L, L), dtype=np.int8)
    z_next = np.empty((L, L), dtype=np.int8)
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
            z_next[i, j] = z[i, j]
    return z, z_next

@njit(cache=True)
def compute_neighbor_states(z: np.ndarray, state_d: np.ndarray, state_p: np.ndarray) -> None:
    L = z.shape[0]
    for i in range(L):
        im = prev_index(i, L)
        ip = next_index(i, L)
        for j in range(L):
            jm = prev_index(j, L)
            jp = next_index(j, L)
            count_d = 0
            count_p = 0

            strategy = z[im, j]
            if strategy == D:
                count_d += 1
            elif strategy == P:
                count_p += 1
            strategy = z[ip, j]
            if strategy == D:
                count_d += 1
            elif strategy == P:
                count_p += 1
            strategy = z[i, jm]
            if strategy == D:
                count_d += 1
            elif strategy == P:
                count_p += 1
            strategy = z[i, jp]
            if strategy == D:
                count_d += 1
            elif strategy == P:
                count_p += 1

            state_d[i, j] = count_d
            state_p[i, j] = count_p

@njit(cache=True)
def choose_actions(
    z: np.ndarray,
    state_d: np.ndarray,
    state_p: np.ndarray,
    q_a: np.ndarray,
    punishment_idx: np.ndarray,
    contribution_value: np.ndarray,
    punishment_value: np.ndarray,
    epsilon_a: float,
    fixed_contribution: float,
    punishment_actions: np.ndarray,
) -> None:
    L = z.shape[0]
    n_actions = punishment_actions.shape[0]
    for i in range(L):
        for j in range(L):
            strategy = z[i, j]
            punishment_idx[i, j] = 0
            contribution_value[i, j] = 0.0
            punishment_value[i, j] = 0.0
            if strategy == C or strategy == P:
                contribution_value[i, j] = fixed_contribution
            if strategy == P:
                greedy = random_argmax_punishment(q_a, i, j, state_d[i, j], state_p[i, j], n_actions)
                chosen = np.random.randint(0, n_actions) if np.random.random() < epsilon_a else greedy
                punishment_idx[i, j] = chosen
                punishment_value[i, j] = punishment_actions[chosen]

@njit(cache=True, inline="always")
def add_member_stats(
    z: np.ndarray,
    contribution_value: np.ndarray,
    punishment_value: np.ndarray,
    x: int,
    y: int,
) -> tuple[float, int, int, float]:
    strategy = z[x, y]
    if strategy == D:
        return 0.0, 0, 1, 0.0
    if strategy == P:
        return contribution_value[x, y], 1, 0, punishment_value[x, y]
    return contribution_value[x, y], 0, 0, 0.0

@njit(cache=True, inline="always")
def add_payoff_to_member(
    payoff: np.ndarray,
    z: np.ndarray,
    contribution_value: np.ndarray,
    punishment_value: np.ndarray,
    x: int,
    y: int,
    public_return: float,
    fine: float,
    n_d_g: int,
) -> None:
    strategy = z[x, y]
    if strategy == D:
        payoff[x, y] += public_return - fine
    elif strategy == C:
        payoff[x, y] += public_return - contribution_value[x, y]
    else:
        payoff[x, y] += public_return - contribution_value[x, y] - punishment_value[x, y] * n_d_g

@njit(cache=True, inline="always")
def add_group_payoff(
    z: np.ndarray,
    contribution_value: np.ndarray,
    punishment_value: np.ndarray,
    payoff: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    x3: int,
    y3: int,
    x4: int,
    y4: int,
    r_value: float,
    beta_F: float,
) -> None:
    total_contribution = 0.0
    n_p = 0
    n_d = 0
    sum_punishment = 0.0

    tc, npg, ndg, sp = add_member_stats(z, contribution_value, punishment_value, x0, y0)
    total_contribution += tc
    n_p += npg
    n_d += ndg
    sum_punishment += sp
    tc, npg, ndg, sp = add_member_stats(z, contribution_value, punishment_value, x1, y1)
    total_contribution += tc
    n_p += npg
    n_d += ndg
    sum_punishment += sp
    tc, npg, ndg, sp = add_member_stats(z, contribution_value, punishment_value, x2, y2)
    total_contribution += tc
    n_p += npg
    n_d += ndg
    sum_punishment += sp
    tc, npg, ndg, sp = add_member_stats(z, contribution_value, punishment_value, x3, y3)
    total_contribution += tc
    n_p += npg
    n_d += ndg
    sum_punishment += sp
    tc, npg, ndg, sp = add_member_stats(z, contribution_value, punishment_value, x4, y4)
    total_contribution += tc
    n_p += npg
    n_d += ndg
    sum_punishment += sp

    fine = 0.0
    if n_p > 0:
        fine = n_p * (np.exp(beta_F * (sum_punishment / n_p)) - 1.0)
    public_return = r_value * total_contribution / GROUP_SIZE

    add_payoff_to_member(payoff, z, contribution_value, punishment_value, x0, y0, public_return, fine, n_d)
    add_payoff_to_member(payoff, z, contribution_value, punishment_value, x1, y1, public_return, fine, n_d)
    add_payoff_to_member(payoff, z, contribution_value, punishment_value, x2, y2, public_return, fine, n_d)
    add_payoff_to_member(payoff, z, contribution_value, punishment_value, x3, y3, public_return, fine, n_d)
    add_payoff_to_member(payoff, z, contribution_value, punishment_value, x4, y4, public_return, fine, n_d)

@njit(cache=True)
def compute_payoff(
    z: np.ndarray,
    contribution_value: np.ndarray,
    punishment_value: np.ndarray,
    payoff: np.ndarray,
    r_value: float,
    beta_F: float,
) -> None:
    L = z.shape[0]
    for i in range(L):
        for j in range(L):
            payoff[i, j] = 0.0

    for i in range(L):
        im = prev_index(i, L)
        ip = next_index(i, L)
        for j in range(L):
            jm = prev_index(j, L)
            jp = next_index(j, L)
            add_group_payoff(z, contribution_value, punishment_value, payoff, i, j, im, j, ip, j, i, jm, i, jp, r_value, beta_F)

@njit(cache=True)
def strategy_update(z: np.ndarray, payoff: np.ndarray, z_next: np.ndarray, K: float) -> None:
    L = z.shape[0]
    for i in range(L):
        im = prev_index(i, L)
        ip = next_index(i, L)
        for j in range(L):
            jm = prev_index(j, L)
            jp = next_index(j, L)
            direction = np.random.randint(0, 4)
            if direction == 0:
                ni = im
                nj = j
            elif direction == 1:
                ni = ip
                nj = j
            elif direction == 2:
                ni = i
                nj = jm
            else:
                ni = i
                nj = jp
            scaled = (payoff[ni, nj] - payoff[i, j]) / K
            if scaled > 60.0:
                scaled = 60.0
            elif scaled < -60.0:
                scaled = -60.0
            imitate_prob = 1.0 / (1.0 + np.exp(-scaled))
            z_next[i, j] = z[ni, nj] if np.random.random() < imitate_prob else z[i, j]

@njit(cache=True, inline="always")
def max_q_at_punishment_state(
    q_table: np.ndarray,
    i: int,
    j: int,
    s_d: int,
    s_p: int,
    n_actions: int,
) -> float:
    max_value = q_table[i, j, s_d, s_p, 0]
    for a in range(1, n_actions):
        value = q_table[i, j, s_d, s_p, a]
        if value > max_value:
            max_value = value
    return max_value

@njit(cache=True, inline="always")
def neighbor_payoff_average(payoff: np.ndarray, i: int, j: int) -> float:
    L = payoff.shape[0]
    im = prev_index(i, L)
    ip = next_index(i, L)
    jm = prev_index(j, L)
    jp = next_index(j, L)
    return 0.25 * (payoff[im, j] + payoff[ip, j] + payoff[i, jm] + payoff[i, jp])

@njit(cache=True)
def update_punishment_q(
    z_old: np.ndarray,
    payoff: np.ndarray,
    state_d_old: np.ndarray,
    state_p_old: np.ndarray,
    state_d_next: np.ndarray,
    state_p_next: np.ndarray,
    q_a: np.ndarray,
    punishment_idx: np.ndarray,
    alpha_a: float,
    gamma_a: float,
    w_a: float,
    punishment_actions: np.ndarray,
) -> None:
    L = z_old.shape[0]
    n_actions = punishment_actions.shape[0]
    neighbor_count = GROUP_SIZE - 1.0
    a_min = punishment_actions[0]
    a_max = punishment_actions[0]
    for a in range(1, n_actions):
        if punishment_actions[a] < a_min:
            a_min = punishment_actions[a]
        if punishment_actions[a] > a_max:
            a_max = punishment_actions[a]
    a_range = a_max - a_min

    for i in range(L):
        for j in range(L):
            if z_old[i, j] == P:
                s_d0 = state_d_old[i, j]
                s_p0 = state_p_old[i, j]
                s_d1 = state_d_next[i, j]
                s_p1 = state_p_next[i, j]
                action = punishment_idx[i, j]
                old_q = q_a[i, j, s_d0, s_p0, action]
                next_max = max_q_at_punishment_state(q_a, i, j, s_d1, s_p1, n_actions)

                avg_neighbor_payoff = neighbor_payoff_average(payoff, i, j)
                if s_d0 == 0 or payoff[i, j] >= avg_neighbor_payoff:
                    a_target = a_min
                else:
                    d_local = s_d0 / neighbor_count
                    a_target = a_min + a_range * d_local

                a_value = punishment_actions[action]
                # double_Q.tex defines the immediate punishment reward as R_a = M_a.
                reward = 1.0 - abs(a_value - a_target) / a_range
                q_a[i, j, s_d0, s_p0, action] = old_q + alpha_a * (reward + gamma_a * next_max - old_q)

@njit(cache=True)
def copy_strategy(z_next: np.ndarray, z: np.ndarray) -> None:
    L = z.shape[0]
    for i in range(L):
        for j in range(L):
            z[i, j] = z_next[i, j]

@njit(cache=True)
def count_strategies(z: np.ndarray) -> tuple[int, int, int]:
    L = z.shape[0]
    count_d = 0
    count_c = 0
    count_p = 0
    for i in range(L):
        for j in range(L):
            if z[i, j] == D:
                count_d += 1
            elif z[i, j] == C:
                count_c += 1
            else:
                count_p += 1
    return count_d, count_c, count_p

MODEL_NAME = "C"


@dataclass(frozen=True)
class ScanParams:
    L: int = 100
    T: int = 30000
    beta_F: float = 1.5
    K: float = 0.5
    fixed_contribution: float = 1.0
    alpha_a: float = 0.8
    gamma_a: float = 0.8
    w_a: float = 0.0
    epsilon_a0: float = 0.1
    epsilon_a_min: float = 0.01
    epsilon_a_dcy: float = 0.99
    init_D: float = 1.0 / 3.0
    init_C: float = 1.0 / 3.0
    init_P: float = 1.0 / 3.0
    r_start: float = 1.0
    r_stop: float = 5.0
    r_step: float = 0.1
    r_values: str | None = None
    runs: int = 20
    seed_start: int = 20
    workers: int = 5
    tail_window: int = 100
    output_dir: Path = Path("data/payoff_r/C")
    dpi: int = 300
    progress: bool = True
    warmup: bool = True


CONFIG = ScanParams()


def parse_r_values(params: ScanParams) -> list[float]:
    if params.r_values:
        values = [float(item.strip()) for item in params.r_values.split(",") if item.strip()]
        if not values:
            raise ValueError("r_values must contain at least one value.")
        return values
    if params.r_step <= 0:
        raise ValueError("r_step must be greater than 0.")
    if params.r_stop < params.r_start:
        raise ValueError("r_stop must be greater than or equal to r_start.")
    count = int(np.floor((params.r_stop - params.r_start) / params.r_step + 1.0e-9)) + 1
    return [round(params.r_start + i * params.r_step, 10) for i in range(count)]


def validate_params(params: ScanParams) -> tuple[float, float, float, int]:
    if params.L <= 0 or params.T <= 0:
        raise ValueError("L and T must be positive integers.")
    if params.K <= 0:
        raise ValueError("K must be greater than 0.")
    if params.beta_F < 0:
        raise ValueError("beta_F must be nonnegative.")
    if params.fixed_contribution < 0:
        raise ValueError("fixed_contribution must be nonnegative.")
    if not (0.0 < params.alpha_a <= 1.0):
        raise ValueError("alpha_a must be in (0, 1].")
    if not (0.0 <= params.gamma_a < 1.0):
        raise ValueError("gamma_a must be in [0, 1).")
    if not (0.0 <= params.w_a <= 1.0):
        raise ValueError("w_a must be in [0, 1].")
    if not (0.0 <= params.epsilon_a0 <= 1.0 and 0.0 <= params.epsilon_a_min <= 1.0):
        raise ValueError("Invalid epsilon values.")
    if params.epsilon_a0 < params.epsilon_a_min:
        raise ValueError("Initial epsilon must be >= minimum epsilon.")
    if not (0.0 < params.epsilon_a_dcy < 1.0):
        raise ValueError("epsilon_a_dcy must be in (0, 1).")
    if params.runs <= 0 or params.workers <= 0:
        raise ValueError("runs and workers must be positive integers.")
    if params.tail_window < 0:
        raise ValueError("tail_window must be nonnegative.")

    probs = np.array([params.init_D, params.init_C, params.init_P], dtype=np.float64)
    if np.any(probs < 0.0) or probs.sum() <= 0.0:
        raise ValueError("Initial strategy fractions must be nonnegative and have positive sum.")
    probs = probs / probs.sum()
    tail_window = min(max(params.tail_window, 0), params.T)
    return float(probs[0]), float(probs[1]), float(probs[2]), int(tail_window)


def validate_r_values(r_values: list[float]) -> None:
    for r_value in r_values:
        if r_value <= 0:
            raise ValueError("All r values must be greater than 0.")


@njit(cache=True)
def average_payoff(payoff: np.ndarray) -> float:
    total = 0.0
    L = payoff.shape[0]
    for i in range(L):
        for j in range(L):
            total += payoff[i, j]
    return total / (L * L)


@njit(cache=True)
def simulate_model_c_payoff(
    L: int,
    T: int,
    r_value: float,
    beta_F: float,
    K: float,
    fixed_contribution: float,
    alpha_a: float,
    gamma_a: float,
    w_a: float,
    epsilon_a0: float,
    epsilon_a_min: float,
    epsilon_a_dcy: float,
    init_D: float,
    init_C: float,
    init_P: float,
    tail_window: int,
    seed: int,
) -> float:
    np.random.seed(seed)
    z, z_next = initialize_strategy(L, init_D, init_C, init_P)
    q_a = np.zeros((L, L, NUM_STATES, NUM_PUNISHER_STATES, PUNISHMENT_ACTIONS.shape[0]), dtype=np.float64)
    state_d_old = np.empty((L, L), dtype=np.int64)
    state_p_old = np.empty((L, L), dtype=np.int64)
    state_d_next = np.empty((L, L), dtype=np.int64)
    state_p_next = np.empty((L, L), dtype=np.int64)
    punishment_idx = np.zeros((L, L), dtype=np.int64)
    contribution_value = np.zeros((L, L), dtype=np.float64)
    punishment_value = np.zeros((L, L), dtype=np.float64)
    payoff = np.zeros((L, L), dtype=np.float64)
    epsilon_a = epsilon_a0
    tail_payoff = 0.0
    tail_count = 0

    for step in range(1, T + 1):
        compute_neighbor_states(z, state_d_old, state_p_old)
        choose_actions(
            z,
            state_d_old,
            state_p_old,
            q_a,
            punishment_idx,
            contribution_value,
            punishment_value,
            epsilon_a,
            fixed_contribution,
            PUNISHMENT_ACTIONS,
        )
        compute_payoff(z, contribution_value, punishment_value, payoff, r_value, beta_F)
        if tail_window > 0 and step > T - tail_window:
            tail_payoff += average_payoff(payoff)
            tail_count += 1
        strategy_update(z, payoff, z_next, K)
        compute_neighbor_states(z_next, state_d_next, state_p_next)
        update_punishment_q(
            z,
            payoff,
            state_d_old,
            state_p_old,
            state_d_next,
            state_p_next,
            q_a,
            punishment_idx,
            alpha_a,
            gamma_a,
            w_a,
            PUNISHMENT_ACTIONS,
        )
        epsilon_a = epsilon_a * epsilon_a_dcy
        if epsilon_a < epsilon_a_min:
            epsilon_a = epsilon_a_min
        copy_strategy(z_next, z)

    if tail_count > 0:
        return tail_payoff / tail_count

    compute_neighbor_states(z, state_d_old, state_p_old)
    choose_actions(z, state_d_old, state_p_old, q_a, punishment_idx, contribution_value, punishment_value, epsilon_a, fixed_contribution, PUNISHMENT_ACTIONS)
    compute_payoff(z, contribution_value, punishment_value, payoff, r_value, beta_F)
    return average_payoff(payoff)


def warmup_numba() -> None:
    simulate_model_c_payoff(4, 2, 2.5, 1.0, 0.5, 1.0, 0.8, 0.8, 0.0, 0.1, 0.01, 0.99, 1 / 3, 1 / 3, 1 / 3, 1, 123)


def simulate_one(params: ScanParams, r_value: float, seed: int) -> dict[str, float | int | str]:
    init_D, init_C, init_P, tail_window = validate_params(params)
    start = time.perf_counter()
    system_payoff = simulate_model_c_payoff(
        int(params.L),
        int(params.T),
        float(r_value),
        float(params.beta_F),
        float(params.K),
        float(params.fixed_contribution),
        float(params.alpha_a),
        float(params.gamma_a),
        float(params.w_a),
        float(params.epsilon_a0),
        float(params.epsilon_a_min),
        float(params.epsilon_a_dcy),
        init_D,
        init_C,
        init_P,
        tail_window,
        int(seed),
    )
    return {
        "model": MODEL_NAME,
        "w_a": float(params.w_a),
        "r": float(r_value),
        "seed": int(seed),
        "system_payoff": float(system_payoff),
        "seconds": time.perf_counter() - start,
    }


def summarize_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    summaries = []
    for r_value in sorted({float(row["r"]) for row in rows}):
        values = np.array([float(row["system_payoff"]) for row in rows if float(row["r"]) == r_value], dtype=np.float64)
        summaries.append(
            {
                "model": MODEL_NAME,
                "w_a": float(rows[0]["w_a"]),
                "r": r_value,
                "runs": int(values.size),
                "system_payoff_mean": float(np.mean(values)),
                "system_payoff_std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_one_figure(fig: plt.Figure, basename: Path, dpi: int) -> list[Path]:
    outputs = []
    basename.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".pdf", ".svg", ".png", ".tiff"]:
        path = basename.with_suffix(suffix)
        fig.savefig(path, dpi=dpi, facecolor="white")
        outputs.append(path)
    return outputs


def plot_summary(params: ScanParams, summaries: list[dict[str, float | int | str]]) -> list[Path]:
    summaries = sorted(summaries, key=lambda row: float(row["r"]))
    r_values = np.array([float(row["r"]) for row in summaries], dtype=np.float64)
    means = np.array([float(row["system_payoff_mean"]) for row in summaries], dtype=np.float64)
    stds = np.array([float(row["system_payoff_std"]) for row in summaries], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    color = "#009E73"
    ax.plot(r_values, means, marker="o", markersize=3.0, linewidth=1.8, color=color, label="Model C")
    if params.runs > 1:
        ax.fill_between(r_values, means - stds, means + stds, color=color, alpha=0.16, linewidth=0.0)
    ax.set_xlabel("Synergy factor r")
    ax.set_ylabel("System average payoff")
    ax.set_title("Model C: system payoff vs r")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    return save_one_figure(fig, params.output_dir / "C_system_payoff_vs_r", params.dpi)


def run_scan(params: ScanParams) -> tuple[Path, Path, list[Path]]:
    r_values = parse_r_values(params)
    validate_r_values(r_values)
    validate_params(params)
    params.output_dir.mkdir(parents=True, exist_ok=True)

    if params.warmup:
        print("compiling numba functions...")
        start = time.perf_counter()
        warmup_numba()
        print(f"numba warmup done: {time.perf_counter() - start:.2f}s")

    tasks = [(r_value, params.seed_start + run) for r_value in r_values for run in range(params.runs)]
    rows = []
    start = time.perf_counter()
    if params.workers == 1:
        iterator = tasks
        if params.progress and tqdm is not None:
            iterator = tqdm(tasks, desc="C payoff r scan", unit="run", dynamic_ncols=True)
        for r_value, seed in iterator:
            rows.append(simulate_one(params, r_value, seed))
    else:
        with ProcessPoolExecutor(max_workers=params.workers) as executor:
            futures = [executor.submit(simulate_one, params, r_value, seed) for r_value, seed in tasks]
            iterator = as_completed(futures)
            if params.progress and tqdm is not None:
                iterator = tqdm(iterator, total=len(futures), desc="C payoff r scan", unit="run", dynamic_ncols=True)
            for future in iterator:
                rows.append(future.result())
    print(f"payoff r scan done: {time.perf_counter() - start:.2f}s")

    rows.sort(key=lambda row: (float(row["r"]), int(row["seed"])))
    summaries = summarize_rows(rows)
    raw_csv = params.output_dir / "C_payoff_r_runs.csv"
    summary_csv = params.output_dir / "C_payoff_r_summary.csv"
    write_csv(raw_csv, rows)
    write_csv(summary_csv, summaries)
    figure_paths = plot_summary(params, summaries)
    plt.close("all")
    return raw_csv, summary_csv, figure_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan Model C system average payoff versus r.")
    parser.add_argument("--L", type=int, default=CONFIG.L)
    parser.add_argument("--T", type=int, default=CONFIG.T)
    parser.add_argument("--beta-F", type=float, default=CONFIG.beta_F)
    parser.add_argument("--K", type=float, default=CONFIG.K)
    parser.add_argument("--fixed-contribution", type=float, default=CONFIG.fixed_contribution)
    parser.add_argument("--alpha-a", type=float, default=CONFIG.alpha_a)
    parser.add_argument("--gamma-a", type=float, default=CONFIG.gamma_a)
    parser.add_argument("--w-a", type=float, default=CONFIG.w_a)
    parser.add_argument("--epsilon-a0", type=float, default=CONFIG.epsilon_a0)
    parser.add_argument("--epsilon-a-min", type=float, default=CONFIG.epsilon_a_min)
    parser.add_argument("--epsilon-a-dcy", type=float, default=CONFIG.epsilon_a_dcy)
    parser.add_argument("--init-D", type=float, default=CONFIG.init_D)
    parser.add_argument("--init-C", type=float, default=CONFIG.init_C)
    parser.add_argument("--init-P", type=float, default=CONFIG.init_P)
    parser.add_argument("--r-start", type=float, default=CONFIG.r_start)
    parser.add_argument("--r-stop", type=float, default=CONFIG.r_stop)
    parser.add_argument("--r-step", type=float, default=CONFIG.r_step)
    parser.add_argument("--r-values", type=str, default=CONFIG.r_values)
    parser.add_argument("--runs", type=int, default=CONFIG.runs)
    parser.add_argument("--seed-start", type=int, default=CONFIG.seed_start)
    parser.add_argument("--workers", type=int, default=CONFIG.workers)
    parser.add_argument("--tail-window", type=int, default=CONFIG.tail_window)
    parser.add_argument("--output-dir", type=Path, default=CONFIG.output_dir)
    parser.add_argument("--dpi", type=int, default=CONFIG.dpi)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    return parser


def params_from_args(args: argparse.Namespace) -> ScanParams:
    return replace(
        CONFIG,
        L=args.L,
        T=args.T,
        beta_F=args.beta_F,
        K=args.K,
        fixed_contribution=args.fixed_contribution,
        alpha_a=args.alpha_a,
        gamma_a=args.gamma_a,
        w_a=args.w_a,
        epsilon_a0=args.epsilon_a0,
        epsilon_a_min=args.epsilon_a_min,
        epsilon_a_dcy=args.epsilon_a_dcy,
        init_D=args.init_D,
        init_C=args.init_C,
        init_P=args.init_P,
        r_start=args.r_start,
        r_stop=args.r_stop,
        r_step=args.r_step,
        r_values=args.r_values,
        runs=args.runs,
        seed_start=args.seed_start,
        workers=args.workers,
        tail_window=args.tail_window,
        output_dir=args.output_dir,
        dpi=args.dpi,
        progress=not args.no_progress,
        warmup=not args.no_warmup,
    )


def main() -> None:
    raw_csv, summary_csv, figure_paths = run_scan(params_from_args(build_parser().parse_args()))
    print(f"raw payoff results: {raw_csv}")
    print(f"summary payoff results: {summary_csv}")
    for path in figure_paths:
        print(f"payoff plot: {path}")


if __name__ == "__main__":
    main()
