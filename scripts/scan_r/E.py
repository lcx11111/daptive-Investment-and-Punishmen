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


D, C, P = 0, 1, 2
GROUP_SIZE = 5
NUM_STATES = 5
NUM_PUNISHER_STATES = 5
MODEL_NAME = "E"

CONTRIBUTION_ACTIONS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float64)
PUNISHMENT_ACTIONS = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)


@dataclass(frozen=True)
class ScanParams:
    L: int = 100
    T: int = 10000
    beta_F: float = 1.5
    K: float = 0.5
    η_c: float = 0.8
    gamma_c: float = 0.8
    η_a: float = 0.8
    gamma_a: float = 0.8
    epsilon_c0: float = 0.3
    epsilon_c_min: float = 0.02
    epsilon_c_dcy: float = 0.9
    epsilon_a0: float = 0.3
    epsilon_a_min: float = 0.02
    epsilon_a_dcy: float = 0.9
    init_D: float = 1.0 / 3.0
    init_C: float = 1.0 / 3.0
    init_P: float = 1.0 / 3.0
    r_start: float = 1.0
    r_stop: float = 5.0
    r_step: float = 0.1
    r_values: str | None = None
    runs: int = 20
    seed_start: int = 1449
    workers: int = 5
    tail_window: int = 100
    output_dir: Path = Path("data/scan_r/E")
    dpi: int = 300
    progress: bool = True
    warmup: bool = True


CONFIG = ScanParams()


PAYOFF_METRICS = (
    ("mean_cumulative_net_payoff", "cumulative_net_payoff", "Mean cumulative system net payoff", "cumulative net payoff per agent"),
    ("mean_cumulative_punisher_cost", "cumulative_punisher_cost", "Mean cumulative punisher cost", "cumulative cost per punisher"),
    ("mean_cumulative_defector_fine", "cumulative_defector_fine", "Mean cumulative defector fine", "cumulative fine per defector"),
    ("mean_cumulative_punishment_burden", "cumulative_punishment_burden", "Mean cumulative punishment burden", "punisher cost + defector fine"),
)


def save_payoff_plots(
    output_dir: Path,
    model_name: str,
    summaries: list[dict[str, float | int | str]],
    dpi: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    for metric_key, filename_key, title, ylabel in PAYOFF_METRICS:
        rows = sorted(summaries, key=lambda row: float(row["r"]))
        r_arr = np.array([float(row["r"]) for row in rows])
        y_arr = np.array([float(row[metric_key]) for row in rows])

        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        ax.plot(r_arr, y_arr, marker=markers[0], linewidth=2)
        ax.set_xlabel("r")
        ax.set_ylabel(ylabel)
        ax.set_xlim(float(r_arr.min()) - 0.05, float(r_arr.max()) + 0.05)
        if filename_key != "cumulative_net_payoff":
            ax.set_ylim(bottom=0.0)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        ax.set_title(f"Model {model_name}: {title} vs r")
        fig.tight_layout()

        path = output_dir / f"{model_name}_{filename_key}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)

    return outputs


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
    if params.beta_F <= 0:
        raise ValueError("beta_F must be greater than 0.")
    if not (0.0 < params.η_c <= 1.0):
        raise ValueError("η_c must be in (0, 1].")
    if not (0.0 <= params.gamma_c < 1.0):
        raise ValueError("gamma_c must be in [0, 1).")
    if not (0.0 < params.η_a <= 1.0):
        raise ValueError("η_a must be in (0, 1].")
    if not (0.0 <= params.gamma_a < 1.0):
        raise ValueError("gamma_a must be in [0, 1).")
    if not (0.0 <= params.epsilon_c0 <= 1.0):
        raise ValueError("epsilon_c0 must be in [0, 1].")
    if not (0.0 <= params.epsilon_c_min <= 1.0):
        raise ValueError("epsilon_c_min must be in [0, 1].")
    if params.epsilon_c0 < params.epsilon_c_min:
        raise ValueError("epsilon_c0 must be greater than or equal to epsilon_c_min.")
    if not (0.0 < params.epsilon_c_dcy < 1.0):
        raise ValueError("epsilon_c_dcy must be in (0, 1).")
    if not (0.0 <= params.epsilon_a0 <= 1.0):
        raise ValueError("epsilon_a0 must be in [0, 1].")
    if not (0.0 <= params.epsilon_a_min <= 1.0):
        raise ValueError("epsilon_a_min must be in [0, 1].")
    if params.epsilon_a0 < params.epsilon_a_min:
        raise ValueError("epsilon_a0 must be greater than or equal to epsilon_a_min.")
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
def random_argmax_contribution(q_table: np.ndarray, i: int, j: int, s: int, n_actions: int) -> int:
    max_value = q_table[i, j, s, 0]
    for a in range(1, n_actions):
        value = q_table[i, j, s, a]
        if value > max_value:
            max_value = value

    count = 0
    for a in range(n_actions):
        if is_close_numba(q_table[i, j, s, a], max_value):
            count += 1

    chosen_rank = np.random.randint(0, count)
    seen = 0
    for a in range(n_actions):
        if is_close_numba(q_table[i, j, s, a], max_value):
            if seen == chosen_rank:
                return a
            seen += 1
    return 0


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
def compute_neighbor_states(
    z: np.ndarray,
    state_d: np.ndarray,
    state_p: np.ndarray,
    state_cp: np.ndarray,
) -> None:
    L = z.shape[0]
    for i in range(L):
        im = prev_index(i, L)
        ip = next_index(i, L)
        for j in range(L):
            jm = prev_index(j, L)
            jp = next_index(j, L)
            count_d = 0
            count_p = 0
            count_cp = 0

            strategy = z[im, j]
            if strategy == D:
                count_d += 1
            else:
                count_cp += 1
                if strategy == P:
                    count_p += 1
            strategy = z[ip, j]
            if strategy == D:
                count_d += 1
            else:
                count_cp += 1
                if strategy == P:
                    count_p += 1
            strategy = z[i, jm]
            if strategy == D:
                count_d += 1
            else:
                count_cp += 1
                if strategy == P:
                    count_p += 1
            strategy = z[i, jp]
            if strategy == D:
                count_d += 1
            else:
                count_cp += 1
                if strategy == P:
                    count_p += 1

            state_d[i, j] = count_d
            state_p[i, j] = count_p
            state_cp[i, j] = count_cp


@njit(cache=True)
def choose_actions(
    z: np.ndarray,
    state_d: np.ndarray,
    state_p: np.ndarray,
    state_cp: np.ndarray,
    q_c: np.ndarray,
    q_a: np.ndarray,
    contribution_idx: np.ndarray,
    punishment_idx: np.ndarray,
    contribution_value: np.ndarray,
    punishment_value: np.ndarray,
    epsilon_c: float,
    epsilon_a: float,
    contribution_actions: np.ndarray,
    punishment_actions: np.ndarray,
) -> None:
    L = z.shape[0]
    n_c_actions = contribution_actions.shape[0]
    n_a_actions = punishment_actions.shape[0]
    for i in range(L):
        for j in range(L):
            strategy = z[i, j]
            contribution_idx[i, j] = 0
            punishment_idx[i, j] = 0
            contribution_value[i, j] = 0.0
            punishment_value[i, j] = 0.0

            if strategy == C or strategy == P:
                s_cp = state_cp[i, j]
                greedy_c = random_argmax_contribution(q_c, i, j, s_cp, n_c_actions)
                chosen_c = np.random.randint(0, n_c_actions) if np.random.random() < epsilon_c else greedy_c
                contribution_idx[i, j] = chosen_c
                contribution_value[i, j] = contribution_actions[chosen_c]

            if strategy == P:
                s_d = state_d[i, j]
                s_p = state_p[i, j]
                greedy_a = random_argmax_punishment(q_a, i, j, s_d, s_p, n_a_actions)
                chosen_a = np.random.randint(0, n_a_actions) if np.random.random() < epsilon_a else greedy_a
                punishment_idx[i, j] = chosen_a
                punishment_value[i, j] = punishment_actions[chosen_a]


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
    punisher_cost: np.ndarray,
    defector_fine: np.ndarray,
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
        defector_fine[x, y] += fine
    elif strategy == C:
        payoff[x, y] += public_return - contribution_value[x, y]
    else:
        cost = punishment_value[x, y] * n_d_g
        payoff[x, y] += public_return - contribution_value[x, y] - cost
        punisher_cost[x, y] += cost


@njit(cache=True, inline="always")
def add_group_payoff(
    z: np.ndarray,
    contribution_value: np.ndarray,
    punishment_value: np.ndarray,
    payoff: np.ndarray,
    punisher_cost: np.ndarray,
    defector_fine: np.ndarray,
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

    add_payoff_to_member(payoff, punisher_cost, defector_fine, z, contribution_value, punishment_value, x0, y0, public_return, fine, n_d)
    add_payoff_to_member(payoff, punisher_cost, defector_fine, z, contribution_value, punishment_value, x1, y1, public_return, fine, n_d)
    add_payoff_to_member(payoff, punisher_cost, defector_fine, z, contribution_value, punishment_value, x2, y2, public_return, fine, n_d)
    add_payoff_to_member(payoff, punisher_cost, defector_fine, z, contribution_value, punishment_value, x3, y3, public_return, fine, n_d)
    add_payoff_to_member(payoff, punisher_cost, defector_fine, z, contribution_value, punishment_value, x4, y4, public_return, fine, n_d)


@njit(cache=True)
def compute_payoff(
    z: np.ndarray,
    contribution_value: np.ndarray,
    punishment_value: np.ndarray,
    payoff: np.ndarray,
    punisher_cost: np.ndarray,
    defector_fine: np.ndarray,
    r_value: float,
    beta_F: float,
) -> tuple[float, float, float]:
    L = z.shape[0]
    for i in range(L):
        for j in range(L):
            payoff[i, j] = 0.0
            punisher_cost[i, j] = 0.0
            defector_fine[i, j] = 0.0

    for i in range(L):
        im = prev_index(i, L)
        ip = next_index(i, L)
        for j in range(L):
            jm = prev_index(j, L)
            jp = next_index(j, L)
            add_group_payoff(
                z,
                contribution_value,
                punishment_value,
                payoff,
                punisher_cost,
                defector_fine,
                i,
                j,
                im,
                j,
                ip,
                j,
                i,
                jm,
                i,
                jp,
                r_value,
                beta_F,
            )

    net_payoff_sum = 0.0
    punisher_cost_sum = 0.0
    defector_fine_sum = 0.0
    for i in range(L):
        for j in range(L):
            net_payoff_sum += payoff[i, j]
            if z[i, j] == P:
                punisher_cost_sum += punisher_cost[i, j]
            elif z[i, j] == D:
                defector_fine_sum += defector_fine[i, j]
    return net_payoff_sum, punisher_cost_sum, defector_fine_sum


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
def max_q_at_contribution_state(q_table: np.ndarray, i: int, j: int, s: int, n_actions: int) -> float:
    max_value = q_table[i, j, s, 0]
    for a in range(1, n_actions):
        value = q_table[i, j, s, a]
        if value > max_value:
            max_value = value
    return max_value


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
def update_q_tables(
    z_old: np.ndarray,
    payoff: np.ndarray,
    state_d_old: np.ndarray,
    state_p_old: np.ndarray,
    state_cp_old: np.ndarray,
    state_d_next: np.ndarray,
    state_p_next: np.ndarray,
    state_cp_next: np.ndarray,
    q_c: np.ndarray,
    q_a: np.ndarray,
    contribution_idx: np.ndarray,
    punishment_idx: np.ndarray,
    η_c: float,
    gamma_c: float,
    η_a: float,
    gamma_a: float,
    contribution_actions: np.ndarray,
    punishment_actions: np.ndarray,
) -> None:
    L = z_old.shape[0]
    n_c_actions = contribution_actions.shape[0]
    n_a_actions = punishment_actions.shape[0]
    neighbor_count = GROUP_SIZE - 1.0

    c_min = contribution_actions[0]
    c_max = contribution_actions[0]
    for a in range(1, n_c_actions):
        if contribution_actions[a] < c_min:
            c_min = contribution_actions[a]
        if contribution_actions[a] > c_max:
            c_max = contribution_actions[a]
    c_range = c_max - c_min

    a_min = punishment_actions[0]
    a_max = punishment_actions[0]
    for a in range(1, n_a_actions):
        if punishment_actions[a] < a_min:
            a_min = punishment_actions[a]
        if punishment_actions[a] > a_max:
            a_max = punishment_actions[a]
    a_range = a_max - a_min

    for i in range(L):
        for j in range(L):
            if z_old[i, j] == C or z_old[i, j] == P:
                s_cp0 = state_cp_old[i, j]
                s_cp1 = state_cp_next[i, j]
                action_c = contribution_idx[i, j]
                old_q_c = q_c[i, j, s_cp0, action_c]
                next_max_c = max_q_at_contribution_state(q_c, i, j, s_cp1, n_c_actions)
                c_target = c_min + c_range * (s_cp0 / neighbor_count)
                c_value = contribution_actions[action_c]
                reward_c = 1.0 - abs(c_value - c_target) / c_range
                q_c[i, j, s_cp0, action_c] = old_q_c + η_c * (reward_c + gamma_c * next_max_c - old_q_c)

            if z_old[i, j] == P:
                s_d0 = state_d_old[i, j]
                s_p0 = state_p_old[i, j]
                s_d1 = state_d_next[i, j]
                s_p1 = state_p_next[i, j]
                action_a = punishment_idx[i, j]
                old_q_a = q_a[i, j, s_d0, s_p0, action_a]
                next_max_a = max_q_at_punishment_state(q_a, i, j, s_d1, s_p1, n_a_actions)

                avg_neighbor_payoff = neighbor_payoff_average(payoff, i, j)
                if s_d0 == 0 or payoff[i, j] >= avg_neighbor_payoff:
                    a_target = a_min
                else:
                    a_target = a_min + a_range * (s_d0 / neighbor_count)

                a_value = punishment_actions[action_a]
                reward_a = 1.0 - abs(a_value - a_target) / a_range
                q_a[i, j, s_d0, s_p0, action_a] = old_q_a + η_a * (reward_a + gamma_a * next_max_a - old_q_a)


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


@njit(cache=True)
def simulate_model_e(
    L: int,
    T: int,
    r_value: float,
    beta_F: float,
    K: float,
    η_c: float,
    gamma_c: float,
    η_a: float,
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
    tail_window: int,
    seed: int,
    contribution_actions: np.ndarray,
    punishment_actions: np.ndarray,
) -> tuple[float, float, float, int, int, float, float, float, float]:
    np.random.seed(seed)
    z, z_next = initialize_strategy(L, init_D, init_C, init_P)
    q_c = np.zeros((L, L, NUM_STATES, contribution_actions.shape[0]), dtype=np.float64)
    q_a = np.zeros((L, L, NUM_STATES, NUM_PUNISHER_STATES, punishment_actions.shape[0]), dtype=np.float64)
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
    punisher_cost = np.zeros((L, L), dtype=np.float64)
    defector_fine = np.zeros((L, L), dtype=np.float64)
    epsilon_c = epsilon_c0
    epsilon_a = epsilon_a0
    tail_d = 0.0
    tail_c = 0.0
    tail_p = 0.0
    cumulative_net_payoff = 0.0
    cumulative_punisher_cost = 0.0
    cumulative_defector_fine = 0.0
    defector_active_steps = 0
    cumulative_end_step = 0
    metrics_closed = False
    tail_count = 0
    total_agents = float(L * L)

    for step in range(1, T + 1):
        count_d_step, _, count_p_step = count_strategies(z)
        metrics_open = False
        if not metrics_closed:
            if count_d_step > 0 and count_p_step > 0:
                metrics_open = True
            else:
                metrics_closed = True
        compute_neighbor_states(z, state_d_old, state_p_old, state_cp_old)
        choose_actions(
            z,
            state_d_old,
            state_p_old,
            state_cp_old,
            q_c,
            q_a,
            contribution_idx,
            punishment_idx,
            contribution_value,
            punishment_value,
            epsilon_c,
            epsilon_a,
            contribution_actions,
            punishment_actions,
        )
        step_net_payoff, step_punisher_cost, step_defector_fine = compute_payoff(
            z,
            contribution_value,
            punishment_value,
            payoff,
            punisher_cost,
            defector_fine,
            r_value,
            beta_F,
        )

        if metrics_open:
            cumulative_net_payoff += step_net_payoff / total_agents
            cumulative_punisher_cost += step_punisher_cost / count_p_step
            defector_active_steps += 1
            cumulative_defector_fine += step_defector_fine / count_d_step
            cumulative_end_step = step

        strategy_update(z, payoff, z_next, K)
        compute_neighbor_states(z_next, state_d_next, state_p_next, state_cp_next)
        update_q_tables(
            z,
            payoff,
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
            η_c,
            gamma_c,
            η_a,
            gamma_a,
            contribution_actions,
            punishment_actions,
        )

        epsilon_c = epsilon_c * epsilon_c_dcy
        if epsilon_c < epsilon_c_min:
            epsilon_c = epsilon_c_min
        epsilon_a = epsilon_a * epsilon_a_dcy
        if epsilon_a < epsilon_a_min:
            epsilon_a = epsilon_a_min
        copy_strategy(z_next, z)

        if tail_window > 0 and step > T - tail_window:
            count_d, count_c, count_p = count_strategies(z)
            total = L * L
            tail_d += count_d / total
            tail_c += count_c / total
            tail_p += count_p / total
            tail_count += 1

    cumulative_punishment_burden = cumulative_punisher_cost + cumulative_defector_fine
    if tail_count > 0:
        return (
            tail_d / tail_count,
            tail_c / tail_count,
            tail_p / tail_count,
            defector_active_steps,
            cumulative_end_step,
            cumulative_net_payoff,
            cumulative_punisher_cost,
            cumulative_defector_fine,
            cumulative_punishment_burden,
        )
    count_d, count_c, count_p = count_strategies(z)
    total = L * L
    return (
        count_d / total,
        count_c / total,
        count_p / total,
        defector_active_steps,
        cumulative_end_step,
        cumulative_net_payoff,
        cumulative_punisher_cost,
        cumulative_defector_fine,
        cumulative_punishment_burden,
    )


def warmup_numba() -> None:
    simulate_model_e(
        4,
        2,
        2.5,
        1.0,
        0.5,
        0.8,
        0.8,
        0.8,
        0.8,
        0.1,
        0.01,
        0.99,
        0.1,
        0.01,
        0.99,
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 3.0,
        1,
        123,
        CONTRIBUTION_ACTIONS,
        PUNISHMENT_ACTIONS,
    )


def simulate_one(params: ScanParams, r_value: float, seed: int) -> dict[str, float | int | str]:
    init_D, init_C, init_P, tail_window = validate_params(params)
    start = time.perf_counter()
    (
        rho_d,
        rho_c,
        rho_p,
        defector_active_steps,
        cumulative_end_step,
        cumulative_net_payoff,
        cumulative_punisher_cost,
        cumulative_defector_fine,
        cumulative_punishment_burden,
    ) = simulate_model_e(
        int(params.L),
        int(params.T),
        float(r_value),
        float(params.beta_F),
        float(params.K),
        float(params.η_c),
        float(params.gamma_c),
        float(params.η_a),
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
        int(tail_window),
        int(seed),
        CONTRIBUTION_ACTIONS,
        PUNISHMENT_ACTIONS,
    )
    return {
        "model": MODEL_NAME,
        "beta_F": float(params.beta_F),
        "r": float(r_value),
        "seed": int(seed),
        "rho_D": float(rho_d),
        "rho_C": float(rho_c),
        "rho_P": float(rho_p),
        "cooperation": float(rho_c + rho_p),
        "defector_active_steps": int(defector_active_steps),
        "cumulative_end_step": int(cumulative_end_step),
        "cumulative_net_payoff": float(cumulative_net_payoff),
        "cumulative_punisher_cost": float(cumulative_punisher_cost),
        "cumulative_defector_fine": float(cumulative_defector_fine),
        "cumulative_punishment_burden": float(cumulative_punishment_burden),
        "net_payoff_avg": float(cumulative_net_payoff),
        "punisher_cost_avg": float(cumulative_punisher_cost),
        "defector_fine_avg": float(cumulative_defector_fine),
        "punishment_burden_avg": float(cumulative_punishment_burden),
        "elapsed_s": float(time.perf_counter() - start),
    }


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    summaries: list[dict[str, float | int | str]] = []
    for r_value in sorted({float(row["r"]) for row in rows}):
        r_rows = [row for row in rows if float(row["r"]) == r_value]
        rho_d = np.array([float(row["rho_D"]) for row in r_rows])
        rho_c = np.array([float(row["rho_C"]) for row in r_rows])
        rho_p = np.array([float(row["rho_P"]) for row in r_rows])
        cooperation = np.array([float(row["cooperation"]) for row in r_rows])
        defector_active_steps = np.array([float(row["defector_active_steps"]) for row in r_rows])
        cumulative_end_step = np.array([float(row["cumulative_end_step"]) for row in r_rows])
        cumulative_net_payoff = np.array([float(row["cumulative_net_payoff"]) for row in r_rows])
        cumulative_punisher_cost = np.array([float(row["cumulative_punisher_cost"]) for row in r_rows])
        cumulative_defector_fine = np.array([float(row["cumulative_defector_fine"]) for row in r_rows])
        cumulative_punishment_burden = np.array([float(row["cumulative_punishment_burden"]) for row in r_rows])
        ddof = 1 if len(r_rows) > 1 else 0
        summaries.append(
            {
                "model": MODEL_NAME,
                "beta_F": float(r_rows[0]["beta_F"]),
                "r": r_value,
                "runs": len(r_rows),
                "mean_defector_active_steps": float(defector_active_steps.mean()),
                "std_defector_active_steps": float(defector_active_steps.std(ddof=ddof)),
                "mean_cumulative_end_step": float(cumulative_end_step.mean()),
                "std_cumulative_end_step": float(cumulative_end_step.std(ddof=ddof)),
                "mean_cooperation": float(cooperation.mean()),
                "std_cooperation": float(cooperation.std(ddof=ddof)),
                "mean_rho_D": float(rho_d.mean()),
                "std_rho_D": float(rho_d.std(ddof=ddof)),
                "mean_rho_C": float(rho_c.mean()),
                "std_rho_C": float(rho_c.std(ddof=ddof)),
                "mean_rho_P": float(rho_p.mean()),
                "std_rho_P": float(rho_p.std(ddof=ddof)),
                "mean_cumulative_net_payoff": float(cumulative_net_payoff.mean()),
                "std_cumulative_net_payoff": float(cumulative_net_payoff.std(ddof=ddof)),
                "mean_cumulative_punisher_cost": float(cumulative_punisher_cost.mean()),
                "std_cumulative_punisher_cost": float(cumulative_punisher_cost.std(ddof=ddof)),
                "mean_cumulative_defector_fine": float(cumulative_defector_fine.mean()),
                "std_cumulative_defector_fine": float(cumulative_defector_fine.std(ddof=ddof)),
                "mean_cumulative_punishment_burden": float(cumulative_punishment_burden.mean()),
                "std_cumulative_punishment_burden": float(cumulative_punishment_burden.std(ddof=ddof)),
                "mean_net_payoff_avg": float(cumulative_net_payoff.mean()),
                "std_net_payoff_avg": float(cumulative_net_payoff.std(ddof=ddof)),
                "mean_punisher_cost_avg": float(cumulative_punisher_cost.mean()),
                "std_punisher_cost_avg": float(cumulative_punisher_cost.std(ddof=ddof)),
                "mean_defector_fine_avg": float(cumulative_defector_fine.mean()),
                "std_defector_fine_avg": float(cumulative_defector_fine.std(ddof=ddof)),
                "mean_punishment_burden_avg": float(cumulative_punishment_burden.mean()),
                "std_punishment_burden_avg": float(cumulative_punishment_burden.std(ddof=ddof)),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(params: ScanParams, summaries: list[dict[str, float | int | str]]) -> Path:
    params.output_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(summaries, key=lambda row: float(row["r"]))
    r_arr = np.array([float(row["r"]) for row in rows])
    mean_cooperation = np.array([float(row["mean_cooperation"]) for row in rows])

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(r_arr, mean_cooperation, marker="o", linewidth=2)
    ax.set_xlabel("r")
    ax.set_ylabel(r"$\rho_C + \rho_P$")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(float(r_arr.min()) - 0.05, float(r_arr.max()) + 0.05)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_title("Model E: cooperation vs r")
    fig.tight_layout()

    path = params.output_dir / "E_cooperation.png"
    fig.savefig(path, dpi=params.dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def model_label() -> str:
    return "Model E: adaptive c + adaptive a"


def print_line(text: str) -> None:
    if tqdm is not None:
        tqdm.write(text)
    else:
        print(text)


def print_run(row: dict[str, float | int | str]) -> None:
    print_line(
        f"run | model={model_label()} r={float(row['r']):.6g} seed={int(row['seed'])} "
        f"rho_C={float(row['rho_C']):.6g} "
        f"rho_P={float(row['rho_P']):.6g} "
        f"rho_C+P={float(row['cooperation']):.6g} "
        f"rho_D={float(row['rho_D']):.6g} "
        f"D_active_steps={int(row['defector_active_steps'])} "
        f"cum_end={int(row['cumulative_end_step'])} "
        f"cum_net={float(row['cumulative_net_payoff']):.6g} "
        f"cum_punisher_cost={float(row['cumulative_punisher_cost']):.6g} "
        f"cum_defector_fine={float(row['cumulative_defector_fine']):.6g} "
        f"elapsed={float(row['elapsed_s']):.2f}s"
    )


def print_summary(row: dict[str, float | int | str]) -> None:
    print(
        f"mean | model={model_label()} r={float(row['r']):.6g} runs={int(row['runs'])} "
        f"rho_C={float(row['mean_rho_C']):.6g} "
        f"rho_P={float(row['mean_rho_P']):.6g} "
        f"rho_C+P={float(row['mean_cooperation']):.6g} "
        f"rho_D={float(row['mean_rho_D']):.6g} "
        f"D_active_steps={float(row['mean_defector_active_steps']):.6g} "
        f"cum_end={float(row['mean_cumulative_end_step']):.6g} "
        f"cum_net={float(row['mean_cumulative_net_payoff']):.6g} "
        f"cum_punisher_cost={float(row['mean_cumulative_punisher_cost']):.6g} "
        f"cum_defector_fine={float(row['mean_cumulative_defector_fine']):.6g}"
    )


def run_scan(params: ScanParams) -> tuple[Path, Path, Path, list[Path]]:
    validate_params(params)
    r_values = parse_r_values(params)
    validate_r_values(r_values)
    seeds = [params.seed_start + i for i in range(params.runs)]
    tasks = [(r_value, seed) for r_value in r_values for seed in seeds]

    if params.warmup:
        print("compiling numba functions...")
        start = time.perf_counter()
        warmup_numba()
        print(f"numba warmup done: {time.perf_counter() - start:.2f}s")

    rows: list[dict[str, float | int | str]] = []
    start = time.perf_counter()
    if params.workers == 1:
        iterator = tasks
        if params.progress and tqdm is not None:
            iterator = tqdm(tasks, desc="Model E", unit="run", dynamic_ncols=True)
        for r_value, seed in iterator:
            row = simulate_one(params, r_value, seed)
            rows.append(row)
            print_run(row)
    else:
        max_workers = min(params.workers, len(tasks))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(simulate_one, params, r_value, seed)
                for r_value, seed in tasks
            ]
            iterator = as_completed(futures)
            if params.progress and tqdm is not None:
                iterator = tqdm(iterator, total=len(futures), desc="Model E", unit="run", dynamic_ncols=True)
            for future in iterator:
                row = future.result()
                rows.append(row)
                print_run(row)

    rows.sort(key=lambda row: (float(row["r"]), int(row["seed"])))
    summaries = summarize(rows)
    for row in summaries:
        print_summary(row)
    print(f"\nscan time: {time.perf_counter() - start:.2f}s")

    raw_csv = params.output_dir / "E_runs.csv"
    summary_csv = params.output_dir / "E_summary.csv"
    write_csv(raw_csv, rows)
    write_csv(summary_csv, summaries)
    plot_path = save_plot(params, summaries)
    payoff_plot_paths = save_payoff_plots(params.output_dir, MODEL_NAME, summaries, params.dpi)
    return raw_csv, summary_csv, plot_path, payoff_plot_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan r values for Model E.")
    parser.add_argument("--L", type=int, default=CONFIG.L)
    parser.add_argument("--T", type=int, default=CONFIG.T)
    parser.add_argument("--beta-F", type=float, default=CONFIG.beta_F)
    parser.add_argument("--K", type=float, default=CONFIG.K)
    parser.add_argument("--eta-c", "--η-c", "--alpha-c", dest="eta_c", type=float, default=CONFIG.η_c)
    parser.add_argument("--gamma-c", type=float, default=CONFIG.gamma_c)
    parser.add_argument("--eta-a", "--η-a", "--alpha-a", dest="eta_a", type=float, default=CONFIG.η_a)
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
        η_c=args.eta_c,
        gamma_c=args.gamma_c,
        η_a=args.eta_a,
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
    params = params_from_args(build_parser().parse_args())
    raw_csv, summary_csv, plot_path, payoff_plot_paths = run_scan(params)
    print(f"raw results: {raw_csv}")
    print(f"combined summary results: {summary_csv}")
    print(f"plot: {plot_path}")
    for path in payoff_plot_paths:
        print(f"payoff plot: {path}")


if __name__ == "__main__":
    main()
