from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

try:
    from numba import njit
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing numba. Please install it first: pip install numba") from exc

# ----------------------------
# Inlined Model E helpers from scripts/scan_r/E.py
# ----------------------------
D, C, P = 0, 1, 2

GROUP_SIZE = 5

NUM_STATES = 5

NUM_PUNISHER_STATES = 5

CONTRIBUTION_ACTIONS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float64)

PUNISHMENT_ACTIONS = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)

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
def random_argmax_action(q_table: np.ndarray, i: int, j: int, s: int, n_actions: int) -> int:
    # docstring removed
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
def random_argmax_action_punishment(
    q_table: np.ndarray,
    i: int,
    j: int,
    s_d: int,
    s_p: int,
    n_actions: int,
) -> int:
    # docstring removed
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
    contribution_value: np.ndarray,
    punishment_idx: np.ndarray,
    punishment_value: np.ndarray,
    epsilon: float,
    epsilon_a: float,
    contribution_actions: np.ndarray,
    punishment_actions: np.ndarray,
) -> None:
    # docstring removed
    L = z.shape[0]
    n_c_actions = contribution_actions.shape[0]
    n_a_actions = punishment_actions.shape[0]

    for i in range(L):
        for j in range(L):
            strategy = z[i, j]
            s_d = state_d[i, j]
            s_p = state_p[i, j]
            s_cp = state_cp[i, j]

            contribution_idx[i, j] = 0
            contribution_value[i, j] = 0.0
            punishment_idx[i, j] = 0
            punishment_value[i, j] = 0.0

            if strategy == C or strategy == P:
                greedy_c = random_argmax_action(q_c, i, j, s_cp, n_c_actions)
                if np.random.random() < epsilon:
                    chosen_c = np.random.randint(0, n_c_actions)
                else:
                    chosen_c = greedy_c
                contribution_idx[i, j] = chosen_c
                contribution_value[i, j] = contribution_actions[chosen_c]

            if strategy == P:
                greedy_a = random_argmax_action_punishment(q_a, i, j, s_d, s_p, n_a_actions)
                if np.random.random() < epsilon_a:
                    chosen_a = np.random.randint(0, n_a_actions)
                else:
                    chosen_a = greedy_a
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
    # docstring removed
    strategy = z[x, y]
    total_contribution = 0.0
    n_p = 0
    n_d = 0
    sum_punishment = 0.0

    if strategy == D:
        n_d = 1
    else:
        total_contribution = contribution_value[x, y]
        if strategy == P:
            n_p = 1
            sum_punishment = punishment_value[x, y]

    return total_contribution, n_p, n_d, sum_punishment

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
    # docstring removed
    strategy = z[x, y]

    if strategy == D:
        #
        payoff[x, y] += public_return - fine
    elif strategy == C:
        # C: r*C^g/G - c_i
        payoff[x, y] += public_return - contribution_value[x, y]
    else:
        # P: r*C^g/G - c_i - a_i*n_D^g
        payoff[x, y] += public_return - contribution_value[x, y] - punishment_value[x, y] * n_d_g

@njit(cache=True)
def compute_payoff_numba(
    z: np.ndarray,
    contribution_value: np.ndarray,
    punishment_value: np.ndarray,
    payoff: np.ndarray,
    r_value: float,
    beta_F: float,
) -> None:
    # docstring removed
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

            total_contribution_g = 0.0
            n_p_g = 0
            n_d_g = 0
            sum_punishment_g = 0.0

            tc, npg, ndg, sp = add_member_stats(z, contribution_value, punishment_value, i, j)
            total_contribution_g += tc
            n_p_g += npg
            n_d_g += ndg
            sum_punishment_g += sp

            tc, npg, ndg, sp = add_member_stats(z, contribution_value, punishment_value, im, j)
            total_contribution_g += tc
            n_p_g += npg
            n_d_g += ndg
            sum_punishment_g += sp

            tc, npg, ndg, sp = add_member_stats(z, contribution_value, punishment_value, ip, j)
            total_contribution_g += tc
            n_p_g += npg
            n_d_g += ndg
            sum_punishment_g += sp

            tc, npg, ndg, sp = add_member_stats(z, contribution_value, punishment_value, i, jm)
            total_contribution_g += tc
            n_p_g += npg
            n_d_g += ndg
            sum_punishment_g += sp

            tc, npg, ndg, sp = add_member_stats(z, contribution_value, punishment_value, i, jp)
            total_contribution_g += tc
            n_p_g += npg
            n_d_g += ndg
            sum_punishment_g += sp

            if n_p_g > 0:
                a_mean_g = sum_punishment_g / n_p_g
                fine_g = n_p_g * (np.exp(beta_F * a_mean_g) - 1.0)
            else:
                fine_g = 0.0

            public_return_g = r_value * total_contribution_g / GROUP_SIZE

            add_payoff_to_member(payoff, z, contribution_value, punishment_value, i, j, public_return_g, fine_g, n_d_g)

            add_payoff_to_member(payoff, z, contribution_value, punishment_value, im, j, public_return_g, fine_g, n_d_g)

            add_payoff_to_member(payoff, z, contribution_value, punishment_value, ip, j, public_return_g, fine_g, n_d_g)

            add_payoff_to_member(payoff, z, contribution_value, punishment_value, i, jm, public_return_g, fine_g, n_d_g)

            add_payoff_to_member(payoff, z, contribution_value, punishment_value, i, jp, public_return_g, fine_g, n_d_g)

@njit(cache=True)
def strategy_update_numba(
    z: np.ndarray,
    payoff: np.ndarray,
    z_next: np.ndarray,
    K: float,
) -> None:
    # docstring removed
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

            delta_payoff = payoff[ni, nj] - payoff[i, j]
            scaled = delta_payoff / K
            if scaled > 60.0:
                scaled = 60.0
            elif scaled < -60.0:
                scaled = -60.0

            imitate_prob = 1.0 / (1.0 + np.exp(-scaled))
            if np.random.random() < imitate_prob:
                z_next[i, j] = z[ni, nj]
            else:
                z_next[i, j] = z[i, j]

@njit(cache=True, inline="always")
def max_q_at_state(q_table: np.ndarray, i: int, j: int, s: int, n_actions: int) -> float:
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

@njit(cache=True)
def update_q_tables_numba(
    z_old: np.ndarray,
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
    payoff: np.ndarray,
    alpha_c: float,
    gamma_c: float,
    alpha_a: float,
    gamma_a: float,
    contribution_actions: np.ndarray,
    punishment_actions: np.ndarray,
) -> None:
    # docstring removed
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

    a_min = punishment_actions[0]
    a_max = punishment_actions[0]
    for a in range(1, n_a_actions):
        if punishment_actions[a] < a_min:
            a_min = punishment_actions[a]
        if punishment_actions[a] > a_max:
            a_max = punishment_actions[a]

    c_range = c_max - c_min
    a_range = a_max - a_min

    for i in range(L):
        for j in range(L):
            strategy = z_old[i, j]
            if strategy == C or strategy == P:
                s_cp0 = state_cp_old[i, j]
                s_cp1 = state_cp_next[i, j]
                a0 = contribution_idx[i, j]

                old_q = q_c[i, j, s_cp0, a0]
                next_max = max_q_at_state(q_c, i, j, s_cp1, n_c_actions)

                coop_level = s_cp0 / neighbor_count
                c_value = contribution_actions[a0]
                c_target = c_min + c_range * coop_level
                match_c = 1.0 - abs(c_value - c_target) / c_range

                target = match_c + gamma_c * next_max
                q_c[i, j, s_cp0, a0] = old_q + alpha_c * (target - old_q)

    for i in range(L):
        im = prev_index(i, L)
        ip = next_index(i, L)
        for j in range(L):
            if z_old[i, j] == P:
                jm = prev_index(j, L)
                jp = next_index(j, L)

                s_d0 = state_d_old[i, j]
                s_p0 = state_p_old[i, j]
                s_d1 = state_d_next[i, j]
                s_p1 = state_p_next[i, j]
                a0 = punishment_idx[i, j]

                old_q = q_a[i, j, s_d0, s_p0, a0]
                next_max = max_q_at_punishment_state(q_a, i, j, s_d1, s_p1, n_a_actions)

                neighbor_payoff_mean = (payoff[im, j] + payoff[ip, j] + payoff[i, jm] + payoff[i, jp]) / neighbor_count
                d_level = s_d0 / neighbor_count
                a_value = punishment_actions[a0]
                boundary_state = s_d0 > 0 and payoff[i, j] < neighbor_payoff_mean
                if not boundary_state:
                    a_target = a_min
                else:
                    a_target = a_min + a_range * d_level

                match_a = 1.0 - abs(a_value - a_target) / a_range

                target = match_a + gamma_a * next_max
                q_a[i, j, s_d0, s_p0, a0] = old_q + alpha_a * (target - old_q)

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

@dataclass(frozen=True)
class QValueParams:
    L: int = 100
    T: int = 10000
    r: float = 4.0
    seed_start: int = 123
    runs: int = 1

    beta_F: float = 1.5
    K: float = 0.5

    alpha_c: float = 0.8
    gamma_c: float = 0.8
    alpha_a: float = 0.8
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

    contribution_actions: str = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
    punishment_actions: str = "0.1,0.2,0.3,0.4,0.5"

    summary_mode: str = "all"

    output_dir: Path = Path("figures/q_values")
    dpi: int = 600
    warmup: bool = True


CONFIG = QValueParams()


def configure_publication_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.edgecolor": "black",
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.7,
            "ytick.minor.width": 0.7,
            "lines.linewidth": 1.5,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
        }
    )


def format_value(value: float | int) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.4g}".replace(".", "p").replace("-", "m")


def parse_float_list(text: str) -> np.ndarray:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Action list must not be empty.")
    return np.array(values, dtype=np.float64)


def validate_params(params: QValueParams) -> tuple[float, float, float]:
    if params.L <= 0 or params.T <= 0 or params.r <= 0:
        raise ValueError("Invalid L, T, or r.")
    if params.runs <= 0:
        raise ValueError("runs must be positive.")
    if params.summary_mode not in {"all", "active"}:
        raise ValueError("summary_mode must be 'all' or 'active'.")
    if params.K <= 0 or params.beta_F <= 0:
        raise ValueError("Invalid K or beta_F.")
    if not (0.0 < params.alpha_c <= 1.0 and 0.0 < params.alpha_a <= 1.0):
        raise ValueError("Learning rates must be in (0, 1].")
    if not (0.0 <= params.gamma_c < 1.0 and 0.0 <= params.gamma_a < 1.0):
        raise ValueError("Discount factors must be in [0, 1).")
    for start, minimum, decay in (
        (params.epsilon_c0, params.epsilon_c_min, params.epsilon_c_dcy),
        (params.epsilon_a0, params.epsilon_a_min, params.epsilon_a_dcy),
    ):
        if not (0.0 <= start <= 1.0 and 0.0 <= minimum <= 1.0):
            raise ValueError("Invalid epsilon values.")
        if start < minimum:
            raise ValueError("Initial epsilon must be >= minimum epsilon.")
        if not (0.0 < decay < 1.0):
            raise ValueError("Epsilon decay must be in (0, 1).")

    probs = np.array([params.init_D, params.init_C, params.init_P], dtype=np.float64)
    if np.any(probs < 0.0) or probs.sum() <= 0.0:
        raise ValueError("Invalid initial strategy probabilities.")
    probs = probs / probs.sum()
    return float(probs[0]), float(probs[1]), float(probs[2])


@njit(cache=True)
def simulate_q_values_numba(
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)

    z = np.empty((L, L), dtype=np.int8)
    z_next = np.empty((L, L), dtype=np.int8)

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

    for _step in range(1, T + 1):
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

        epsilon_c = epsilon_c * epsilon_c_dcy
        if epsilon_c < epsilon_c_min:
            epsilon_c = epsilon_c_min

        epsilon_a = epsilon_a * epsilon_a_dcy
        if epsilon_a < epsilon_a_min:
            epsilon_a = epsilon_a_min

        copy_strategy(z_next, z)

    compute_neighbor_states(z, state_d_old, state_p_old, state_cp_old)

    q_c_all = np.zeros((NUM_STATES, contribution_actions.shape[0]), dtype=np.float64)
    q_a_all = np.zeros((NUM_STATES, NUM_PUNISHER_STATES, punishment_actions.shape[0]), dtype=np.float64)
    total_sites = L * L
    for i in range(L):
        for j in range(L):
            for s in range(NUM_STATES):
                for a in range(contribution_actions.shape[0]):
                    q_c_all[s, a] += q_c[i, j, s, a] / total_sites
            for sd in range(NUM_STATES):
                for sp in range(NUM_PUNISHER_STATES):
                    for a in range(punishment_actions.shape[0]):
                        q_a_all[sd, sp, a] += q_a[i, j, sd, sp, a] / total_sites

    q_c_active_sum = np.zeros((NUM_STATES, contribution_actions.shape[0]), dtype=np.float64)
    q_c_active_count = np.zeros(NUM_STATES, dtype=np.float64)
    q_a_active_sum = np.zeros((NUM_STATES, NUM_PUNISHER_STATES, punishment_actions.shape[0]), dtype=np.float64)
    q_a_active_count = np.zeros((NUM_STATES, NUM_PUNISHER_STATES), dtype=np.float64)

    for i in range(L):
        for j in range(L):
            if z[i, j] == C or z[i, j] == P:
                s_cp = state_cp_old[i, j]
                q_c_active_count[s_cp] += 1.0
                for a in range(contribution_actions.shape[0]):
                    q_c_active_sum[s_cp, a] += q_c[i, j, s_cp, a]

            if z[i, j] == P:
                s_d = state_d_old[i, j]
                s_p = state_p_old[i, j]
                q_a_active_count[s_d, s_p] += 1.0
                for a in range(punishment_actions.shape[0]):
                    q_a_active_sum[s_d, s_p, a] += q_a[i, j, s_d, s_p, a]

    count_d, count_c, count_p = count_strategies(z)
    densities = np.array(
        [
            count_d / total_sites,
            count_c / total_sites,
            count_p / total_sites,
            (count_c + count_p) / total_sites,
        ],
        dtype=np.float64,
    )

    return (
        q_c_all,
        q_a_all,
        q_c_active_sum,
        q_c_active_count,
        q_a_active_sum,
        q_a_active_count,
        densities,
    )


def warmup_numba() -> None:
    simulate_q_values_numba(
        4,
        2,
        2.5,
        1.5,
        0.5,
        0.8,
        0.8,
        0.8,
        0.8,
        0.3,
        0.02,
        0.9,
        0.3,
        0.02,
        0.9,
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 3.0,
        123,
        CONTRIBUTION_ACTIONS,
        PUNISHMENT_ACTIONS,
    )


def divide_with_nan(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    denominator = np.asarray(denominator, dtype=float)
    expanded = denominator
    while expanded.ndim < numerator.ndim:
        expanded = expanded[..., None]
    out = np.full_like(numerator, np.nan, dtype=float)
    np.divide(numerator, expanded, out=out, where=expanded > 0)
    return out


def collect_q_values(
    params: QValueParams,
    contribution_actions: np.ndarray,
    punishment_actions: np.ndarray,
) -> dict[str, np.ndarray]:
    init_D, init_C, init_P = validate_params(params)

    q_c_all_sum = np.zeros((NUM_STATES, len(contribution_actions)), dtype=float)
    q_a_all_sum = np.zeros((NUM_STATES, NUM_PUNISHER_STATES, len(punishment_actions)), dtype=float)
    q_c_active_sum = np.zeros((NUM_STATES, len(contribution_actions)), dtype=float)
    q_c_active_count = np.zeros(NUM_STATES, dtype=float)
    q_a_active_sum = np.zeros((NUM_STATES, NUM_PUNISHER_STATES, len(punishment_actions)), dtype=float)
    q_a_active_count = np.zeros((NUM_STATES, NUM_PUNISHER_STATES), dtype=float)
    density_sum = np.zeros(4, dtype=float)

    for run in range(params.runs):
        seed = params.seed_start + run
        (
            q_c_all,
            q_a_all,
            q_c_active_run_sum,
            q_c_active_run_count,
            q_a_active_run_sum,
            q_a_active_run_count,
            densities,
        ) = simulate_q_values_numba(
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
            int(seed),
            contribution_actions,
            punishment_actions,
        )
        q_c_all_sum += q_c_all
        q_a_all_sum += q_a_all
        q_c_active_sum += q_c_active_run_sum
        q_c_active_count += q_c_active_run_count
        q_a_active_sum += q_a_active_run_sum
        q_a_active_count += q_a_active_run_count
        density_sum += densities

    q_c_all_mean = q_c_all_sum / params.runs
    q_a_all_mean = q_a_all_sum / params.runs
    q_c_active_mean = divide_with_nan(q_c_active_sum, q_c_active_count)
    q_a_active_mean = divide_with_nan(q_a_active_sum, q_a_active_count)

    q_c_selected = q_c_all_mean if params.summary_mode == "all" else q_c_active_mean
    q_a_selected = q_a_all_mean if params.summary_mode == "all" else q_a_active_mean

    return {
        "q_c_all": q_c_all_mean,
        "q_a_all": q_a_all_mean,
        "q_c_active": q_c_active_mean,
        "q_a_active": q_a_active_mean,
        "q_c_selected": q_c_selected,
        "q_a_selected": q_a_selected,
        "q_c_active_count": q_c_active_count,
        "q_a_active_count": q_a_active_count,
        "densities": density_sum / params.runs,
    }


def valid_punishment_pairs() -> list[tuple[int, int]]:
    neighbor_count = GROUP_SIZE - 1
    pairs: list[tuple[int, int]] = []
    for s_d in range(NUM_STATES):
        for s_p in range(NUM_PUNISHER_STATES):
            if s_d + s_p <= neighbor_count:
                pairs.append((s_d, s_p))
    return pairs


def punishment_matrix(q_a: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    matrix = np.zeros((len(pairs), q_a.shape[-1]), dtype=float)
    for row, (s_d, s_p) in enumerate(pairs):
        matrix[row, :] = q_a[s_d, s_p, :]
    return matrix


def finite_limits(*arrays: np.ndarray) -> tuple[float, float]:
    finite = np.concatenate([array[np.isfinite(array)].ravel() for array in arrays])
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return vmin, vmax


def action_labels(actions: np.ndarray) -> list[str]:
    return [f"{action:g}" for action in actions]


def investment_state_labels() -> list[str]:
    return [rf"$s_{{CP}}={state}$" for state in range(NUM_STATES)]


def punishment_state_labels(pairs: list[tuple[int, int]]) -> list[str]:
    return [rf"$({s_d},{s_p})$" for s_d, s_p in pairs]


def padded_y_limits(values: np.ndarray) -> tuple[float, float]:
    vmin, vmax = finite_limits(values)
    vmin = min(0.0, vmin)
    vmax = max(0.0, vmax)
    span = vmax - vmin
    if np.isclose(span, 0.0):
        span = 1.0
    return vmin - 0.06 * span, vmax + 0.10 * span


def draw_state_bar_panels(
    values: np.ndarray,
    actions: np.ndarray,
    state_labels: list[str],
    title: str,
    ylabel: str,
    action_label: str,
    color: str,
    ncols: int,
) -> plt.Figure:
    values = np.asarray(values, dtype=float)
    n_states, n_actions = values.shape
    ncols = min(ncols, n_states)
    nrows = int(np.ceil(n_states / ncols))
    fig_width = max(2.0 * ncols, 6.8)
    fig_height = max(1.85 * nrows + 0.55, 2.35)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        sharey=True,
        constrained_layout=True,
    )
    axes_array = np.asarray(axes).reshape(nrows, ncols)
    x = np.arange(n_actions)
    y_min, y_max = padded_y_limits(values)
    labels = action_labels(actions)
    label_rotation = 90 if n_actions > 6 else 0
    horizontal_align = "center"

    for idx, ax in enumerate(axes_array.ravel()):
        if idx >= n_states:
            ax.set_visible(False)
            continue

        row_values = values[idx, :]
        plot_values = np.nan_to_num(row_values, nan=0.0)
        bar_colors = [color if np.isfinite(value) else "#D9D9D9" for value in row_values]
        ax.bar(
            x,
            plot_values,
            width=0.72,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.35,
        )
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.set_ylim(y_min, y_max)
        ax.set_title(state_labels[idx], fontsize=8.8, pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=label_rotation, ha=horizontal_align)
        ax.tick_params(top=False, right=False)
        ax.margins(x=0.04)

        if idx % ncols == 0:
            ax.set_ylabel(ylabel)
        if idx // ncols == nrows - 1:
            ax.set_xlabel(action_label)

    fig.suptitle(title, fontsize=10.2, fontweight="bold")
    return fig


def draw_q_table_heatmap(
    values: np.ndarray,
    actions: np.ndarray,
    state_labels: list[str],
    title: str,
    action_label: str,
    ylabel: str,
) -> plt.Figure:
    values = np.asarray(values, dtype=float)
    vmin, vmax = finite_limits(values)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.colormaps["viridis"].copy()
    cmap.set_bad("#F2F2F2")

    fig_width = max(0.48 * values.shape[1] + 2.0, 5.4)
    fig_height = max(0.30 * values.shape[0] + 1.55, 2.7)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    masked_values = np.ma.masked_invalid(values)
    image = ax.imshow(masked_values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10.0, fontweight="bold", pad=5)
    ax.set_xlabel(action_label)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_xticklabels(action_labels(actions), rotation=45 if values.shape[1] > 6 else 0, ha="right")
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_yticklabels(state_labels, fontsize=7.0 if values.shape[0] > 10 else 7.8)

    ax.set_xticks(np.arange(values.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(values.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(top=False, right=False)

    text_size = 5.9 if values.shape[0] > 10 else 6.4
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                text_color = "white" if norm(value) < 0.43 else "black"
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=text_size,
                    color=text_color,
                )
            else:
                ax.text(col, row, "NA", ha="center", va="center", fontsize=text_size, color="#666666")

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(r"Mean Q value, $\bar{Q}$")
    return fig


def q_value_rows(
    q_c: np.ndarray,
    q_a: np.ndarray,
    contribution_actions: np.ndarray,
    punishment_actions: np.ndarray,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    c_rows = []
    for s_cp in range(q_c.shape[0]):
        for action_idx, action in enumerate(contribution_actions):
            c_rows.append(
                {
                    "table": "investment",
                    "mode": mode,
                    "state_cp": s_cp,
                    "action_index": action_idx,
                    "action_value": float(action),
                    "mean_q": float(q_c[s_cp, action_idx]),
                }
            )

    a_rows = []
    for s_d, s_p in valid_punishment_pairs():
        for action_idx, action in enumerate(punishment_actions):
            a_rows.append(
                {
                    "table": "punishment",
                    "mode": mode,
                    "state_d": s_d,
                    "state_p": s_p,
                    "action_index": action_idx,
                    "action_value": float(action),
                    "mean_q": float(q_a[s_d, s_p, action_idx]),
                }
            )
    return pd.DataFrame(c_rows), pd.DataFrame(a_rows)


def save_figure(fig: plt.Figure, basename: Path, dpi: int) -> list[Path]:
    basename.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in [".pdf", ".svg", ".png", ".tiff"]:
        path = basename.with_suffix(suffix)
        fig.savefig(path, dpi=dpi, facecolor="white")
        outputs.append(path)
    return outputs


def output_prefix(params: QValueParams) -> str:
    code = "E"
    return (
        f"model_{code}_q"
        f"_r{format_value(params.r)}"
        f"_L{params.L}_T{params.T}"
        f"_runs{params.runs}"
        f"_{params.summary_mode}"
    )


def run(params: QValueParams) -> None:
    configure_publication_style()
    contribution_actions = parse_float_list(params.contribution_actions)
    punishment_actions = parse_float_list(params.punishment_actions)

    if params.warmup:
        warmup_numba()

    results = collect_q_values(params, contribution_actions, punishment_actions)
    q_c = results["q_c_selected"]
    q_a = results["q_a_selected"]

    params.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix(params)

    npz_path = params.output_dir / f"{prefix}.npz"
    np.savez_compressed(
        npz_path,
        contribution_actions=contribution_actions,
        punishment_actions=punishment_actions,
        q_c_all=results["q_c_all"],
        q_a_all=results["q_a_all"],
        q_c_active=results["q_c_active"],
        q_a_active=results["q_a_active"],
        q_c_active_count=results["q_c_active_count"],
        q_a_active_count=results["q_a_active_count"],
        densities=results["densities"],
    )

    c_csv, a_csv = q_value_rows(q_c, q_a, contribution_actions, punishment_actions, params.summary_mode)
    c_csv_path = params.output_dir / f"{prefix}_investment_q.csv"
    a_csv_path = params.output_dir / f"{prefix}_punishment_q.csv"
    c_csv.to_csv(c_csv_path, index=False)
    a_csv.to_csv(a_csv_path, index=False)

    pairs = valid_punishment_pairs()
    q_a_rows = punishment_matrix(q_a, pairs)
    c_state_labels = investment_state_labels()
    a_state_labels = punishment_state_labels(pairs)
    mode_text = "all individuals" if params.summary_mode == "all" else "active individuals"

    investment_bars_fig = draw_state_bar_panels(
        q_c,
        contribution_actions,
        c_state_labels,
        rf"Investment Q values ({mode_text})",
        r"Mean Q value, $\bar{Q}_c$",
        r"Investment action, $c$",
        "#0072B2",
        ncols=5,
    )
    investment_bars_outputs = save_figure(
        investment_bars_fig,
        params.output_dir / f"{prefix}_investment_bars",
        params.dpi,
    )
    plt.close(investment_bars_fig)

    punishment_bars_fig = draw_state_bar_panels(
        q_a_rows,
        punishment_actions,
        a_state_labels,
        rf"Punishment Q values ({mode_text})",
        r"Mean Q value, $\bar{Q}_a$",
        r"Punishment action, $a$",
        "#D55E00",
        ncols=5,
    )
    punishment_bars_outputs = save_figure(
        punishment_bars_fig,
        params.output_dir / f"{prefix}_punishment_bars",
        params.dpi,
    )
    plt.close(punishment_bars_fig)

    investment_table_fig = draw_q_table_heatmap(
        q_c,
        contribution_actions,
        c_state_labels,
        rf"Investment Q table ({mode_text})",
        r"Investment action, $c$",
        r"State, $s_{CP}$",
    )
    investment_table_outputs = save_figure(
        investment_table_fig,
        params.output_dir / f"{prefix}_investment_table",
        params.dpi,
    )
    plt.close(investment_table_fig)

    punishment_table_fig = draw_q_table_heatmap(
        q_a_rows,
        punishment_actions,
        a_state_labels,
        rf"Punishment Q table ({mode_text})",
        r"Punishment action, $a$",
        r"State, $(s_D,s_P)$",
    )
    punishment_table_outputs = save_figure(
        punishment_table_fig,
        params.output_dir / f"{prefix}_punishment_table",
        params.dpi,
    )
    plt.close(punishment_table_fig)

    density_labels = ["rho_D", "rho_C", "rho_P", "rho_C_plus_P"]
    figure_outputs = (
        investment_bars_outputs
        + punishment_bars_outputs
        + investment_table_outputs
        + punishment_table_outputs
    )
    print("Saved Q-value summaries:")
    print(npz_path)
    print(c_csv_path)
    print(a_csv_path)
    for path in figure_outputs:
        print(path)
    print("Final densities:")
    for label, value in zip(density_labels, results["densities"]):
        print(f"{label}={value:.6g}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot full state-action Q values for the double Q-learning Model E.")
    parser.add_argument("--L", type=int, default=CONFIG.L)
    parser.add_argument("--T", type=int, default=CONFIG.T)
    parser.add_argument("--r", type=float, default=CONFIG.r)
    parser.add_argument("--seed-start", type=int, default=CONFIG.seed_start)
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
    parser.add_argument("--summary-mode", choices=["all", "active"], default=CONFIG.summary_mode)
    parser.add_argument("--output-dir", type=Path, default=CONFIG.output_dir)
    parser.add_argument("--dpi", type=int, default=CONFIG.dpi)
    parser.add_argument("--no-warmup", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    params = replace(
        CONFIG,
        L=args.L,
        T=args.T,
        r=args.r,
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
        summary_mode=args.summary_mode,
        output_dir=args.output_dir,
        dpi=args.dpi,
        warmup=not args.no_warmup,
    )
    run(params)


if __name__ == "__main__":
    main()
