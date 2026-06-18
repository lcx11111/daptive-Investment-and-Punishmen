from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

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


@njit(cache=True)
def store_strategy_snapshot(
    snapshot_idx: int,
    z: np.ndarray,
    snapshots: np.ndarray,
    densities: np.ndarray,
) -> None:
    L = z.shape[0]
    count_d = 0
    count_c = 0
    count_p = 0

    for i in range(L):
        for j in range(L):
            strategy = z[i, j]
            snapshots[snapshot_idx, i, j] = strategy

            if strategy == D:
                count_d += 1
            elif strategy == C:
                count_c += 1
            else:
                count_p += 1

    total = L * L
    densities[snapshot_idx, 0] = count_d / total
    densities[snapshot_idx, 1] = count_c / total
    densities[snapshot_idx, 2] = count_p / total


@njit(cache=True)
def store_action_snapshot(
    snapshot_idx: int,
    z: np.ndarray,
    contribution_idx: np.ndarray,
    punishment_idx: np.ndarray,
    investment_snapshots: np.ndarray,
    punishment_snapshots: np.ndarray,
) -> None:
    L = z.shape[0]
    for i in range(L):
        for j in range(L):
            strategy = z[i, j]
            if strategy == D:
                investment_snapshots[snapshot_idx, i, j] = 0
                punishment_snapshots[snapshot_idx, i, j] = 0
            elif strategy == C:
                investment_snapshots[snapshot_idx, i, j] = contribution_idx[i, j] + 1
                punishment_snapshots[snapshot_idx, i, j] = 0
            else:
                investment_snapshots[snapshot_idx, i, j] = contribution_idx[i, j] + 1
                punishment_snapshots[snapshot_idx, i, j] = punishment_idx[i, j] + 1

# ============================================================
# Model E snapshots from double_Q .tex
#
# Model E:
#   1. C/P learn investment cost c_i(t) with Q-learning.
#   2. P learns punishment cost a_i(t) with Q-learning.
#   3. Strategies D/C/P evolve by synchronous Fermi imitation.
#
# This script saves per-step evolution snapshots under data/snap/r<value>.
# ============================================================


def format_value(value: float | int) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.4g}".replace(".", "p").replace("-", "m")


def parse_float_list(text: str) -> np.ndarray:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Invalid parameter.")
    return np.array(values, dtype=np.float64)


def parse_step_list(text: str, T: int) -> list[int]:
    steps = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    if not steps:
        raise ValueError("Invalid parameter.")
    if steps[0] < 0 or steps[-1] > T:
        raise ValueError("Invalid parameter.")
    return steps


def parse_unbounded_step_list(text: str) -> list[int]:
    steps = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    if not steps or steps[0] < 0:
        raise ValueError("Invalid parameter.")
    return steps


def parse_panel_step_list(text: str, T: int) -> list[int] | None:
    if text.strip().lower() == "auto":
        return None
    return parse_step_list(text, T)


def unique_sorted_indices(indices: list[int]) -> list[int]:
    return sorted(set(indices))


def nearest_index(values: np.ndarray, target: float, start_idx: int = 0) -> int:
    segment = values[start_idx:]
    if segment.size == 0:
        return start_idx
    return int(start_idx + np.argmin(np.abs(segment - target)))


def fill_representative_indices(
    indices: list[int],
    steps: list[int],
    max_count: int,
) -> list[int]:
    selected = unique_sorted_indices(indices)
    if len(selected) >= max_count:
        return trim_representative_indices(selected, max_count)

    # Fill sparse selections with log-spaced time anchors so early dynamics are visible.
    first_step = max(1, int(steps[0]))
    last_step = max(first_step, int(steps[-1]))
    targets = np.geomspace(first_step, last_step, max_count)
    step_array = np.asarray(steps, dtype=float)
    for target in targets:
        idx = int(np.argmin(np.abs(step_array - target)))
        selected = unique_sorted_indices([*selected, idx])
        if len(selected) >= max_count:
            break
    return trim_representative_indices(selected, max_count)


def trim_representative_indices(indices: list[int], max_count: int) -> list[int]:
    selected = unique_sorted_indices(indices)
    if len(selected) <= max_count:
        return selected
    if max_count <= 2:
        return [selected[0], selected[-1]][:max_count]

    middle = selected[1:-1]
    keep = max_count - 2
    if keep <= 0 or not middle:
        return [selected[0], selected[-1]][:max_count]
    positions = np.linspace(0, len(middle) - 1, keep).round().astype(int)
    return [selected[0], *[middle[pos] for pos in positions], selected[-1]]


def select_representative_snapshot_indices(
    steps: list[int],
    densities: np.ndarray,
    max_count: int,
) -> list[int]:
    if max_count < 1:
        raise ValueError("Invalid parameter.")
    if len(steps) <= max_count:
        return list(range(len(steps)))

    cooperation = densities[:, 1] + densities[:, 2]
    selected: list[int] = [0]

    min_idx = int(np.argmin(cooperation))
    has_clear_dip = min_idx > 0 and cooperation[0] - cooperation[min_idx] >= 0.05
    base_idx = min_idx if has_clear_dip else 0
    if has_clear_dip:
        selected.append(min_idx)

    base_level = float(cooperation[base_idx])
    final_level = float(cooperation[-1])
    growth = final_level - base_level
    if growth >= 0.05:
        fractions = (0.50, 0.95) if has_clear_dip else (0.30, 0.65, 0.95)
        for fraction in fractions:
            target = base_level + fraction * growth
            selected.append(nearest_index(cooperation, target, base_idx))
    else:
        selected.append(nearest_index(cooperation, float(np.median(cooperation)), 0))

    selected.append(len(steps) - 1)
    selected = unique_sorted_indices(selected)

    if len(selected) > max_count:
        # Keep the first and last states; thin middle points while preserving time order.
        selected = trim_representative_indices(selected, max_count)

    return fill_representative_indices(selected, steps, max_count)


@dataclass(frozen=True)
class SnapshotParams:
    L: int = 100
    T: int = 5000
    r: float =  2.1
    seed: int = 1213

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

    # double_Q .tex: A_c={0.1,...,1.0}, A_a={0.1,...,0.5}
    contribution_actions: str = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
    punishment_actions: str = "0.1,0.2,0.3,0.4,0.5"

    snapshot_steps: str = "1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,600,700,800,900,1000,1500,2000,2500,3000,3500,4000,4500,5000"
    panel_steps: str = "auto"
    max_panel_steps: int = 5
    save_individual_snapshots: bool = False
    three_r_strategy: bool = False
    compare_rs: str = "2.1,3.0,4.0"
    compare_steps: str = "1,10,100,500,1000"
    evolution_dir: Path = Path("data/evolution")
    output_dir: Path = Path("data/snap")
    dpi: int = 300
    warmup: bool = True


CONFIG = SnapshotParams()


def validate_params(params: SnapshotParams) -> tuple[float, float, float]:
    if params.L <= 0 or params.T <= 0:
        raise ValueError("Invalid parameter.")
    if params.r <= 0:
        raise ValueError("Invalid parameter.")
    if params.K <= 0:
        raise ValueError("Invalid parameter.")
    if params.beta_F <= 0:
        raise ValueError("Invalid parameter.")
    if not (0.0 < params.alpha_c <= 1.0 and 0.0 < params.alpha_a <= 1.0):
        raise ValueError("Invalid parameter.")
    if not (0.0 <= params.gamma_c < 1.0 and 0.0 <= params.gamma_a < 1.0):
        raise ValueError("Invalid parameter.")
    for name, start, minimum, decay in (
        ("epsilon_c", params.epsilon_c0, params.epsilon_c_min, params.epsilon_c_dcy),
        ("epsilon_a", params.epsilon_a0, params.epsilon_a_min, params.epsilon_a_dcy),
    ):
        if not (0.0 <= start <= 1.0 and 0.0 <= minimum <= 1.0):
            raise ValueError("Invalid parameter.")
        if start < minimum:
            raise ValueError("Invalid parameter.")
        if not (0.0 < decay < 1.0):
            raise ValueError("Invalid parameter.")

    probs = np.array([params.init_D, params.init_C, params.init_P], dtype=np.float64)
    if np.any(probs < 0.0) or probs.sum() <= 0.0:
        raise ValueError("Invalid parameter.")
    probs = probs / probs.sum()
    return float(probs[0]), float(probs[1]), float(probs[2])


@njit(cache=True)
def simulate_snapshots_numba(
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
    snapshot_steps: np.ndarray,
    contribution_actions: np.ndarray,
    punishment_actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)

    n_snapshots = snapshot_steps.shape[0]
    snapshots = np.empty((n_snapshots, L, L), dtype=np.int8)
    investment_snapshots = np.empty((n_snapshots, L, L), dtype=np.int16)
    punishment_snapshots = np.empty((n_snapshots, L, L), dtype=np.int16)
    densities = np.zeros((n_snapshots, 3), dtype=np.float64)

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

    next_strategy_snapshot = 0
    while next_strategy_snapshot < n_snapshots and snapshot_steps[next_strategy_snapshot] == 0:
        store_strategy_snapshot(
            next_strategy_snapshot,
            z,
            snapshots,
            densities,
        )
        next_strategy_snapshot += 1

    next_action_snapshot = 0

    epsilon_c = epsilon_c0
    epsilon_a = epsilon_a0

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

        action_step = step - 1
        while next_action_snapshot < n_snapshots and snapshot_steps[next_action_snapshot] == action_step:
            store_action_snapshot(
                next_action_snapshot,
                z,
                contribution_idx,
                punishment_idx,
                investment_snapshots,
                punishment_snapshots,
            )
            next_action_snapshot += 1

        if next_strategy_snapshot >= n_snapshots and next_action_snapshot >= n_snapshots:
            break

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

        while next_strategy_snapshot < n_snapshots and snapshot_steps[next_strategy_snapshot] == step:
            store_strategy_snapshot(
                next_strategy_snapshot,
                z,
                snapshots,
                densities,
            )
            next_strategy_snapshot += 1

        if next_strategy_snapshot >= n_snapshots and next_action_snapshot >= n_snapshots:
            break

    if next_action_snapshot < n_snapshots:
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

        while next_action_snapshot < n_snapshots and snapshot_steps[next_action_snapshot] == T:
            store_action_snapshot(
                next_action_snapshot,
                z,
                contribution_idx,
                punishment_idx,
                investment_snapshots,
                punishment_snapshots,
            )
            next_action_snapshot += 1

    return snapshots, investment_snapshots, punishment_snapshots, densities


def warmup_numba() -> None:
    simulate_snapshots_numba(
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
        np.array([0, 2], dtype=np.int64),
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float64),
        np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64),
    )


def build_tag(params: SnapshotParams, steps: list[int], contribution_actions: np.ndarray, punishment_actions: np.ndarray) -> str:
    c_tag = "-".join(format_value(x) for x in contribution_actions)
    a_tag = "-".join(format_value(x) for x in punishment_actions)
    step_tag = "-".join(str(step) for step in steps)
    return (
        f"modelE_L{params.L}_T{params.T}_r{format_value(params.r)}_s{params.seed}"
        f"_bF{format_value(params.beta_F)}_K{format_value(params.K)}"
        f"_aC{format_value(params.alpha_c)}_gC{format_value(params.gamma_c)}"
        f"_aA{format_value(params.alpha_a)}_gA{format_value(params.gamma_a)}"
        f"_eC{format_value(params.epsilon_c0)}-{format_value(params.epsilon_c_min)}-{format_value(params.epsilon_c_dcy)}"
        f"_eP{format_value(params.epsilon_a0)}-{format_value(params.epsilon_a_min)}-{format_value(params.epsilon_a_dcy)}"
        f"_Ac{c_tag}_Aa{a_tag}_steps{step_tag}"
    )


STRATEGY_COLORS = [
    "#F60101",  # D
    "#1F77B4",  # C
    "#F3B303",  # P
]

ACTION_HEATMAP_CMAP = "jet"


def action_colors(n_classes: int) -> list[str]:
    cmap = plt.get_cmap(ACTION_HEATMAP_CMAP)
    return [mcolors.to_hex(cmap(value)) for value in np.linspace(0.0, 1.0, n_classes)]


def categorical_norm(n_classes: int) -> mcolors.BoundaryNorm:
    return mcolors.BoundaryNorm(np.arange(-0.5, n_classes + 0.5, 1.0), n_classes)


def action_labels(zero_label: str, prefix: str, actions: np.ndarray) -> list[str]:
    return [zero_label, *[f"{prefix}={float(action):g}" for action in actions]]


def action_value_labels(actions: np.ndarray) -> list[str]:
    return ["0", *[f"{float(action):g}" for action in actions]]


def add_legend(
    fig: plt.Figure,
    colors: list[str],
    labels: list[str],
    title: str,
    anchor_y: float,
    ncol: int = 1,
    anchor_x: float = 0.835,
) -> None:
    handles = [mpatches.Patch(facecolor=color, edgecolor="black", linewidth=0.35, label=label) for color, label in zip(colors, labels)]
    fig.legend(
        handles=handles,
        title=title,
        loc="center left",
        bbox_to_anchor=(anchor_x, anchor_y),
        frameon=False,
        fontsize=7,
        title_fontsize=8,
        handlelength=1.0,
        handletextpad=0.45,
        columnspacing=0.8,
        borderaxespad=0.0,
        ncol=ncol,
    )


def add_labeled_color_strip(
    fig: plt.Figure,
    colors: list[str],
    labels: list[str],
    left: float,
    top: float,
    width: float = 0.070,
    swatch_width: float = 0.014,
    swatch_height: float = 0.017,
    row_height: float = 0.023,
) -> None:
    total_height = len(colors) * row_height
    ax = fig.add_axes([left, top - total_height, width, total_height])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()

    swatch_w = swatch_width / width
    swatch_h = swatch_height / total_height
    row_h = row_height / total_height
    label_x = swatch_w + 0.045
    y = 1.0

    for color, label in zip(colors, labels):
        y -= row_h
        rect_y = y + 0.5 * (row_h - swatch_h)
        ax.add_patch(
            mpatches.Rectangle(
                (0.0, rect_y),
                swatch_w,
                swatch_h,
                facecolor=color,
                edgecolor="black",
                linewidth=0.35,
                transform=ax.transAxes,
            )
        )
        ax.text(label_x, y + 0.5 * row_h, label, ha="left", va="center", fontsize=6.4, transform=ax.transAxes)


def save_single_snapshot_plot(
    params: SnapshotParams,
    step: int,
    snapshot_idx: int,
    output_dir: Path,
    snapshots: np.ndarray,
    investment_snapshots: np.ndarray,
    punishment_snapshots: np.ndarray,
    contribution_actions: np.ndarray,
    punishment_actions: np.ndarray,
) -> Path:
    strategy_cmap = mcolors.ListedColormap(STRATEGY_COLORS)
    strategy_norm = categorical_norm(3)

    punishment_colors = action_colors(len(punishment_actions) + 1)
    punishment_cmap = mcolors.ListedColormap(punishment_colors)
    punishment_norm = categorical_norm(len(punishment_colors))

    investment_colors = action_colors(len(contribution_actions) + 1)
    investment_cmap = mcolors.ListedColormap(investment_colors)
    investment_norm = categorical_norm(len(investment_colors))

    fig, axes = plt.subplots(3, 1, figsize=(5.8, 9.0), squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.66, bottom=0.04, top=0.93, hspace=0.18)
    fig.suptitle(f"r = {params.r:g}, t = {step}", fontsize=12, fontweight="bold", y=0.985)

    panels = (
        ("Strategy", snapshots[snapshot_idx], strategy_cmap, strategy_norm),
        ("Punishment", punishment_snapshots[snapshot_idx], punishment_cmap, punishment_norm),
        ("Investment", investment_snapshots[snapshot_idx], investment_cmap, investment_norm),
    )

    for ax, (title, values, cmap, norm) in zip(axes[:, 0], panels):
        ax.imshow(values, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title, fontsize=10, fontweight="bold", pad=7)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    add_legend(fig, STRATEGY_COLORS, ["D", "C", "P"], "Strategy", 0.820, anchor_x=0.705)
    add_legend(
        fig,
        punishment_colors,
        action_labels("D/C: 0", "a", punishment_actions),
        "Punishment cost",
        0.530,
        anchor_x=0.705,
    )
    add_legend(
        fig,
        investment_colors,
        action_labels("D: 0", "c", contribution_actions),
        "Investment",
        0.220,
        ncol=2 if len(contribution_actions) >= 7 else 1,
        anchor_x=0.705,
    )

    path = output_dir / f"snapshot_L{params.L}_T{params.T}_s{params.seed}_t{step}.png"
    fig.savefig(path, dpi=params.dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def save_snapshot_row_plot(
    params: SnapshotParams,
    steps: list[int],
    panel_indices: list[int],
    output_dir: Path,
    snapshots: np.ndarray,
    investment_snapshots: np.ndarray,
    punishment_snapshots: np.ndarray,
    contribution_actions: np.ndarray,
    punishment_actions: np.ndarray,
) -> list[Path]:
    strategy_cmap = mcolors.ListedColormap(STRATEGY_COLORS)
    strategy_norm = categorical_norm(3)

    punishment_colors = action_colors(len(punishment_actions) + 1)
    punishment_cmap = mcolors.ListedColormap(punishment_colors)
    punishment_norm = categorical_norm(len(punishment_colors))

    investment_colors = action_colors(len(contribution_actions) + 1)
    investment_cmap = mcolors.ListedColormap(investment_colors)
    investment_norm = categorical_norm(len(investment_colors))

    ncols = len(panel_indices)
    fig_width = max(7.2, 1.35 * ncols + 1.10)
    fig, axes = plt.subplots(3, ncols, figsize=(fig_width, 4.35), squeeze=False)
    fig.subplots_adjust(left=0.075, right=0.855, bottom=0.065, top=0.94, wspace=0.035, hspace=0.10)

    panels = (
        ("Strategy", snapshots, strategy_cmap, strategy_norm),
        ("Punishment", punishment_snapshots, punishment_cmap, punishment_norm),
        ("Investment", investment_snapshots, investment_cmap, investment_norm),
    )

    for row_idx, (row_label, values, cmap, norm) in enumerate(panels):
        for col_idx, snapshot_idx in enumerate(panel_indices):
            ax = axes[row_idx, col_idx]
            ax.imshow(values[snapshot_idx], cmap=cmap, norm=norm, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(f"t = {steps[snapshot_idx]}", fontsize=7.6, pad=4)
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=7.6, fontweight="bold", labelpad=12)
            for spine in ax.spines.values():
                spine.set_linewidth(0.45)

    legend_specs = (
        (STRATEGY_COLORS, ["D", "C", "P"]),
        (punishment_colors, action_value_labels(punishment_actions)),
        (investment_colors, action_value_labels(contribution_actions)),
    )
    for row_idx, (colors, labels) in enumerate(legend_specs):
        last_panel = axes[row_idx, -1].get_position()
        add_labeled_color_strip(
            fig,
            colors,
            labels,
            left=last_panel.x1 + 0.012,
            top=last_panel.y1,
        )

    step_tag = "-".join(str(steps[idx]) for idx in panel_indices)
    stem = output_dir / f"snapshot_row_L{params.L}_T{params.T}_s{params.seed}_steps{step_tag}"
    png_path = stem.with_suffix(".png")
    tiff_path = stem.with_suffix(".tiff")
    raster_dpi = max(params.dpi, 600)
    fig.savefig(png_path, dpi=raster_dpi, bbox_inches="tight")
    fig.savefig(
        tiff_path,
        dpi=raster_dpi,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    return [png_path, tiff_path]


def format_r_label(value: float) -> str:
    if float(value).is_integer():
        return f"{float(value):.1f}"
    return f"{float(value):g}"


def find_evolution_csvs(evolution_dir: Path, r_values: list[float]) -> dict[str, Path]:
    tokens = {format_value(r_value): f"_r{format_value(r_value)}_runs" for r_value in r_values}
    grouped_by_parent: dict[Path, dict[str, Path]] = {}
    matches_by_r: dict[str, list[Path]] = {token: [] for token in tokens}

    for path in evolution_dir.rglob("*.csv"):
        if not path.name.startswith("modelE_evolution_logt_"):
            continue
        for token, needle in tokens.items():
            if needle in path.name:
                grouped_by_parent.setdefault(path.parent, {})[token] = path
                matches_by_r[token].append(path)

    complete_groups = [
        (parent, paths)
        for parent, paths in grouped_by_parent.items()
        if all(token in paths for token in tokens)
    ]
    if complete_groups:
        _, paths = max(
            complete_groups,
            key=lambda item: min(path.stat().st_mtime for path in item[1].values()),
        )
        return {token: paths[token] for token in tokens}

    selected: dict[str, Path] = {}
    for token, matches in matches_by_r.items():
        if not matches:
            raise FileNotFoundError(f"Missing evolution CSV for r={token} under {evolution_dir}.")
        selected[token] = max(matches, key=lambda path: path.stat().st_mtime)
    return selected


def load_evolution_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    required = ("t_plus_1", "mean_rho_D", "mean_rho_C", "mean_rho_P")
    for name in required:
        if name not in data.dtype.names:
            raise ValueError(f"Evolution CSV is missing column {name}: {path}")
    return (
        np.asarray(data["t_plus_1"], dtype=float),
        np.asarray(data["mean_rho_D"], dtype=float),
        np.asarray(data["mean_rho_C"], dtype=float),
        np.asarray(data["mean_rho_P"], dtype=float),
    )


def format_strategy_evolution_axis(
    ax: plt.Axes,
    csv_path: Path,
    row_idx: int,
    selected_steps: list[int],
) -> None:
    x, rho_d, rho_c, rho_p = load_evolution_csv(csv_path)
    for step in selected_steps:
        ax.axvline(step + 1, color="#D0D0D0", linewidth=0.35, linestyle="--", zorder=0)
    ax.plot(x, rho_d, color=STRATEGY_COLORS[D], linewidth=0.9, label=r"$D$")
    ax.plot(x, rho_c, color=STRATEGY_COLORS[C], linewidth=0.9, label=r"$C$")
    ax.plot(x, rho_p, color=STRATEGY_COLORS[P], linewidth=0.9, label=r"$P$")
    ax.set_xscale("log")
    ax.set_xlim(1, float(np.max(x)))
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("")
    ax.set_ylabel("Fraction", fontsize=6.5)
    if row_idx == 2:
        ax.set_xlabel(r"$t+1$", fontsize=6.5, labelpad=1)
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    if row_idx == 0:
        ax.legend(loc="upper left", fontsize=5.6, handlelength=1.2, borderpad=0.15, labelspacing=0.15)
    ax.tick_params(axis="both", which="major", labelsize=5.8, length=2.2, width=0.55, direction="in")
    ax.tick_params(axis="both", which="minor", length=1.4, width=0.45, direction="in")
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)


def save_three_r_strategy_figure(
    params: SnapshotParams,
    r_values: list[float],
    steps: list[int],
    evolution_csvs: dict[str, Path],
    snapshots_by_r: dict[str, np.ndarray],
) -> list[Path]:
    output_dir = params.output_dir / "three_r_strategy"
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy_cmap = mcolors.ListedColormap(STRATEGY_COLORS)
    strategy_norm = categorical_norm(3)

    fig = plt.figure(figsize=(7.35, 3.55))
    gridspec = fig.add_gridspec(
        3,
        len(steps) + 1,
        width_ratios=[1.18, *([1.0] * len(steps))],
        wspace=0.025,
        hspace=0.18,
        left=0.060,
        right=0.995,
        top=0.905,
        bottom=0.095,
    )

    for row_idx, r_value in enumerate(r_values):
        token = format_value(r_value)
        ax_curve = fig.add_subplot(gridspec[row_idx, 0])
        format_strategy_evolution_axis(ax_curve, evolution_csvs[token], row_idx, steps)

        snapshots = snapshots_by_r[token]
        for col_idx, step in enumerate(steps):
            ax = fig.add_subplot(gridspec[row_idx, col_idx + 1])
            ax.imshow(snapshots[col_idx], cmap=strategy_cmap, norm=strategy_norm, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(f"t={step}", fontsize=8.0, fontweight="normal", pad=3)
            for spine in ax.spines.values():
                spine.set_linewidth(0.55)

    step_tag = "-".join(str(step) for step in steps)
    r_tag = "-".join(format_value(r_value) for r_value in r_values)
    stem = output_dir / f"three_r_strategy_snapshots_L{params.L}_s{params.seed}_r{r_tag}_steps{step_tag}"
    paths: list[Path] = []
    raster_dpi = max(params.dpi, 600)
    for suffix, save_dpi in ((".pdf", params.dpi), (".svg", params.dpi), (".png", raster_dpi), (".tiff", raster_dpi)):
        path = stem.with_suffix(suffix)
        if suffix == ".tiff":
            fig.savefig(path, dpi=save_dpi, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
        else:
            fig.savefig(path, dpi=save_dpi, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def run_three_r_strategy_figure(params: SnapshotParams) -> list[Path]:
    init_D, init_C, init_P = validate_params(params)
    contribution_actions = parse_float_list(params.contribution_actions)
    punishment_actions = parse_float_list(params.punishment_actions)
    r_values = [float(value) for value in parse_float_list(params.compare_rs)]
    steps = parse_unbounded_step_list(params.compare_steps)

    if params.warmup:
        print("compiling numba functions...")
        start = time.perf_counter()
        warmup_numba()
        print(f"numba warmup done: {time.perf_counter() - start:.2f}s")

    evolution_csvs = find_evolution_csvs(params.evolution_dir, r_values)
    snapshots_by_r: dict[str, np.ndarray] = {}
    sim_T = max(params.T, int(max(steps)))
    snapshot_steps = np.asarray(steps, dtype=np.int64)
    for r_value in r_values:
        token = format_value(r_value)
        print(f"simulating strategy snapshots: r={format_r_label(r_value)}, T={','.join(str(step) for step in steps)}")
        snapshots, _, _, _ = simulate_snapshots_numba(
            int(params.L),
            int(sim_T),
            float(r_value),
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
            int(params.seed),
            snapshot_steps,
            contribution_actions,
            punishment_actions,
        )
        snapshots_by_r[token] = snapshots

    print("common T:", ", ".join(str(step) for step in steps))
    return save_three_r_strategy_figure(params, r_values, steps, evolution_csvs, snapshots_by_r)


def save_snapshot_plots(
    params: SnapshotParams,
    steps: list[int],
    panel_indices: list[int],
    snapshots: np.ndarray,
    investment_snapshots: np.ndarray,
    punishment_snapshots: np.ndarray,
    densities: np.ndarray,
    contribution_actions: np.ndarray,
    punishment_actions: np.ndarray,
) -> list[Path]:
    _ = densities
    output_dir = params.output_dir / f"r{format_value(params.r)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = save_snapshot_row_plot(
        params,
        steps,
        panel_indices,
        output_dir,
        snapshots,
        investment_snapshots,
        punishment_snapshots,
        contribution_actions,
        punishment_actions,
    )
    if params.save_individual_snapshots:
        for snapshot_idx in panel_indices:
            paths.append(
                save_single_snapshot_plot(
                    params,
                    steps[snapshot_idx],
                    snapshot_idx,
                    output_dir,
                    snapshots,
                    investment_snapshots,
                    punishment_snapshots,
                    contribution_actions,
                    punishment_actions,
                )
            )
    print("representative T:", ", ".join(str(steps[idx]) for idx in panel_indices))
    return paths


def run_snapshots(params: SnapshotParams) -> list[Path]:
    init_D, init_C, init_P = validate_params(params)
    contribution_actions = parse_float_list(params.contribution_actions)
    punishment_actions = parse_float_list(params.punishment_actions)
    steps = parse_step_list(params.snapshot_steps, params.T)
    manual_panel_steps = parse_panel_step_list(params.panel_steps, params.T)
    if manual_panel_steps is not None:
        steps = sorted(set([*steps, *manual_panel_steps]))

    if params.warmup:
        print("compiling numba functions...")
        start = time.perf_counter()
        warmup_numba()
        print(f"numba warmup done: {time.perf_counter() - start:.2f}s")

    snapshots, investment_snapshots, punishment_snapshots, densities = simulate_snapshots_numba(
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
        int(params.seed),
        np.array(steps, dtype=np.int64),
        contribution_actions,
        punishment_actions,
    )
    if manual_panel_steps is None:
        panel_indices = select_representative_snapshot_indices(steps, densities, params.max_panel_steps)
    else:
        step_to_idx = {step: idx for idx, step in enumerate(steps)}
        panel_indices = [step_to_idx[step] for step in manual_panel_steps]
    return save_snapshot_plots(
        params,
        steps,
        panel_indices,
        snapshots,
        investment_snapshots,
        punishment_snapshots,
        densities,
        contribution_actions,
        punishment_actions,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Save Model E per-step strategy/action snapshots to data/snap/r<value>.")
    parser.add_argument("--L", type=int, default=CONFIG.L)
    parser.add_argument("--T", type=int, default=CONFIG.T)
    parser.add_argument("--r", type=float, default=CONFIG.r)
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
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
    parser.add_argument("--snapshot-steps", type=str, default=CONFIG.snapshot_steps)
    parser.add_argument("--panel-steps", type=str, default=CONFIG.panel_steps, help="Comma-separated T values for the row figure, or 'auto'.")
    parser.add_argument("--max-panel-steps", type=int, default=CONFIG.max_panel_steps)
    parser.add_argument("--individual-snapshots", action="store_true", help="Also save one vertical snapshot figure per selected T.")
    parser.add_argument("--three-r-strategy", action="store_true", help="Save a three-row r=2.1,3.0,4.0 strategy snapshot figure.")
    parser.add_argument("--compare-rs", type=str, default=CONFIG.compare_rs, help="Comma-separated r values for --three-r-strategy.")
    parser.add_argument("--compare-steps", type=str, default=CONFIG.compare_steps, help="Common T values for --three-r-strategy.")
    parser.add_argument("--evolution-dir", type=Path, default=CONFIG.evolution_dir, help="Directory containing existing evolution CSV files.")
    parser.add_argument("--output-dir", type=Path, default=CONFIG.output_dir)
    parser.add_argument("--dpi", type=int, default=CONFIG.dpi)
    parser.add_argument("--no-warmup", action="store_true")
    return parser


def params_from_args(args: argparse.Namespace) -> SnapshotParams:
    return replace(
        CONFIG,
        L=args.L,
        T=args.T,
        r=args.r,
        seed=args.seed,
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
        snapshot_steps=args.snapshot_steps,
        panel_steps=args.panel_steps,
        max_panel_steps=args.max_panel_steps,
        save_individual_snapshots=args.individual_snapshots,
        three_r_strategy=args.three_r_strategy,
        compare_rs=args.compare_rs,
        compare_steps=args.compare_steps,
        evolution_dir=args.evolution_dir,
        output_dir=args.output_dir,
        dpi=args.dpi,
        warmup=not args.no_warmup,
    )


def main() -> None:
    params = params_from_args(build_parser().parse_args())
    if params.three_r_strategy:
        paths = run_three_r_strategy_figure(params)
    else:
        paths = run_snapshots(params)
    if paths:
        print(f"snapshot dir: {paths[0].parent}")
    for path in paths:
        print(f"snapshot plot: {path}")


if __name__ == "__main__":
    main()




