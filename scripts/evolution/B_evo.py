from __future__ import annotations

# Standalone script: evolution helpers are inlined here intentionally.
import argparse
import time
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

try:
    from numba import njit
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing numba. Please install it first: pip install numba") from exc

# ----------------------------
# Inlined Model A helpers from scripts/scan_r/A.py
# ----------------------------
D, C, P = 0, 1, 2

GROUP_SIZE = 5

@njit(cache=True, inline="always")
def prev_index(x: int, L: int) -> int:
    return L - 1 if x == 0 else x - 1

@njit(cache=True, inline="always")
def next_index(x: int, L: int) -> int:
    return 0 if x == L - 1 else x + 1

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
def choose_fixed_actions(
    z: np.ndarray,
    contribution_value: np.ndarray,
    punishment_value: np.ndarray,
    fixed_contribution: float,
    χ: float,
) -> None:
    L = z.shape[0]
    for i in range(L):
        for j in range(L):
            strategy = z[i, j]
            contribution_value[i, j] = 0.0
            punishment_value[i, j] = 0.0
            if strategy == C or strategy == P:
                contribution_value[i, j] = fixed_contribution
            if strategy == P:
                punishment_value[i, j] = χ

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

# ----------------------------
# Inlined Model B helpers from scripts/scan_r/B.py
# ----------------------------
D, C, P = 0, 1, 2

GROUP_SIZE = 5

CONTRIBUTION_ACTIONS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float64)

PUNISHMENT_ACTIONS = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)

@njit(cache=True, inline="always")
def prev_index(x: int, L: int) -> int:
    return L - 1 if x == 0 else x - 1

@njit(cache=True, inline="always")
def next_index(x: int, L: int) -> int:
    return 0 if x == L - 1 else x + 1

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
def choose_random_actions(
    z: np.ndarray,
    contribution_value: np.ndarray,
    punishment_value: np.ndarray,
    contribution_actions: np.ndarray,
    punishment_actions: np.ndarray,
) -> None:
    L = z.shape[0]
    n_c_actions = contribution_actions.shape[0]
    n_a_actions = punishment_actions.shape[0]
    for i in range(L):
        for j in range(L):
            strategy = z[i, j]
            contribution_value[i, j] = 0.0
            punishment_value[i, j] = 0.0
            if strategy == C or strategy == P:
                contribution_value[i, j] = contribution_actions[np.random.randint(0, n_c_actions)]
            if strategy == P:
                punishment_value[i, j] = punishment_actions[np.random.randint(0, n_a_actions)]

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

choose_punishment_actions = choose_actions

# ----------------------------
# Inlined Model D helpers from scripts/scan_r/D.py
# ----------------------------
D, C, P = 0, 1, 2

GROUP_SIZE = 5

NUM_STATES = 5

CONTRIBUTION_ACTIONS = np.array([0.1,0.2, 0.3, 0.4,0.5,0.6,0.7,0.8,0.9,1.0], dtype=np.float64)

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
def compute_neighbor_cp(z: np.ndarray, state_cp: np.ndarray) -> None:
    L = z.shape[0]
    for i in range(L):
        im = prev_index(i, L)
        ip = next_index(i, L)
        for j in range(L):
            jm = prev_index(j, L)
            jp = next_index(j, L)
            count_cp = 0
            if z[im, j] != D:
                count_cp += 1
            if z[ip, j] != D:
                count_cp += 1
            if z[i, jm] != D:
                count_cp += 1
            if z[i, jp] != D:
                count_cp += 1
            state_cp[i, j] = count_cp

@njit(cache=True)
def choose_contribution_actions(
    z: np.ndarray,
    state_cp: np.ndarray,
    q_c: np.ndarray,
    contribution_idx: np.ndarray,
    contribution_value: np.ndarray,
    punishment_value: np.ndarray,
    epsilon_c: float,
    χ: float,
    contribution_actions: np.ndarray,
) -> None:
    L = z.shape[0]
    n_actions = contribution_actions.shape[0]
    for i in range(L):
        for j in range(L):
            strategy = z[i, j]
            contribution_idx[i, j] = 0
            contribution_value[i, j] = 0.0
            punishment_value[i, j] = 0.0
            if strategy == C or strategy == P:
                s_cp = state_cp[i, j]
                greedy = random_argmax_action(q_c, i, j, s_cp, n_actions)
                chosen = np.random.randint(0, n_actions) if np.random.random() < epsilon_c else greedy
                contribution_idx[i, j] = chosen
                contribution_value[i, j] = contribution_actions[chosen]
            if strategy == P:
                punishment_value[i, j] = χ

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
def add_group_effects(
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
def compute_payoff_and_reward(
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
            add_group_effects(
                z,
                contribution_value,
                punishment_value,
                payoff,
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
def max_q_at_state(q_table: np.ndarray, i: int, j: int, s: int, n_actions: int) -> float:
    max_value = q_table[i, j, s, 0]
    for a in range(1, n_actions):
        value = q_table[i, j, s, a]
        if value > max_value:
            max_value = value
    return max_value

@njit(cache=True)
def update_contribution_q(
    z_old: np.ndarray,
    state_cp_old: np.ndarray,
    state_cp_next: np.ndarray,
    q_c: np.ndarray,
    contribution_idx: np.ndarray,
    payoff: np.ndarray,
    η_c: float,
    gamma_c: float,
    w_c: float,
    contribution_actions: np.ndarray,
) -> None:
    L = z_old.shape[0]
    n_actions = contribution_actions.shape[0]
    neighbor_count = GROUP_SIZE - 1.0
    c_min = contribution_actions[0]
    c_max = contribution_actions[0]
    for a in range(1, n_actions):
        if contribution_actions[a] < c_min:
            c_min = contribution_actions[a]
        if contribution_actions[a] > c_max:
            c_max = contribution_actions[a]
    c_range = c_max - c_min

    for i in range(L):
        im = prev_index(i, L)
        ip = next_index(i, L)
        for j in range(L):
            if z_old[i, j] == C or z_old[i, j] == P:
                jm = prev_index(j, L)
                jp = next_index(j, L)
                s0 = state_cp_old[i, j]
                s1 = state_cp_next[i, j]
                action = contribution_idx[i, j]
                old_q = q_c[i, j, s0, action]
                next_max = max_q_at_state(q_c, i, j, s1, n_actions)
                coop_level = s0 / neighbor_count
                c_value = contribution_actions[action]
                c_target = c_min + c_range * coop_level
                # double_Q.tex defines the immediate contribution reward as R_c = M_c.
                reward = 1.0 - abs(c_value - c_target) / c_range
                q_c[i, j, s0, action] = old_q + η_c * (reward + gamma_c * next_max - old_q)

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

MODEL_TITLES = {
    "A": "Model A: fixed c + fixed chi",
    "B": "Model B: random c + random a",
    "C": "Model C: fixed c + adaptive a",
    "D": "Model D: adaptive c + fixed chi",
}


@dataclass(frozen=True)
class EvolutionParams:
    model: str = "A"
    L: int = 100
    T: int = 10000
    r: float = 2.1
    seed: int = 123

    beta_F: float = 1.5
    K: float = 0.5

    fixed_contribution: float = 1.0
    chi: float = 0.5

    eta_c: float = 0.8
    gamma_c: float = 0.8
    alpha_a: float = 0.8
    gamma_a: float = 0.8
    w_c: float = 1.0
    w_a: float = 1.0

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

    output_dir: Path = Path("data/evolution")
    dpi: int = 600
    warmup: bool = True


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
            "axes.linewidth": 0.9,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.edgecolor": "black",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.minor.size": 2.2,
            "ytick.minor.size": 2.2,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.minor.width": 0.75,
            "ytick.minor.width": 0.75,
            "lines.linewidth": 1.65,
            "legend.fontsize": 8.5,
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
        raise ValueError("Invalid parameter.")
    return np.array(values, dtype=np.float64)


def validate_params(params: EvolutionParams) -> tuple[float, float, float]:
    if params.model not in MODEL_TITLES:
        raise ValueError("model must be one of A, B, C, D.")
    if params.L <= 0 or params.T <= 0 or params.r <= 0:
        raise ValueError("Invalid parameter.")
    if params.K <= 0 or params.beta_F <= 0:
        raise ValueError("Invalid parameter.")
    if params.fixed_contribution < 0 or params.chi < 0:
        raise ValueError("Invalid parameter.")
    if not (0.0 < params.eta_c <= 1.0 and 0.0 < params.alpha_a <= 1.0):
        raise ValueError("Invalid parameter.")
    if not (0.0 <= params.gamma_c < 1.0 and 0.0 <= params.gamma_a < 1.0):
        raise ValueError("Invalid parameter.")
    if not (0.0 <= params.w_c <= 1.0 and 0.0 <= params.w_a <= 1.0):
        raise ValueError("Invalid parameter.")
    for start, minimum, decay in (
        (params.epsilon_c0, params.epsilon_c_min, params.epsilon_c_dcy),
        (params.epsilon_a0, params.epsilon_a_min, params.epsilon_a_dcy),
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
def initialize_strategy(
    L: int,
    init_D: float,
    init_C: float,
    init_P: float,
) -> tuple[np.ndarray, np.ndarray]:
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
def record_densities(z: np.ndarray, densities: np.ndarray, step: int) -> None:
    total = z.shape[0] * z.shape[1]
    count_d, count_c, count_p = count_strategies(z)
    densities[step, 0] = count_d / total
    densities[step, 1] = count_c / total
    densities[step, 2] = count_p / total
    densities[step, 3] = (count_c + count_p) / total


@njit(cache=True)
def simulate_model_a_evolution(
    L: int,
    T: int,
    r_value: float,
    beta_F: float,
    K: float,
    fixed_contribution: float,
    chi: float,
    init_D: float,
    init_C: float,
    init_P: float,
    seed: int,
) -> np.ndarray:
    np.random.seed(seed)
    z, z_next = initialize_strategy(L, init_D, init_C, init_P)
    contribution_value = np.zeros((L, L), dtype=np.float64)
    punishment_value = np.zeros((L, L), dtype=np.float64)
    payoff = np.zeros((L, L), dtype=np.float64)
    densities = np.zeros((T + 1, 4), dtype=np.float64)
    record_densities(z, densities, 0)

    for step in range(1, T + 1):
        choose_fixed_actions(z, contribution_value, punishment_value, fixed_contribution, chi)
        compute_payoff_numba(z, contribution_value, punishment_value, payoff, r_value, beta_F)
        strategy_update_numba(z, payoff, z_next, K)
        copy_strategy(z_next, z)
        record_densities(z, densities, step)

    return densities


@njit(cache=True)
def simulate_model_b_evolution(
    L: int,
    T: int,
    r_value: float,
    beta_F: float,
    K: float,
    init_D: float,
    init_C: float,
    init_P: float,
    seed: int,
    contribution_actions: np.ndarray,
    punishment_actions: np.ndarray,
) -> np.ndarray:
    np.random.seed(seed)
    z, z_next = initialize_strategy(L, init_D, init_C, init_P)
    contribution_value = np.zeros((L, L), dtype=np.float64)
    punishment_value = np.zeros((L, L), dtype=np.float64)
    payoff = np.zeros((L, L), dtype=np.float64)
    densities = np.zeros((T + 1, 4), dtype=np.float64)
    record_densities(z, densities, 0)

    for step in range(1, T + 1):
        choose_random_actions(z, contribution_value, punishment_value, contribution_actions, punishment_actions)
        compute_payoff_numba(z, contribution_value, punishment_value, payoff, r_value, beta_F)
        strategy_update_numba(z, payoff, z_next, K)
        copy_strategy(z_next, z)
        record_densities(z, densities, step)

    return densities


@njit(cache=True)
def simulate_model_c_evolution(
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
    seed: int,
    punishment_actions: np.ndarray,
) -> np.ndarray:
    np.random.seed(seed)
    z, z_next = initialize_strategy(L, init_D, init_C, init_P)
    q_a = np.zeros((L, L, NUM_STATES, NUM_PUNISHER_STATES, punishment_actions.shape[0]), dtype=np.float64)
    state_d_old = np.empty((L, L), dtype=np.int64)
    state_p_old = np.empty((L, L), dtype=np.int64)
    state_cp_old = np.empty((L, L), dtype=np.int64)
    state_d_next = np.empty((L, L), dtype=np.int64)
    state_p_next = np.empty((L, L), dtype=np.int64)
    state_cp_next = np.empty((L, L), dtype=np.int64)
    punishment_idx = np.zeros((L, L), dtype=np.int64)
    contribution_value = np.zeros((L, L), dtype=np.float64)
    punishment_value = np.zeros((L, L), dtype=np.float64)
    payoff = np.zeros((L, L), dtype=np.float64)
    densities = np.zeros((T + 1, 4), dtype=np.float64)
    epsilon_a = epsilon_a0
    record_densities(z, densities, 0)

    for step in range(1, T + 1):
        compute_neighbor_states(z, state_d_old, state_p_old, state_cp_old)
        choose_punishment_actions(
            z,
            state_d_old,
            state_p_old,
            q_a,
            punishment_idx,
            contribution_value,
            punishment_value,
            epsilon_a,
            fixed_contribution,
            punishment_actions,
        )
        compute_payoff_numba(z, contribution_value, punishment_value, payoff, r_value, beta_F)
        strategy_update_numba(z, payoff, z_next, K)
        compute_neighbor_states(z_next, state_d_next, state_p_next, state_cp_next)
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
            punishment_actions,
        )
        epsilon_a = max(epsilon_a * epsilon_a_dcy, epsilon_a_min)
        copy_strategy(z_next, z)
        record_densities(z, densities, step)

    return densities


@njit(cache=True)
def simulate_model_d_evolution(
    L: int,
    T: int,
    r_value: float,
    beta_F: float,
    K: float,
    eta_c: float,
    gamma_c: float,
    w_c: float,
    epsilon_c0: float,
    epsilon_c_min: float,
    epsilon_c_dcy: float,
    chi: float,
    init_D: float,
    init_C: float,
    init_P: float,
    seed: int,
    contribution_actions: np.ndarray,
) -> np.ndarray:
    np.random.seed(seed)
    z, z_next = initialize_strategy(L, init_D, init_C, init_P)
    q_c = np.zeros((L, L, NUM_STATES, contribution_actions.shape[0]), dtype=np.float64)
    state_d_old = np.empty((L, L), dtype=np.int64)
    state_p_old = np.empty((L, L), dtype=np.int64)
    state_cp_old = np.empty((L, L), dtype=np.int64)
    state_d_next = np.empty((L, L), dtype=np.int64)
    state_p_next = np.empty((L, L), dtype=np.int64)
    state_cp_next = np.empty((L, L), dtype=np.int64)
    contribution_idx = np.zeros((L, L), dtype=np.int64)
    contribution_value = np.zeros((L, L), dtype=np.float64)
    punishment_value = np.zeros((L, L), dtype=np.float64)
    payoff = np.zeros((L, L), dtype=np.float64)
    densities = np.zeros((T + 1, 4), dtype=np.float64)
    epsilon_c = epsilon_c0
    record_densities(z, densities, 0)

    for step in range(1, T + 1):
        compute_neighbor_states(z, state_d_old, state_p_old, state_cp_old)
        choose_contribution_actions(
            z,
            state_cp_old,
            q_c,
            contribution_idx,
            contribution_value,
            punishment_value,
            epsilon_c,
            chi,
            contribution_actions,
        )
        compute_payoff_numba(z, contribution_value, punishment_value, payoff, r_value, beta_F)
        strategy_update_numba(z, payoff, z_next, K)
        compute_neighbor_states(z_next, state_d_next, state_p_next, state_cp_next)
        update_contribution_q(
            z,
            state_cp_old,
            state_cp_next,
            q_c,
            contribution_idx,
            payoff,
            eta_c,
            gamma_c,
            w_c,
            contribution_actions,
        )
        epsilon_c = max(epsilon_c * epsilon_c_dcy, epsilon_c_min)
        copy_strategy(z_next, z)
        record_densities(z, densities, step)

    return densities


def warmup_numba(model: str) -> None:
    contribution_actions = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    punishment_actions = np.array([0.1, 0.2], dtype=np.float64)
    if model == "A":
        simulate_model_a_evolution(4, 2, 2.5, 1.0, 0.5, 1.0, 0.3, 1 / 3, 1 / 3, 1 / 3, 123)
    elif model == "B":
        simulate_model_b_evolution(4, 2, 2.5, 1.0, 0.5, 1 / 3, 1 / 3, 1 / 3, 123, contribution_actions, punishment_actions)
    elif model == "C":
        simulate_model_c_evolution(4, 2, 2.5, 1.0, 0.5, 1.0, 0.8, 0.8, 0.0, 0.5, 0.05, 0.99, 1 / 3, 1 / 3, 1 / 3, 123, punishment_actions)
    elif model == "D":
        simulate_model_d_evolution(4, 2, 2.5, 1.0, 0.5, 0.8, 0.8, 0.0, 0.5, 0.05, 0.99, 0.3, 1 / 3, 1 / 3, 1 / 3, 123, contribution_actions)


def simulate_evolution(
    params: EvolutionParams,
    init_D: float,
    init_C: float,
    init_P: float,
    contribution_actions: np.ndarray,
    punishment_actions: np.ndarray,
) -> np.ndarray:
    model = params.model
    if model == "A":
        return simulate_model_a_evolution(
            int(params.L),
            int(params.T),
            float(params.r),
            float(params.beta_F),
            float(params.K),
            float(params.fixed_contribution),
            float(params.chi),
            init_D,
            init_C,
            init_P,
            int(params.seed),
        )
    if model == "B":
        return simulate_model_b_evolution(
            int(params.L),
            int(params.T),
            float(params.r),
            float(params.beta_F),
            float(params.K),
            init_D,
            init_C,
            init_P,
            int(params.seed),
            contribution_actions,
            punishment_actions,
        )
    if model == "C":
        return simulate_model_c_evolution(
            int(params.L),
            int(params.T),
            float(params.r),
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
            int(params.seed),
            punishment_actions,
        )
    if model == "D":
        return simulate_model_d_evolution(
            int(params.L),
            int(params.T),
            float(params.r),
            float(params.beta_F),
            float(params.K),
            float(params.eta_c),
            float(params.gamma_c),
            float(params.w_c),
            float(params.epsilon_c0),
            float(params.epsilon_c_min),
            float(params.epsilon_c_dcy),
            float(params.chi),
            init_D,
            init_C,
            init_P,
            int(params.seed),
            contribution_actions,
        )
    raise ValueError("Invalid parameter.")


def action_tag(actions: np.ndarray) -> str:
    return "-".join(format_value(x) for x in actions)


def build_tag(params: EvolutionParams, contribution_actions: np.ndarray, punishment_actions: np.ndarray) -> str:
    base = (
        f"model{params.model}_evolution_logt_L{params.L}_T{params.T}_r{format_value(params.r)}_s{params.seed}"
        f"_bF{format_value(params.beta_F)}_K{format_value(params.K)}"
    )
    if params.model == "A":
        return base + f"_c{format_value(params.fixed_contribution)}_chi{format_value(params.chi)}"
    if params.model == "B":
        return base + f"_Ac{action_tag(contribution_actions)}_Aa{action_tag(punishment_actions)}"
    if params.model == "C":
        return (
            base
            + f"_c{format_value(params.fixed_contribution)}"
            + f"_aA{format_value(params.alpha_a)}_gA{format_value(params.gamma_a)}_wa{format_value(params.w_a)}"
            + f"_eP{format_value(params.epsilon_a0)}-{format_value(params.epsilon_a_min)}-{format_value(params.epsilon_a_dcy)}"
            + f"_Aa{action_tag(punishment_actions)}"
        )
    return (
        base
        + f"_etaC{format_value(params.eta_c)}_gC{format_value(params.gamma_c)}_wc{format_value(params.w_c)}"
        + f"_chi{format_value(params.chi)}"
        + f"_eC{format_value(params.epsilon_c0)}-{format_value(params.epsilon_c_min)}-{format_value(params.epsilon_c_dcy)}"
        + f"_Ac{action_tag(contribution_actions)}"
    )


def save_one_figure(fig: plt.Figure, basename: Path, dpi: int) -> list[Path]:
    figure_paths: list[Path] = []
    for suffix, save_dpi in ((".pdf", dpi), (".svg", dpi), (".png", dpi), (".tiff", dpi)):
        path = basename.with_suffix(suffix)
        fig.savefig(path, dpi=save_dpi, facecolor="white")
        figure_paths.append(path)
    return figure_paths


def format_evolution_axis(ax: plt.Axes, params: EvolutionParams, ylabel: str) -> None:
    ax.set_xscale("log")
    ax.set_xlim(1, params.T + 1)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"$t+1$")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", handlelength=2.2)


def title_for_model(params: EvolutionParams) -> str:
    title = rf"Model {params.model}, $r={params.r:g}$"
    if params.model in {"A", "D"}:
        title += rf", $\chi={params.chi:g}$"
    return title


def save_evolution_outputs(
    params: EvolutionParams,
    densities: np.ndarray,
    contribution_actions: np.ndarray,
    punishment_actions: np.ndarray,
) -> tuple[Path, list[Path]]:
    params.output_dir.mkdir(parents=True, exist_ok=True)
    tag = build_tag(params, contribution_actions, punishment_actions)

    t = np.arange(params.T + 1, dtype=np.int64)
    t_plus_1 = t + 1

    csv_path = params.output_dir / f"{tag}.csv"
    header = "t,t_plus_1,rho_D,rho_C,rho_P,rho_C_plus_P"
    csv_data = np.column_stack((t, t_plus_1, densities))
    np.savetxt(
        csv_path,
        csv_data,
        delimiter=",",
        header=header,
        comments="",
        fmt=["%d", "%d", "%.10f", "%.10f", "%.10f", "%.10f"],
    )

    fig_three, ax_three = plt.subplots(figsize=(4.0, 3.0), constrained_layout=True)
    ax_three.plot(t_plus_1, densities[:, 0], color="#F60101", linestyle="-", label=r"$\rho_D$")
    ax_three.plot(t_plus_1, densities[:, 1], color="#1F77B4", linestyle="-", label=r"$\rho_C$")
    ax_three.plot(t_plus_1, densities[:, 2], color="#F3B303", linestyle="-", label=r"$\rho_P$")
    format_evolution_axis(ax_three, params, "Strategy density")
    ax_three.set_title(title_for_model(params), fontsize=11, pad=4)

    three_basename = params.output_dir / f"{tag}_three_strategies"
    three_paths = save_one_figure(fig_three, three_basename, params.dpi)
    plt.close(fig_three)

    fig_total, ax_total = plt.subplots(figsize=(4.0, 3.0), constrained_layout=True)
    ax_total.plot(
        t_plus_1,
        densities[:, 3],
        color="#000000",
        linestyle="-",
        linewidth=1.9,
        label=r"$\rho_C+\rho_P$",
    )
    format_evolution_axis(ax_total, params, r"$\rho_C+\rho_P$")
    ax_total.set_title(title_for_model(params), fontsize=11, pad=4)

    total_basename = params.output_dir / f"{tag}_C_plus_P"
    total_paths = save_one_figure(fig_total, total_basename, params.dpi)
    plt.close(fig_total)

    return csv_path, three_paths + total_paths


def run_evolution(params: EvolutionParams) -> tuple[Path, list[Path]]:
    configure_publication_style()
    init_D, init_C, init_P = validate_params(params)
    contribution_actions = parse_float_list(params.contribution_actions)
    punishment_actions = parse_float_list(params.punishment_actions)

    if params.warmup:
        print("compiling numba functions...")
        start = time.perf_counter()
        warmup_numba(params.model)
        print(f"numba warmup done: {time.perf_counter() - start:.2f}s")

    densities = simulate_evolution(params, init_D, init_C, init_P, contribution_actions, punishment_actions)
    return save_evolution_outputs(params, densities, contribution_actions, punishment_actions)


def build_parser(config: EvolutionParams) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Save separated {MODEL_TITLES[config.model]} time-evolution curves with a logarithmic t+1 axis."
    )
    parser.add_argument("--L", type=int, default=config.L)
    parser.add_argument("--T", type=int, default=config.T)
    parser.add_argument("--r", type=float, default=config.r)
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--beta-F", type=float, default=config.beta_F)
    parser.add_argument("--K", type=float, default=config.K)
    if config.model in {"A", "C"}:
        parser.add_argument("--fixed-contribution", type=float, default=config.fixed_contribution)
    if config.model in {"A", "D"}:
        parser.add_argument("--chi", "--蠂", dest="chi", type=float, default=config.chi)
    if config.model in {"D"}:
        parser.add_argument("--eta-c", type=float, default=config.eta_c)
        parser.add_argument("--gamma-c", type=float, default=config.gamma_c)
        parser.add_argument("--w-c", type=float, default=config.w_c)
        parser.add_argument("--epsilon-c0", type=float, default=config.epsilon_c0)
        parser.add_argument("--epsilon-c-min", type=float, default=config.epsilon_c_min)
        parser.add_argument("--epsilon-c-dcy", type=float, default=config.epsilon_c_dcy)
        parser.add_argument("--contribution-actions", type=str, default=config.contribution_actions)
    if config.model in {"C"}:
        parser.add_argument("--alpha-a", type=float, default=config.alpha_a)
        parser.add_argument("--gamma-a", type=float, default=config.gamma_a)
        parser.add_argument("--w-a", type=float, default=config.w_a)
        parser.add_argument("--epsilon-a0", type=float, default=config.epsilon_a0)
        parser.add_argument("--epsilon-a-min", type=float, default=config.epsilon_a_min)
        parser.add_argument("--epsilon-a-dcy", type=float, default=config.epsilon_a_dcy)
        parser.add_argument("--punishment-actions", type=str, default=config.punishment_actions)
    if config.model in {"B"}:
        parser.add_argument("--contribution-actions", type=str, default=config.contribution_actions)
        parser.add_argument("--punishment-actions", type=str, default=config.punishment_actions)
    parser.add_argument("--init-D", type=float, default=config.init_D)
    parser.add_argument("--init-C", type=float, default=config.init_C)
    parser.add_argument("--init-P", type=float, default=config.init_P)
    parser.add_argument("--output-dir", type=Path, default=config.output_dir)
    parser.add_argument("--dpi", type=int, default=config.dpi)
    parser.add_argument("--no-warmup", action="store_true")
    return parser


def params_from_args(config: EvolutionParams, args: argparse.Namespace) -> EvolutionParams:
    values = {
        "L": args.L,
        "T": args.T,
        "r": args.r,
        "seed": args.seed,
        "beta_F": args.beta_F,
        "K": args.K,
        "init_D": args.init_D,
        "init_C": args.init_C,
        "init_P": args.init_P,
        "output_dir": args.output_dir,
        "dpi": args.dpi,
        "warmup": not args.no_warmup,
    }
    for name in (
        "fixed_contribution",
        "chi",
        "eta_c",
        "gamma_c",
        "w_c",
        "epsilon_c0",
        "epsilon_c_min",
        "epsilon_c_dcy",
        "alpha_a",
        "gamma_a",
        "w_a",
        "epsilon_a0",
        "epsilon_a_min",
        "epsilon_a_dcy",
        "contribution_actions",
        "punishment_actions",
    ):
        if hasattr(args, name):
            values[name] = getattr(args, name)
    return replace(config, **values)


def main_for_model(config: EvolutionParams) -> None:
    params = params_from_args(config, build_parser(config).parse_args())
    csv_path, figure_paths = run_evolution(params)
    print("evolution data:", csv_path)
    for path in figure_paths:
        print("evolution plot:", path)


CONFIG = replace(
    EvolutionParams(),
    model="B",
    T=10000,
    seed=123,
    output_dir=Path("data/B_evo"),
)


if __name__ == "__main__":
    main_for_model(CONFIG)

