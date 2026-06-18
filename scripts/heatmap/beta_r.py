from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from numba import cuda
    from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
except ImportError as exc:  # pragma: no cover
    raise SystemExit("CUDA GPU is required.")

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

# ============================================================
# Double Q-learning Model E beta_F-r heatmap - single GPU CUDA version
#
# Model E follows the document-defined double Q-learning model:
#   Q_c: S_c x A_c -> R, A_c = {0.1, ..., 1.0}
#   Q_a: S_a x A_a -> R, A_a = {0.1, ..., 0.5}
#   R_c = M_c
#   R_a = M_a
#
# The scan varies:
#   1) punishment fine sensitivity beta_F
#   2) public goods multiplication factor r
#
# It outputs three heatmaps:
#   1) cooperation level rho_CP
#   2) mean contribution cost c_bar
#   3) mean punishment cost a_bar_D_gt_0
#      - global mean punishment cost over D>0 sampling moments
#
# Example:
#   python beta_r_modelE.py --gpu-batch 1024 --threads-per-block 256
# ============================================================

D, C, P = 0, 1, 2
MODEL_NAME = "E"
GROUP_SIZE = 5
NUM_STATES = 5
NUM_PUNISHER_STATES = 5
N_C_ACTIONS = 10
N_A_ACTIONS = 5

CONTRIBUTION_ACTIONS = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], dtype=np.float32)
PUNISHMENT_ACTIONS = np.array([0.1,0.2,0.3,0.4,0.5], dtype=np.float32)

METRIC_NAMES = (
    "avg_payoff",
    "rho_CP",
    "rho_D",
    "rho_C",
    "rho_P",
    "avg_contribution_active",
    "avg_punishment_punishers",
)

METRIC_LABELS = {
    "avg_payoff": "Average payoff",
    "rho_CP": r"Cooperation level $\rho_{C+P}$",
    "rho_D": r"Defector density $\rho_D$",
    "rho_C": r"Cooperator density $\rho_C$",
    "rho_P": r"Punisher density $\rho_P$",
    "avg_contribution_active": r"Mean contribution action $\bar c$",
    "avg_punishment_punishers": r"Global mean punishment action $\bar a_{D>0}$",
}


@dataclass(frozen=True)
class ScanParams:
    L: int = 100
    T: int = 10000

    beta_min: float = 1.0
    beta_max: float = 2.0
    beta_points: int = 41
    r_min: float = 1.0
    r_max: float = 5.0
    r_points: int = 41

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

    runs: int =40
    seed_start: int = 131
    measure_window: int = 500
    measure_every: int = 10

    #
    gpu_batch: int = 1024
    threads_per_block: int = 256
    gpu_id: int = 0

    output_dir: Path = Path("data/E_beta_r_heatmap")
    dpi: int = 600
    progress: bool = True
    warmup: bool = True


CONFIG_SCAN = ScanParams()


# ----------------------------
# CUDA device helpers
# ----------------------------

@cuda.jit(device=True, inline=True)
def _prev_index(x: int, L: int) -> int:
    return L - 1 if x == 0 else x - 1


@cuda.jit(device=True, inline=True)
def _next_index(x: int, L: int) -> int:
    return 0 if x == L - 1 else x + 1


@cuda.jit(device=True, inline=True)
def _cell_index(sim: int, cell: int, N: int) -> int:
    return sim * N + cell


@cuda.jit(device=True, inline=True)
def _q_c_index(base: int, s: int, a: int) -> int:
    # q_c shape: [B, N, 5, N_C_ACTIONS]
    return (base * NUM_STATES + s) * N_C_ACTIONS + a


@cuda.jit(device=True, inline=True)
def _q_a_index(base: int, s_d: int, s_p: int, a: int) -> int:
    # q_a shape: [B, N, 5, 5, N_A_ACTIONS]
    return ((base * NUM_STATES + s_d) * NUM_PUNISHER_STATES + s_p) * N_A_ACTIONS + a


@cuda.jit(device=True, inline=True)
def _contribution_action(a: int) -> float:
    return 0.1 * (a + 1)


@cuda.jit(device=True, inline=True)
def _punishment_action(a: int) -> float:
    return 0.1 * (a + 1)


@cuda.jit(device=True, inline=True)
def _randint_c(rng_states, tid: int) -> int:
    u = xoroshiro128p_uniform_float32(rng_states, tid)
    v = int(u * N_C_ACTIONS)
    if v >= N_C_ACTIONS:
        v = N_C_ACTIONS - 1
    return v


@cuda.jit(device=True, inline=True)
def _randint_a(rng_states, tid: int) -> int:
    u = xoroshiro128p_uniform_float32(rng_states, tid)
    v = int(u * N_A_ACTIONS)
    if v >= N_A_ACTIONS:
        v = N_A_ACTIONS - 1
    return v


@cuda.jit(device=True, inline=True)
def _randint4(rng_states, tid: int) -> int:
    u = xoroshiro128p_uniform_float32(rng_states, tid)
    v = int(u * 4.0)
    if v > 3:
        v = 3
    return v


@cuda.jit(device=True, inline=True)
def _random_argmax_c(q_c, base: int, s: int, rng_states, tid: int) -> int:
    max_v = q_c[_q_c_index(base, s, 0)]
    for a in range(1, N_C_ACTIONS):
        value = q_c[_q_c_index(base, s, a)]
        if value > max_v:
            max_v = value

    count = 0
    for a in range(N_C_ACTIONS):
        if abs(q_c[_q_c_index(base, s, a)] - max_v) <= 1.0e-6:
            count += 1

    chosen_rank = int(xoroshiro128p_uniform_float32(rng_states, tid) * count)
    if chosen_rank >= count:
        chosen_rank = count - 1

    seen = 0
    for a in range(N_C_ACTIONS):
        if abs(q_c[_q_c_index(base, s, a)] - max_v) <= 1.0e-6:
            if seen == chosen_rank:
                return a
            seen += 1
    return 0


@cuda.jit(device=True, inline=True)
def _random_argmax_a(q_a, base: int, s_d: int, s_p: int, rng_states, tid: int) -> int:
    max_v = q_a[_q_a_index(base, s_d, s_p, 0)]
    for a in range(1, N_A_ACTIONS):
        value = q_a[_q_a_index(base, s_d, s_p, a)]
        if value > max_v:
            max_v = value

    count = 0
    for a in range(N_A_ACTIONS):
        if abs(q_a[_q_a_index(base, s_d, s_p, a)] - max_v) <= 1.0e-6:
            count += 1

    chosen_rank = int(xoroshiro128p_uniform_float32(rng_states, tid) * count)
    if chosen_rank >= count:
        chosen_rank = count - 1

    seen = 0
    for a in range(N_A_ACTIONS):
        if abs(q_a[_q_a_index(base, s_d, s_p, a)] - max_v) <= 1.0e-6:
            if seen == chosen_rank:
                return a
            seen += 1
    return 0


@cuda.jit(device=True, inline=True)
def _add_group_member_stats(z, contribution_value, punishment_value, idx: int):
    strategy = z[idx]
    total_c = 0.0
    n_p = 0
    n_d = 0
    sum_a = 0.0
    if strategy == D:
        n_d = 1
    else:
        total_c = contribution_value[idx]
        if strategy == P:
            n_p = 1
            sum_a = punishment_value[idx]
    return total_c, n_p, n_d, sum_a


@cuda.jit(device=True, inline=True)
def _atomic_add_member_payoff(
    z,
    contribution_value,
    punishment_value,
    payoff,
    idx: int,
    public_return: float,
    fine: float,
    n_d_g: int,
):
    strategy = z[idx]
    if strategy == D:
        cuda.atomic.add(payoff, idx, public_return - fine)
    elif strategy == C:
        cuda.atomic.add(payoff, idx, public_return - contribution_value[idx])
    else:
        cuda.atomic.add(payoff, idx, public_return - contribution_value[idx] - punishment_value[idx] * n_d_g)


@cuda.jit(device=True, inline=True)
def _max_q_c(q_c, base: int, s: int) -> float:
    m = q_c[_q_c_index(base, s, 0)]
    for a in range(1, N_C_ACTIONS):
        value = q_c[_q_c_index(base, s, a)]
        if value > m:
            m = value
    return m


@cuda.jit(device=True, inline=True)
def _max_q_a(q_a, base: int, s_d: int, s_p: int) -> float:
    m = q_a[_q_a_index(base, s_d, s_p, 0)]
    for a in range(1, N_A_ACTIONS):
        value = q_a[_q_a_index(base, s_d, s_p, a)]
        if value > m:
            m = value
    return m


# ----------------------------
# CUDA kernels
# ----------------------------

@cuda.jit
def init_z_kernel(z, rng_states, L: int, N: int, init_D: float, init_C: float, total: int):
    tid = cuda.grid(1)
    if tid >= total:
        return
    u = xoroshiro128p_uniform_float32(rng_states, tid)
    threshold_D = init_D
    threshold_C = init_D + init_C
    if u < threshold_D:
        z[tid] = D
    elif u < threshold_C:
        z[tid] = C
    else:
        z[tid] = P


@cuda.jit
def zero_float_kernel(arr, total: int):
    tid = cuda.grid(1)
    if tid < total:
        arr[tid] = 0.0


@cuda.jit
def zero_int_kernel(arr, total: int):
    tid = cuda.grid(1)
    if tid < total:
        arr[tid] = 0


@cuda.jit
def states_and_actions_kernel(
    z,
    q_c,
    q_a,
    state_d,
    state_p,
    state_cp,
    contribution_idx,
    punishment_idx,
    contribution_value,
    punishment_value,
    rng_states,
    L: int,
    N: int,
    epsilon_c: float,
    epsilon_a: float,
    total: int,
):
    tid = cuda.grid(1)
    if tid >= total:
        return

    sim = tid // N
    cell = tid - sim * N
    i = cell // L
    j = cell - i * L

    im = _prev_index(i, L)
    ip = _next_index(i, L)
    jm = _prev_index(j, L)
    jp = _next_index(j, L)

    idx_up = _cell_index(sim, im * L + j, N)
    idx_down = _cell_index(sim, ip * L + j, N)
    idx_left = _cell_index(sim, i * L + jm, N)
    idx_right = _cell_index(sim, i * L + jp, N)

    count_d = 0
    count_p = 0
    count_cp = 0

    s = z[idx_up]
    if s == D:
        count_d += 1
    else:
        count_cp += 1
        if s == P:
            count_p += 1

    s = z[idx_down]
    if s == D:
        count_d += 1
    else:
        count_cp += 1
        if s == P:
            count_p += 1

    s = z[idx_left]
    if s == D:
        count_d += 1
    else:
        count_cp += 1
        if s == P:
            count_p += 1

    s = z[idx_right]
    if s == D:
        count_d += 1
    else:
        count_cp += 1
        if s == P:
            count_p += 1

    state_d[tid] = count_d
    state_p[tid] = count_p
    state_cp[tid] = count_cp

    contribution_idx[tid] = 0
    punishment_idx[tid] = 0
    contribution_value[tid] = 0.0
    punishment_value[tid] = 0.0

    strategy = z[tid]
    if strategy == C or strategy == P:
        greedy_c = _random_argmax_c(q_c, tid, count_cp, rng_states, tid)
        if xoroshiro128p_uniform_float32(rng_states, tid) < epsilon_c:
            chosen_c = _randint_c(rng_states, tid)
        else:
            chosen_c = greedy_c
        contribution_idx[tid] = chosen_c
        contribution_value[tid] = _contribution_action(chosen_c)

    if strategy == P:
        greedy_a = _random_argmax_a(q_a, tid, count_d, count_p, rng_states, tid)
        if xoroshiro128p_uniform_float32(rng_states, tid) < epsilon_a:
                chosen_a = _randint_a(rng_states, tid)
        else:
            chosen_a = greedy_a
        punishment_idx[tid] = chosen_a
        punishment_value[tid] = _punishment_action(chosen_a)


@cuda.jit
def compute_payoff_kernel(
    z,
    contribution_value,
    punishment_value,
    payoff,
    r_values,
    beta_values,
    L: int,
    N: int,
    total: int,
):
    tid = cuda.grid(1)
    if tid >= total:
        return

    sim = tid // N
    cell = tid - sim * N
    i = cell // L
    j = cell - i * L

    im = _prev_index(i, L)
    ip = _next_index(i, L)
    jm = _prev_index(j, L)
    jp = _next_index(j, L)

    idx0 = tid
    idx1 = _cell_index(sim, im * L + j, N)
    idx2 = _cell_index(sim, ip * L + j, N)
    idx3 = _cell_index(sim, i * L + jm, N)
    idx4 = _cell_index(sim, i * L + jp, N)

    total_c = 0.0
    n_p_g = 0
    n_d_g = 0
    sum_a = 0.0

    tc, npg, ndg, sp = _add_group_member_stats(z, contribution_value, punishment_value, idx0)
    total_c += tc
    n_p_g += npg
    n_d_g += ndg
    sum_a += sp

    tc, npg, ndg, sp = _add_group_member_stats(z, contribution_value, punishment_value, idx1)
    total_c += tc
    n_p_g += npg
    n_d_g += ndg
    sum_a += sp

    tc, npg, ndg, sp = _add_group_member_stats(z, contribution_value, punishment_value, idx2)
    total_c += tc
    n_p_g += npg
    n_d_g += ndg
    sum_a += sp

    tc, npg, ndg, sp = _add_group_member_stats(z, contribution_value, punishment_value, idx3)
    total_c += tc
    n_p_g += npg
    n_d_g += ndg
    sum_a += sp

    tc, npg, ndg, sp = _add_group_member_stats(z, contribution_value, punishment_value, idx4)
    total_c += tc
    n_p_g += npg
    n_d_g += ndg
    sum_a += sp

    fine = 0.0
    if n_p_g > 0:
        a_mean = sum_a / n_p_g
        fine = n_p_g * (math.exp(beta_values[sim] * a_mean) - 1.0)

    public_return = r_values[sim] * total_c / GROUP_SIZE

    _atomic_add_member_payoff(z, contribution_value, punishment_value, payoff, idx0, public_return, fine, n_d_g)
    _atomic_add_member_payoff(z, contribution_value, punishment_value, payoff, idx1, public_return, fine, n_d_g)
    _atomic_add_member_payoff(z, contribution_value, punishment_value, payoff, idx2, public_return, fine, n_d_g)
    _atomic_add_member_payoff(z, contribution_value, punishment_value, payoff, idx3, public_return, fine, n_d_g)
    _atomic_add_member_payoff(z, contribution_value, punishment_value, payoff, idx4, public_return, fine, n_d_g)


@cuda.jit
def summary_raw_kernel(
    z,
    contribution_value,
    punishment_value,
    payoff,
    raw_stats,
    L: int,
    N: int,
    total: int,
):
    tid = cuda.grid(1)
    if tid >= total:
        return

    sim = tid // N
    base = sim * 6
    strategy = z[tid]

    cuda.atomic.add(raw_stats, base + 0, payoff[tid])
    if strategy == D:
        cuda.atomic.add(raw_stats, base + 1, 1.0)
    elif strategy == C:
        cuda.atomic.add(raw_stats, base + 2, 1.0)
        cuda.atomic.add(raw_stats, base + 4, contribution_value[tid])
    else:
        cuda.atomic.add(raw_stats, base + 3, 1.0)
        cuda.atomic.add(raw_stats, base + 4, contribution_value[tid])
        cuda.atomic.add(raw_stats, base + 5, punishment_value[tid])


@cuda.jit
def summary_finalize_kernel(
    raw_stats,
    metric_sums,
    sample_counts,
    punishment_weighted_counts,
    N: int,
    B: int,
):
    sim = cuda.grid(1)
    if sim >= B:
        return

    raw = sim * 6
    out = sim * 7

    total_payoff = raw_stats[raw + 0]
    n_d = raw_stats[raw + 1]
    n_c = raw_stats[raw + 2]
    n_p = raw_stats[raw + 3]
    sum_c = raw_stats[raw + 4]
    sum_a = raw_stats[raw + 5]

    n_active = n_c + n_p

    metric_sums[out + 0] += total_payoff / N
    metric_sums[out + 1] += n_active / N
    metric_sums[out + 2] += n_d / N
    metric_sums[out + 3] += n_c / N
    metric_sums[out + 4] += n_p / N

    if n_active > 0.0:
        metric_sums[out + 5] += sum_c / n_active

    # Metric 6:
    # Same global mean as evo_cost's red dashed line:
    # sum(total punishment) / sum(number of punishers), using D>0 moments.
    if n_d > 0.0 and n_p > 0.0:
        metric_sums[out + 6] += sum_a
        punishment_weighted_counts[sim] += n_p

    sample_counts[sim] += 1


@cuda.jit
def strategy_update_kernel(z, payoff, z_next, rng_states, L: int, N: int, K: float, total: int):
    tid = cuda.grid(1)
    if tid >= total:
        return

    sim = tid // N
    cell = tid - sim * N
    i = cell // L
    j = cell - i * L

    direction = _randint4(rng_states, tid)
    ni = i
    nj = j
    if direction == 0:
        ni = _prev_index(i, L)
    elif direction == 1:
        ni = _next_index(i, L)
    elif direction == 2:
        nj = _prev_index(j, L)
    else:
        nj = _next_index(j, L)

    nidx = _cell_index(sim, ni * L + nj, N)
    scaled = (payoff[nidx] - payoff[tid]) / K
    if scaled > 60.0:
        scaled = 60.0
    elif scaled < -60.0:
        scaled = -60.0

    imitate_prob = 1.0 / (1.0 + math.exp(-scaled))
    if xoroshiro128p_uniform_float32(rng_states, tid) < imitate_prob:
        z_next[tid] = z[nidx]
    else:
        z_next[tid] = z[tid]


@cuda.jit
def neighbor_state_kernel(z_like, state_d, state_p, state_cp, L: int, N: int, total: int):
    tid = cuda.grid(1)
    if tid >= total:
        return

    sim = tid // N
    cell = tid - sim * N
    i = cell // L
    j = cell - i * L

    im = _prev_index(i, L)
    ip = _next_index(i, L)
    jm = _prev_index(j, L)
    jp = _next_index(j, L)

    idx_up = _cell_index(sim, im * L + j, N)
    idx_down = _cell_index(sim, ip * L + j, N)
    idx_left = _cell_index(sim, i * L + jm, N)
    idx_right = _cell_index(sim, i * L + jp, N)

    count_d = 0
    count_p = 0
    count_cp = 0

    s = z_like[idx_up]
    if s == D:
        count_d += 1
    else:
        count_cp += 1
        if s == P:
            count_p += 1

    s = z_like[idx_down]
    if s == D:
        count_d += 1
    else:
        count_cp += 1
        if s == P:
            count_p += 1

    s = z_like[idx_left]
    if s == D:
        count_d += 1
    else:
        count_cp += 1
        if s == P:
            count_p += 1

    s = z_like[idx_right]
    if s == D:
        count_d += 1
    else:
        count_cp += 1
        if s == P:
            count_p += 1

    state_d[tid] = count_d
    state_p[tid] = count_p
    state_cp[tid] = count_cp


@cuda.jit
def q_update_and_copy_kernel(
    z_old,
    z_next,
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
    L: int,
    N: int,
    alpha_c: float,
    gamma_c: float,
    alpha_a: float,
    gamma_a: float,
    total: int,
):
    tid = cuda.grid(1)
    if tid >= total:
        return

    sim = tid // N
    cell = tid - sim * N
    i = cell // L
    j = cell - i * L
    strategy = z_old[tid]

    im = _prev_index(i, L)
    ip = _next_index(i, L)
    jm = _prev_index(j, L)
    jp = _next_index(j, L)
    idx_up = _cell_index(sim, im * L + j, N)
    idx_down = _cell_index(sim, ip * L + j, N)
    idx_left = _cell_index(sim, i * L + jm, N)
    idx_right = _cell_index(sim, i * L + jp, N)

    # ------------------------------------------------------------
    # Contribution Q-table update for document-defined Model E:
    #   c* = 0.1 + 0.9 * (n_CP / 4)
    #   M_c = 1 - |c - c*| / 0.9
    #   R_c = M_c
    # ------------------------------------------------------------
    if strategy == C or strategy == P:
        s0 = state_cp_old[tid]
        s1 = state_cp_next[tid]
        a0 = contribution_idx[tid]

        old_q = q_c[_q_c_index(tid, s0, a0)]
        next_max = _max_q_c(q_c, tid, s1)

        coop_level = s0 / 4.0
        c_value = _contribution_action(a0)
        c_target = 0.1 + 0.9 * coop_level
        reward_c = 1.0 - abs(c_value - c_target) / 0.9

        target_c = reward_c + gamma_c * next_max
        q_c[_q_c_index(tid, s0, a0)] = old_q + alpha_c * (target_c - old_q)

    # ------------------------------------------------------------
    # Punishment Q-table update for document-defined Model E:
    #   a* = 0.1, if n_D = 0
    #   a* = 0.1, if n_D > 0 and Pi_i >= neighbor mean payoff
    #   a* = 0.1 + 0.4 * (n_D / 4), otherwise
    #   M_a = 1 - |a - a*| / 0.4
    #   R_a = M_a
    # ------------------------------------------------------------
    if strategy == P:
        sd0 = state_d_old[tid]
        sp0 = state_p_old[tid]
        sd1 = state_d_next[tid]
        sp1 = state_p_next[tid]
        a0 = punishment_idx[tid]

        old_q_a = q_a[_q_a_index(tid, sd0, sp0, a0)]
        next_max_a = _max_q_a(q_a, tid, sd1, sp1)

        neighbor_payoff_mean = (
            payoff[idx_up] + payoff[idx_down] + payoff[idx_left] + payoff[idx_right]
        ) / 4.0

        a_value = _punishment_action(a0)
        if sd0 > 0 and payoff[tid] < neighbor_payoff_mean:
            d_level = sd0 / 4.0
            a_target = 0.1 + 0.4 * d_level
        else:
            a_target = 0.1

        reward_a = 1.0 - abs(a_value - a_target) / 0.4

        target_a = reward_a + gamma_a * next_max_a
        q_a[_q_a_index(tid, sd0, sp0, a0)] = old_q_a + alpha_a * (target_a - old_q_a)

    # Copy strategy z_next -> z_old for the next evolution round.
    z_old[tid] = z_next[tid]


# ----------------------------
# Python host side
# ----------------------------

def validate_scan_params(params: ScanParams) -> tuple[float, float, float]:
    if params.L <= 0 or params.T <= 0:
        raise ValueError("Invalid parameter.")
    if params.beta_points < 2 or params.r_points < 2:
        raise ValueError("Invalid parameter.")
    if params.beta_min < 0 or params.beta_max <= params.beta_min:
        raise ValueError("Invalid parameter.")
    if params.r_min <= 0 or params.r_max <= params.r_min:
        raise ValueError("Invalid parameter.")
    if params.K <= 0:
        raise ValueError("Invalid parameter.")
    if params.runs <= 0:
        raise ValueError("Invalid parameter.")
    if params.measure_window <= 0 or params.measure_every <= 0:
        raise ValueError("Invalid parameter.")
    if params.measure_window > params.T:
        raise ValueError("Invalid parameter.")
    if params.gpu_batch <= 0:
        raise ValueError("Invalid parameter.")
    if params.threads_per_block <= 0:
        raise ValueError("Invalid parameter.")

    probs = np.array([params.init_D, params.init_C, params.init_P], dtype=np.float64)
    if np.any(probs < 0.0) or probs.sum() <= 0.0:
        raise ValueError("Invalid parameter.")
    probs = probs / probs.sum()
    return float(probs[0]), float(probs[1]), float(probs[2])


def format_value(value: float | int) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.4g}".replace(".", "p").replace("-", "m")


def scan_tag(params: ScanParams) -> str:
    return (
        f"DoubleQ_ModelE_GPU_L{params.L}_T{params.T}"
        f"_betaF{format_value(params.beta_min)}-{format_value(params.beta_max)}x{params.beta_points}"
        f"_r{format_value(params.r_min)}-{format_value(params.r_max)}x{params.r_points}"
        f"_runs{params.runs}_seed{params.seed_start}"
        f"_measure{params.measure_window}_every{params.measure_every}"
        f"_batch{params.gpu_batch}"
    )


def estimate_batch_memory_gb(L: int, B: int) -> float:
    # docstring removed
    N = L * L
    bytes_total = 0
    #
    bytes_total += B * N * (2 + 6 + 2)  # int8-like arrays
    bytes_total += B * N * 3 * 4        # contribution_value, punishment_value, payoff
    bytes_total += B * N * 5 * 3 * 4    # q_c float32
    bytes_total += B * N * 5 * 5 * 3 * 4  # q_a float32
    bytes_total += B * N * 16           # xoroshiro RNG states roughly
    bytes_total += B * (2 + 7 + 6) * 4  # counters + metric sums + raw stats
    return bytes_total / (1024 ** 3)


def _launch_1d(kernel, total: int, threads_per_block: int, *args):
    blocks = (total + threads_per_block - 1) // threads_per_block
    kernel[blocks, threads_per_block](*args)


def run_gpu_batch(
    params: ScanParams,
    batch_tasks: list[tuple[int, int, int, float, float]],
    batch_start: int,
) -> tuple[np.ndarray, np.ndarray]:
    # docstring removed
    init_D, init_C, init_P = validate_scan_params(params)
    B = len(batch_tasks)
    L = int(params.L)
    N = L * L
    total = B * N
    tpb = int(params.threads_per_block)

    r_host = np.array([x[3] for x in batch_tasks], dtype=np.float32)
    beta_host = np.array([x[4] for x in batch_tasks], dtype=np.float32)

    #
    #
    rng_seed = int(params.seed_start + 104729 * (batch_start + 1))
    rng_states = create_xoroshiro128p_states(total, seed=rng_seed)

    z = cuda.device_array(total, dtype=np.int8)
    z_next = cuda.device_array(total, dtype=np.int8)

    q_c = cuda.device_array(total * NUM_STATES * N_C_ACTIONS, dtype=np.float32)
    q_a = cuda.device_array(total * NUM_STATES * NUM_PUNISHER_STATES * N_A_ACTIONS, dtype=np.float32)

    state_d_old = cuda.device_array(total, dtype=np.int8)
    state_p_old = cuda.device_array(total, dtype=np.int8)
    state_cp_old = cuda.device_array(total, dtype=np.int8)
    state_d_next = cuda.device_array(total, dtype=np.int8)
    state_p_next = cuda.device_array(total, dtype=np.int8)
    state_cp_next = cuda.device_array(total, dtype=np.int8)

    contribution_idx = cuda.device_array(total, dtype=np.int8)
    punishment_idx = cuda.device_array(total, dtype=np.int8)
    contribution_value = cuda.device_array(total, dtype=np.float32)
    punishment_value = cuda.device_array(total, dtype=np.float32)
    payoff = cuda.device_array(total, dtype=np.float32)

    r_dev = cuda.to_device(r_host)
    beta_dev = cuda.to_device(beta_host)
    raw_stats = cuda.device_array(B * 6, dtype=np.float32)
    metric_sums = cuda.device_array(B * 7, dtype=np.float32)
    sample_counts = cuda.device_array(B, dtype=np.int32)
    punishment_weighted_counts = cuda.device_array(B, dtype=np.float32)

    #
    _launch_1d(init_z_kernel, total, tpb, z, rng_states, L, N, float(init_D), float(init_C), total)
    _launch_1d(zero_float_kernel, q_c.size, tpb, q_c, q_c.size)
    _launch_1d(zero_float_kernel, q_a.size, tpb, q_a, q_a.size)
    _launch_1d(zero_float_kernel, metric_sums.size, tpb, metric_sums, metric_sums.size)
    _launch_1d(zero_int_kernel, sample_counts.size, tpb, sample_counts, sample_counts.size)
    _launch_1d(zero_float_kernel, punishment_weighted_counts.size, tpb, punishment_weighted_counts, punishment_weighted_counts.size)

    measure_start = max(1, int(params.T) - int(params.measure_window) + 1)
    epsilon_c = float(params.epsilon_c0)
    epsilon_a = float(params.epsilon_a0)

    for step in range(1, int(params.T) + 1):
        _launch_1d(
            states_and_actions_kernel,
            total,
            tpb,
            z,
            q_c,
            q_a,
            state_d_old,
            state_p_old,
            state_cp_old,
            contribution_idx,
            punishment_idx,
            contribution_value,
            punishment_value,
            rng_states,
            L,
            N,
            float(epsilon_c),
            float(epsilon_a),
            total,
        )

        _launch_1d(zero_float_kernel, total, tpb, payoff, total)
        _launch_1d(
            compute_payoff_kernel,
            total,
            tpb,
            z,
            contribution_value,
            punishment_value,
            payoff,
            r_dev,
            beta_dev,
            L,
            N,
            total,
        )

        if step >= measure_start and (step % int(params.measure_every) == 0 or step == int(params.T)):
            _launch_1d(zero_float_kernel, raw_stats.size, tpb, raw_stats, raw_stats.size)
            _launch_1d(
                summary_raw_kernel,
                total,
                tpb,
                z,
                contribution_value,
                punishment_value,
                payoff,
                raw_stats,
                L,
                N,
                total,
            )
            _launch_1d(
                summary_finalize_kernel,
                B,
                tpb,
                raw_stats,
                metric_sums,
                sample_counts,
                punishment_weighted_counts,
                N,
                B,
            )

        _launch_1d(strategy_update_kernel, total, tpb, z, payoff, z_next, rng_states, L, N, float(params.K), total)
        _launch_1d(neighbor_state_kernel, total, tpb, z_next, state_d_next, state_p_next, state_cp_next, L, N, total)
        _launch_1d(
            q_update_and_copy_kernel,
            total,
            tpb,
            z,
            z_next,
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
            L,
            N,
            float(params.alpha_c),
            float(params.gamma_c),
            float(params.alpha_a),
            float(params.gamma_a),
            total,
        )

        epsilon_c *= float(params.epsilon_c_dcy)
        if epsilon_c < float(params.epsilon_c_min):
            epsilon_c = float(params.epsilon_c_min)
        epsilon_a *= float(params.epsilon_a_dcy)
        if epsilon_a < float(params.epsilon_a_min):
            epsilon_a = float(params.epsilon_a_min)

    cuda.synchronize()
    sums = metric_sums.copy_to_host().reshape(B, 7).astype(np.float64)
    counts = sample_counts.copy_to_host().astype(np.float64)
    punishment_weighted_counts_host = punishment_weighted_counts.copy_to_host().astype(np.float64)

    counts[counts == 0] = 1.0

    values = np.empty_like(sums)
    values[:, :6] = sums[:, :6] / counts[:, None]

    # Mean punishment cost:
    # same as evo_cost's red dashed line: total punishment / total punishers
    # using sampled moments with D>0.
    values[:, 6] = 0.0
    valid = punishment_weighted_counts_host > 0.0
    values[valid, 6] = sums[valid, 6] / punishment_weighted_counts_host[valid]

    return values, punishment_weighted_counts_host


def warmup_gpu(params: ScanParams) -> None:
    small = replace(params, L=8, T=3, measure_window=2, measure_every=1, gpu_batch=2, beta_points=2, r_points=2, runs=1)
    _ = run_gpu_batch(small, [(0, 0, 0, 3.0, 1.5), (0, 1, 0, 3.0, 2.0)], 0)


def write_heatmap_csv(path: Path, r_values: np.ndarray, beta_values: np.ndarray, mean: np.ndarray, std: np.ndarray, sem: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["r", "beta_F"]
    for metric in METRIC_NAMES:
        fields += [f"mean_{metric}", f"std_{metric}", f"sem_{metric}"]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, r_value in enumerate(r_values):
            for j, beta_F in enumerate(beta_values):
                row: dict[str, float] = {"r": float(r_value), "beta_F": float(beta_F)}
                for k, metric in enumerate(METRIC_NAMES):
                    row[f"mean_{metric}"] = float(mean[i, j, k])
                    row[f"std_{metric}"] = float(std[i, j, k])
                    row[f"sem_{metric}"] = float(sem[i, j, k])
                writer.writerow(row)


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    })


def plot_metric_heatmap(
    output_dir: Path,
    tag: str,
    r_values: np.ndarray,
    beta_values: np.ndarray,
    matrix: np.ndarray,
    metric_name: str,
    dpi: int,
    vmin: float | None = None,
    vmax: float | None = None,
    contours: tuple[float, ...] | None = None,
) -> list[Path]:
    """Plot one publication-style beta_F-r heatmap for Model E."""
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    if metric_name == "rho_CP":
        metric_symbol = r"$\rho_{C+P}$"
        title = r"Model E: cooperation level $\rho_{C+P}$ over $\beta_F$ and $r$"
        file_key = "rhoCP"
        default_vmin, default_vmax = 0.0, 1.0
    elif metric_name == "avg_contribution_active":
        metric_symbol = r"$\bar{c}$"
        title = r"Model E: mean contribution cost $\bar{c}$ over $\beta_F$ and $r$"
        file_key = "avgContribution"
        default_vmin, default_vmax = 0.0, 1.0
    elif metric_name == "avg_punishment_punishers":
        metric_symbol = r"$\bar{a}_{D>0}$"
        title = r"Model E: global mean punishment cost $\bar{a}_{D>0}$ over $\beta_F$ and $r$"
        file_key = "avgPunishmentDPositive"
        default_vmin, default_vmax = 0.0, 0.5
    else:
        metric_symbol = METRIC_LABELS.get(metric_name, metric_name)
        title = rf"Model E: {metric_symbol} over $\beta_F$ and $r$"
        file_key = metric_name
        default_vmin, default_vmax = 0.0, 1.0

    if vmin is None:
        vmin = default_vmin
    if vmax is None:
        vmax = default_vmax

    levels = np.linspace(vmin, vmax, 101)

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")
    masked_matrix = np.ma.masked_invalid(matrix.T)

    mesh = ax.contourf(
        r_values,
        beta_values,
        masked_matrix,
        levels=levels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.10, ticks=np.linspace(vmin, vmax, 6))
    cbar.ax.set_title(metric_symbol, pad=8)

    if contours:
        cs = ax.contour(
            r_values,
            beta_values,
            matrix.T,
            levels=list(contours),
            colors="white",
            linewidths=0.9,
        )
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.2g")

    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\beta_F$")
    ax.set_title(title)
    ax.set_xlim(float(r_values.min()), float(r_values.max()))
    ax.set_ylim(float(beta_values.min()), float(beta_values.max()))
    ax.tick_params(direction="in", top=False, right=False)
    fig.tight_layout()

    base = output_dir / f"heatmap_doubleQ_ModelE_betaF_r_{file_key}_viridis_{tag}"
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = base.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths



def run_scan(params: ScanParams) -> tuple[Path, Path, list[Path]]:
    validate_scan_params(params)
    params.output_dir.mkdir(parents=True, exist_ok=True)

    if not cuda.is_available():
        raise SystemExit("CUDA GPU is required.")

    cuda.select_device(int(params.gpu_id))
    dev = cuda.get_current_device()
    print(f"GPU: {dev.name.decode() if isinstance(dev.name, bytes) else dev.name}")
    print(f"estimated main memory per batch: {estimate_batch_memory_gb(params.L, params.gpu_batch):.2f} GiB")

    if params.warmup:
        print("compiling CUDA kernels...")
        t0 = time.perf_counter()
        warmup_gpu(params)
        print(f"CUDA warmup done: {time.perf_counter() - t0:.2f}s")

    beta_values = np.linspace(params.beta_min, params.beta_max, params.beta_points, dtype=np.float64)
    r_values = np.linspace(params.r_min, params.r_max, params.r_points, dtype=np.float64)

    raw = np.full((params.r_points, params.beta_points, params.runs, len(METRIC_NAMES)), np.nan, dtype=np.float64)
    punishment_weighted_counts_raw = np.zeros((params.r_points, params.beta_points, params.runs), dtype=np.float64)

    tasks: list[tuple[int, int, int, float, float]] = []
    for i, r in enumerate(r_values):
        for j, beta in enumerate(beta_values):
            for run_idx in range(params.runs):
                tasks.append((i, j, run_idx, float(r), float(beta)))

    start = time.perf_counter()
    batches = [tasks[i:i + params.gpu_batch] for i in range(0, len(tasks), params.gpu_batch)]
    iterator = enumerate(batches)
    if params.progress and tqdm is not None:
        iterator = tqdm(iterator, total=len(batches), desc="Model E beta_F-r scan", unit="batch", dynamic_ncols=True)

    for batch_id, batch in iterator:
        t0 = time.perf_counter()
        vals, punishment_weighted_counts = run_gpu_batch(params, batch, batch_id)
        for row_idx, task in enumerate(batch):
            i, j, run_idx, _, _ = task
            raw[i, j, run_idx, :] = vals[row_idx, :]
            punishment_weighted_counts_raw[i, j, run_idx] = punishment_weighted_counts[row_idx]
        elapsed = time.perf_counter() - t0
        #
        rho_cp = vals[:, 1]
        print(
            f"batch {batch_id + 1}/{len(batches)}: "
            f"rho_CP mean={np.nanmean(rho_cp):.4f}, min={np.nanmin(rho_cp):.4f}, max={np.nanmax(rho_cp):.4f}, "
            f"elapsed={elapsed:.2f}s"
        )

    mean = np.nanmean(raw, axis=2)
    std = np.nanstd(raw, axis=2, ddof=1) if params.runs > 1 else np.zeros_like(mean)
    sem = std / math.sqrt(params.runs)

    metric_index = {name: idx for idx, name in enumerate(METRIC_NAMES)}
    punishment_idx = metric_index["avg_punishment_punishers"]
    punishment_denominator = np.sum(punishment_weighted_counts_raw, axis=2)
    punishment_numerator = np.nansum(raw[:, :, :, punishment_idx] * punishment_weighted_counts_raw, axis=2)
    punishment_mean = mean[:, :, punishment_idx]
    punishment_mean[:, :] = 0.0
    valid_punishment = punishment_denominator > 0.0
    punishment_mean[valid_punishment] = punishment_numerator[valid_punishment] / punishment_denominator[valid_punishment]

    tag = scan_tag(params)
    csv_path = params.output_dir / f"doubleQ_modelE_betaF_r_heatmap_summary_{tag}.csv"
    npz_path = params.output_dir / f"doubleQ_modelE_betaF_r_heatmap_data_{tag}.npz"
    write_heatmap_csv(csv_path, r_values, beta_values, mean, std, sem)
    np.savez_compressed(
        npz_path,
        r_values=r_values,
        beta_values=beta_values,
        raw=raw,
        mean=mean,
        std=std,
        sem=sem,
        metric_names=np.array(METRIC_NAMES),
        punishment_weighted_counts=punishment_weighted_counts_raw,
        params=np.array([str(params)], dtype=object),
    )

    figure_paths: list[Path] = []

    # 1) Cooperation level heatmap
    figure_paths += plot_metric_heatmap(
        params.output_dir,
        tag,
        r_values,
        beta_values,
        mean[:, :, metric_index["rho_CP"]],
        "rho_CP",
        params.dpi,
        vmin=0.0,
        vmax=1.0,
        contours=None,
    )

    # 2) Mean contribution cost heatmap
    figure_paths += plot_metric_heatmap(
        params.output_dir,
        tag,
        r_values,
        beta_values,
        mean[:, :, metric_index["avg_contribution_active"]],
        "avg_contribution_active",
        params.dpi,
        vmin=0.0,
        vmax=1.0,
        contours=None,
    )

    # 3) Mean punishment cost heatmap
    #    Same as evo_cost's red dashed line: total punishment / total punishers
    #    over sampled moments with D>0.
    figure_paths += plot_metric_heatmap(
        params.output_dir,
        tag,
        r_values,
        beta_values,
        mean[:, :, metric_index["avg_punishment_punishers"]],
        "avg_punishment_punishers",
        params.dpi,
        vmin=0.0,
        vmax=0.5,
        contours=None,
    )

    print(f"scan time: {time.perf_counter() - start:.2f}s")
    return csv_path, npz_path, figure_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPU heatmap for double Q-learning Model E over beta_F and r.")
    parser.add_argument("--L", type=int, default=CONFIG_SCAN.L)
    parser.add_argument("--T", type=int, default=CONFIG_SCAN.T)
    parser.add_argument("--beta-f-min", "--beta-min", dest="beta_min", type=float, default=CONFIG_SCAN.beta_min)
    parser.add_argument("--beta-f-max", "--beta-max", dest="beta_max", type=float, default=CONFIG_SCAN.beta_max)
    parser.add_argument("--beta-f-points", "--beta-points", dest="beta_points", type=int, default=CONFIG_SCAN.beta_points)
    parser.add_argument("--r-min", type=float, default=CONFIG_SCAN.r_min)
    parser.add_argument("--r-max", type=float, default=CONFIG_SCAN.r_max)
    parser.add_argument("--r-points", type=int, default=CONFIG_SCAN.r_points)
    parser.add_argument("--K", type=float, default=CONFIG_SCAN.K)
    parser.add_argument("--alpha-c", type=float, default=CONFIG_SCAN.alpha_c)
    parser.add_argument("--gamma-c", type=float, default=CONFIG_SCAN.gamma_c)
    parser.add_argument("--alpha-a", type=float, default=CONFIG_SCAN.alpha_a)
    parser.add_argument("--gamma-a", type=float, default=CONFIG_SCAN.gamma_a)
    parser.add_argument("--epsilon-c0", type=float, default=CONFIG_SCAN.epsilon_c0)
    parser.add_argument("--epsilon-c-min", type=float, default=CONFIG_SCAN.epsilon_c_min)
    parser.add_argument("--epsilon-c-dcy", type=float, default=CONFIG_SCAN.epsilon_c_dcy)
    parser.add_argument("--epsilon-a0", type=float, default=CONFIG_SCAN.epsilon_a0)
    parser.add_argument("--epsilon-a-min", type=float, default=CONFIG_SCAN.epsilon_a_min)
    parser.add_argument("--epsilon-a-dcy", type=float, default=CONFIG_SCAN.epsilon_a_dcy)
    parser.add_argument("--init-D", type=float, default=CONFIG_SCAN.init_D)
    parser.add_argument("--init-C", type=float, default=CONFIG_SCAN.init_C)
    parser.add_argument("--init-P", type=float, default=CONFIG_SCAN.init_P)
    parser.add_argument("--runs", type=int, default=CONFIG_SCAN.runs)
    parser.add_argument("--seed-start", type=int, default=CONFIG_SCAN.seed_start)
    parser.add_argument("--measure-window", type=int, default=CONFIG_SCAN.measure_window)
    parser.add_argument("--measure-every", type=int, default=CONFIG_SCAN.measure_every)
    parser.add_argument("--gpu-batch", type=int, default=CONFIG_SCAN.gpu_batch)
    parser.add_argument("--threads-per-block", type=int, default=CONFIG_SCAN.threads_per_block)
    parser.add_argument("--gpu-id", type=int, default=CONFIG_SCAN.gpu_id)
    parser.add_argument("--output-dir", type=Path, default=CONFIG_SCAN.output_dir)
    parser.add_argument("--dpi", type=int, default=CONFIG_SCAN.dpi)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    return parser


def params_from_args(args: argparse.Namespace) -> ScanParams:
    return replace(
        CONFIG_SCAN,
        L=args.L,
        T=args.T,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        beta_points=args.beta_points,
        r_min=args.r_min,
        r_max=args.r_max,
        r_points=args.r_points,
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
        runs=args.runs,
        seed_start=args.seed_start,
        measure_window=args.measure_window,
        measure_every=args.measure_every,
        gpu_batch=args.gpu_batch,
        threads_per_block=args.threads_per_block,
        gpu_id=args.gpu_id,
        output_dir=args.output_dir,
        dpi=args.dpi,
        progress=not args.no_progress,
        warmup=not args.no_warmup,
    )


def main() -> None:
    params = params_from_args(build_parser().parse_args())
    csv_path, npz_path, figure_paths = run_scan(params)
    print(f"summary csv: {csv_path}")
    print(f"data npz: {npz_path}")
    for path in figure_paths:
        print(f"figure: {path}")


if __name__ == "__main__":
    main()



