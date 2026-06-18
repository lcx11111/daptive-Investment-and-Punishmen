from __future__ import annotations

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

# ----------------------------
# Inlined evolution support from scripts/evolution/evo.py
# ----------------------------
def format_value(value: float | int) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.4g}".replace(".", "p").replace("-", "m")

def parse_float_list(text: str) -> np.ndarray:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Invalid parameter.")
    return np.array(values, dtype=np.float64)

@dataclass(frozen=True)
class EvolutionParams:
    L: int = 100
    T: int = 10000
    r: float =3.0
    seed: int = 1449
    seed_start: int = 1449
    runs: int = 20

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

def validate_params(params: EvolutionParams) -> tuple[float, float, float]:
    if params.L <= 0 or params.T <= 0:
        raise ValueError("Invalid parameter.")
    if params.runs <= 0:
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

def save_one_figure(fig: plt.Figure, basename: Path, dpi: int) -> list[Path]:
    """Save one figure in publication-friendly raster/vector formats."""
    figure_paths: list[Path] = []
    for suffix, save_dpi in ((".pdf", dpi), (".svg", dpi), (".png", dpi), (".tiff", dpi)):
        path = basename.with_suffix(suffix)
        fig.savefig(path, dpi=save_dpi, facecolor="white")
        figure_paths.append(path)
    return figure_paths

# ============================================================
# Model E action-cost evolution from evo.py.
#
# The curve records each realized round after action selection:
#   mean investment cost: average c_i among C and P players;
#   mean punishment cost: average a_i among P players.
#
# The annotated global means are weighted over relevant players,
# not a simple average of per-round means. The punishment global
# mean only uses rounds where at least one defector exists.
# ============================================================


CONFIG = replace(EvolutionParams(), output_dir=Path("data/evolution_cost"))


@njit(cache=True)
def record_cost_metrics(
    z: np.ndarray,
    contribution_value: np.ndarray,
    punishment_value: np.ndarray,
    metrics: np.ndarray,
    row: int,
) -> None:
    total_contribution = 0.0
    total_punishment = 0.0
    n_investors = 0
    n_punishers = 0
    n_defectors = 0

    L = z.shape[0]
    for i in range(L):
        for j in range(L):
            strategy = z[i, j]
            if strategy == D:
                n_defectors += 1
            elif strategy == C or strategy == P:
                total_contribution += contribution_value[i, j]
                n_investors += 1
                if strategy == P:
                    total_punishment += punishment_value[i, j]
                    n_punishers += 1

    if n_investors > 0:
        metrics[row, 0] = total_contribution / n_investors
    else:
        metrics[row, 0] = 0.0

    if n_punishers > 0:
        metrics[row, 1] = total_punishment / n_punishers
    else:
        metrics[row, 1] = 0.0

    metrics[row, 2] = total_contribution
    metrics[row, 3] = n_investors
    metrics[row, 4] = total_punishment
    metrics[row, 5] = n_punishers
    metrics[row, 6] = n_defectors


@njit(cache=True)
def simulate_cost_evolution_numba(
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
) -> np.ndarray:
    """Return cost metrics for rounds t=1,...,T.

    Columns:
      0 mean investment cost among C/P players
      1 mean punishment cost among P players
      2 total investment cost among C/P players
      3 number of C/P players
      4 total punishment cost among P players
      5 number of P players
      6 number of D players
    """
    np.random.seed(seed)

    metrics = np.zeros((T, 7), dtype=np.float64)

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

    for row in range(T):
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
        record_cost_metrics(z, contribution_value, punishment_value, metrics, row)
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

    return metrics


def warmup_cost_numba() -> None:
    simulate_cost_evolution_numba(
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
        np.array([0.1, 0.2, 0.3], dtype=np.float64),
        np.array([0.1, 0.2], dtype=np.float64),
    )


def action_tag(actions: np.ndarray) -> str:
    return "-".join(format_value(float(value)) for value in actions)


def build_tag(params: EvolutionParams, contribution_actions: np.ndarray, punishment_actions: np.ndarray) -> str:
    return (
        f"modelE_cost_evolution_logt_L{params.L}_T{params.T}_r{format_value(params.r)}"
        f"_runs{params.runs}_seed{params.seed_start}"
        f"_bF{format_value(params.beta_F)}_K{format_value(params.K)}"
        f"_aC{format_value(params.alpha_c)}_gC{format_value(params.gamma_c)}"
        f"_aA{format_value(params.alpha_a)}_gA{format_value(params.gamma_a)}"
        f"_eC{format_value(params.epsilon_c0)}-{format_value(params.epsilon_c_min)}-{format_value(params.epsilon_c_dcy)}"
        f"_eP{format_value(params.epsilon_a0)}-{format_value(params.epsilon_a_min)}-{format_value(params.epsilon_a_dcy)}"
        f"_Ac{action_tag(contribution_actions)}_Aa{action_tag(punishment_actions)}"
    )


def compute_global_means(metrics: np.ndarray) -> tuple[float, float]:
    if metrics.ndim == 3:
        total_contribution = float(metrics[:, :, 2].sum())
        n_investors = float(metrics[:, :, 3].sum())
        rows_with_defectors = metrics[:, :, 6] > 0.0
        total_punishment = float(metrics[:, :, 4][rows_with_defectors].sum())
        n_punishers = float(metrics[:, :, 5][rows_with_defectors].sum())
    else:
        total_contribution = float(metrics[:, 2].sum())
        n_investors = float(metrics[:, 3].sum())
        rows_with_defectors = metrics[:, 6] > 0.0
        total_punishment = float(metrics[rows_with_defectors, 4].sum())
        n_punishers = float(metrics[rows_with_defectors, 5].sum())

    global_contribution = total_contribution / n_investors if n_investors > 0 else 0.0
    global_punishment = total_punishment / n_punishers if n_punishers > 0 else 0.0
    return global_contribution, global_punishment


def save_cost_outputs(
    params: EvolutionParams,
    metrics_mean: np.ndarray,
    metrics_std: np.ndarray,
    metrics_sem: np.ndarray,
    metrics_all: np.ndarray,
    contribution_actions: np.ndarray,
    punishment_actions: np.ndarray,
) -> tuple[Path, list[Path], float, float]:
    params.output_dir.mkdir(parents=True, exist_ok=True)
    tag = build_tag(params, contribution_actions, punishment_actions)
    global_contribution, global_punishment = compute_global_means(metrics_all)

    t = np.arange(1, params.T + 1, dtype=np.int64)
    csv_path = params.output_dir / f"{tag}.csv"
    header = (
        "t,mean_investment_cost,mean_punishment_cost,total_investment_cost,"
        "n_investors,total_punishment_cost,n_punishers,n_defectors,"
        "std_mean_investment_cost,std_mean_punishment_cost,"
        "sem_mean_investment_cost,sem_mean_punishment_cost"
    )
    csv_data = np.column_stack(
        (
            t,
            metrics_mean[:, 0],
            metrics_mean[:, 1],
            metrics_mean[:, 2],
            metrics_mean[:, 3],
            metrics_mean[:, 4],
            metrics_mean[:, 5],
            metrics_mean[:, 6],
            metrics_std[:, 0],
            metrics_std[:, 1],
            metrics_sem[:, 0],
            metrics_sem[:, 1],
        )
    )
    np.savetxt(
        csv_path,
        csv_data,
        delimiter=",",
        header=header,
        comments="",
        fmt=["%d", "%.10f", "%.10f", "%.10f", "%.6f", "%.10f", "%.6f", "%.6f", "%.10f", "%.10f", "%.10f", "%.10f"],
    )

    fig, ax = plt.subplots(figsize=(4.25, 3.05), constrained_layout=True)
    ax.plot(t, metrics_mean[:, 0], color="#1F77B4", linestyle="-", label=r"Mean investment cost $\langle c_i\rangle$")
    ax.plot(t, metrics_mean[:, 1], color="#D62728", linestyle="-", label=r"Mean punishment cost $\langle a_i\rangle$")
    ax.axhline(global_contribution, color="#1F77B4", linestyle="--", linewidth=1.1, alpha=0.85)
    ax.axhline(global_punishment, color="#D62728", linestyle="--", linewidth=1.1, alpha=0.85)

    max_action = max(float(contribution_actions.max()), float(punishment_actions.max()))
    ax.set_xscale("log")
    ax.set_xlim(1, params.T)
    ax.set_ylim(-0.03, max_action + 0.08)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Average cost")
    ax.set_title(rf"Model E, $r={params.r:g}$", fontsize=11, pad=4)
    ax.legend(loc="best", handlelength=2.2)
    ax.text(
        0.03,
        0.96,
        rf"Global mean: $\bar c={global_contribution:.4f}$"
        + "\n"
        + rf"Global mean ($D>0$): $\bar a={global_punishment:.4f}$",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "linewidth": 0.7, "alpha": 0.92},
    )

    figure_paths = save_one_figure(fig, params.output_dir / f"{tag}_costs", params.dpi)
    plt.close(fig)
    return csv_path, figure_paths, global_contribution, global_punishment


def summarize_cost_runs(metrics_all: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(metrics_all, axis=0)
    std = np.std(metrics_all, axis=0, ddof=1) if metrics_all.shape[0] > 1 else np.zeros_like(mean)
    sem = std / np.sqrt(metrics_all.shape[0])
    return mean, std, sem


def run_cost_evolution(params: EvolutionParams) -> tuple[Path, list[Path], float, float]:
    configure_publication_style()
    init_D, init_C, init_P = validate_params(params)
    contribution_actions = parse_float_list(params.contribution_actions)
    punishment_actions = parse_float_list(params.punishment_actions)

    if params.warmup:
        print("compiling numba functions...")
        start = time.perf_counter()
        warmup_cost_numba()
        print(f"numba warmup done: {time.perf_counter() - start:.2f}s")

    metrics_all = np.empty((int(params.runs), int(params.T), 7), dtype=np.float64)
    for run_idx in range(int(params.runs)):
        seed = int(params.seed_start) + run_idx
        print(f"running cost evolution run {run_idx + 1}/{params.runs}, seed={seed}...")
        start = time.perf_counter()
        metrics_all[run_idx] = simulate_cost_evolution_numba(
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
        print(f"run {run_idx + 1}/{params.runs} done: {time.perf_counter() - start:.2f}s")

    metrics_mean, metrics_std, metrics_sem = summarize_cost_runs(metrics_all)
    return save_cost_outputs(params, metrics_mean, metrics_std, metrics_sem, metrics_all, contribution_actions, punishment_actions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Save Model E investment/punishment cost evolution curves.")
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
    params = params_from_args(build_parser().parse_args())
    csv_path, figure_paths, global_contribution, global_punishment = run_cost_evolution(params)
    print("cost evolution data:", csv_path)
    print(f"global mean investment cost: {global_contribution:.10f}")
    print(f"global mean punishment cost: {global_punishment:.10f}")
    for path in figure_paths:
        print("cost evolution plot:", path)


if __name__ == "__main__":
    main()
