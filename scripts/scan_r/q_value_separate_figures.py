from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle


MM = 1 / 25.4
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "Q"
OUT_DIR = ROOT / "figures"
INVESTMENT_CSV = DATA_DIR / "model_F_q_r2p1_L100_T10000_runs20_all_investment_q.csv"
PUNISHMENT_CSV = DATA_DIR / "model_F_q_r2p1_L100_T10000_runs20_all_punishment_q.csv"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "dejavusans",
        "font.size": 6.2,
        "axes.linewidth": 0.55,
        "axes.labelsize": 6.3,
        "axes.titlesize": 7.0,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


COL = {
    "ink": "#202124",
    "muted": "#68707A",
    "grid": "#E7EAEE",
    "bar": "#D7DEE8",
    "best": "#0072B2",
    "investment_bar": "#F7D48A",
    "investment_best": "#F2A000",
    "investment_edge": "#4A3A16",
    "line": "#2E3033",
    "blank": "#F6F8FA",
}


def fmt_action(x):
    return "1.0" if abs(float(x) - 1.0) < 1e-9 else f"{float(x):.1f}"


def scale_state(group):
    q = group["mean_q"].to_numpy(dtype=float)
    spread = q.max() - q.min()
    if spread <= 0:
        return np.zeros_like(q)
    return (q - q.min()) / spread


def save_all(fig, out_base):
    fig.savefig(f"{out_base}.svg", bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_base}.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def style_profile_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=COL["grid"], linewidth=0.45, zorder=0)
    ax.tick_params(length=2.2, width=0.5, color=COL["line"], pad=1.5)


def investment_summary(inv):
    rows = []
    for state, group in inv.groupby("state_cp", sort=True):
        ranked = group.sort_values("mean_q", ascending=False).reset_index(drop=True)
        rows.append(
            {
                "state_cp": int(state),
                "best_action": float(ranked.loc[0, "action_value"]),
                "max_q": float(ranked.loc[0, "mean_q"]),
                "gap": float(ranked.loc[0, "mean_q"] - ranked.loc[1, "mean_q"]),
            }
        )
    return pd.DataFrame(rows)


def punishment_summary(pun):
    rows = []
    for (state_d, state_p), group in pun.groupby(["state_d", "state_p"], sort=True):
        ranked = group.sort_values("mean_q", ascending=False).reset_index(drop=True)
        rows.append(
            {
                "state_d": int(state_d),
                "state_p": int(state_p),
                "best_action": float(ranked.loc[0, "action_value"]),
                "max_q": float(ranked.loc[0, "mean_q"]),
                "gap": float(ranked.loc[0, "mean_q"] - ranked.loc[1, "mean_q"]),
            }
        )
    return pd.DataFrame(rows)


def draw_investment_profiles(inv):
    summary = investment_summary(inv).set_index("state_cp")
    actions = sorted(inv["action_value"].unique())
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(118 * MM, 92 * MM),
        dpi=300,
        sharex=True,
        gridspec_kw={"hspace": 0.13, "left": 0.19, "right": 0.80, "bottom": 0.12, "top": 0.86},
    )


    for i, state in enumerate(sorted(inv["state_cp"].unique())):
        ax = axes[i]
        group = inv[inv["state_cp"] == state].sort_values("action_value").copy()
        group["scaled_q"] = scale_state(group)
        best = summary.loc[state, "best_action"]
        colors = [
            COL["investment_best"] if np.isclose(v, best) else COL["investment_bar"]
            for v in group["action_value"]
        ]
        ax.bar(
            group["action_value"],
            group["scaled_q"],
            width=0.075,
            color=colors,
            edgecolor=COL["investment_edge"],
            linewidth=0.38,
        )
        ax.set_ylim(0, 1.08)
        ax.set_xlim(0.04, 1.06)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["0", "1"] if i == 2 else [])
        ax.text(
            -0.13,
            0.52,
            rf"$s_{{CP}}={int(state)}$",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=6.0,
        )
        ax.text(
            1.02,
            0.88,
            rf"$\hat c={fmt_action(best)}$" + "\n" + rf"$Q_{{max}}={summary.loc[state, 'max_q']:.2f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.35,
            clip_on=False,
        )
        style_profile_axis(ax)

    axes[-1].set_xticks(actions)
    axes[-1].set_xticklabels([fmt_action(v) for v in actions])
    axes[-1].set_xlabel("Investment action, $c$")
    fig.text(0.035, 0.50, "scaled mean Q", rotation=90, ha="center", va="center", fontsize=6.3)
    save_all(fig, OUT_DIR / "model_F_investment_q_profiles")


def draw_punishment_bar_facets(pun):
    summary = punishment_summary(pun).set_index(["state_d", "state_p"])
    action_labels = [fmt_action(v) for v in sorted(pun["action_value"].unique())]
    fig = plt.figure(figsize=(165 * MM, 118 * MM), dpi=300)
    gs = fig.add_gridspec(5, 5, left=0.09, right=0.94, bottom=0.14, top=0.84, wspace=0.18, hspace=0.22)

    fig.text(0.09, 0.955, "Punishment Q profiles across local states", fontsize=7.4, fontweight="bold", ha="left")
    fig.text(
        0.09,
        0.925,
        "Each small panel is one valid state $(n_D,n_P)$; bars are scaled within state and the best action is highlighted.",
        fontsize=5.4,
        color=COL["muted"],
        ha="left",
    )

    for row_index, state_d in enumerate([4, 3, 2, 1, 0]):
        for state_p in range(5):
            ax = fig.add_subplot(gs[row_index, state_p])
            if state_d + state_p > 4:
                ax.set_facecolor(COL["blank"])
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                continue

            group = pun[(pun["state_d"] == state_d) & (pun["state_p"] == state_p)].sort_values("action_value").copy()
            group["scaled_q"] = scale_state(group)
            best = summary.loc[(state_d, state_p), "best_action"]
            colors = [COL["best"] if np.isclose(v, best) else COL["bar"] for v in group["action_value"]]
            ax.bar(np.arange(len(group)), group["scaled_q"], width=0.72, color=colors, edgecolor="white", linewidth=0.3)
            ax.set_ylim(0, 1.08)
            ax.set_xlim(-0.55, len(group) - 0.45)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["0", "1"] if state_p == 0 and state_d == 2 else [])
            ax.set_xticks(np.arange(len(group)))
            if state_d == 0:
                ax.set_xticklabels(action_labels, rotation=0)
            else:
                ax.set_xticklabels([])
            ax.text(
                0.04,
                0.90,
                rf"$\hat a={best:.1f}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=4.8,
                color=COL["ink"],
            )
            ax.text(
                0.04,
                0.72,
                rf"$Q_{{max}}={summary.loc[(state_d, state_p), 'max_q']:.2f}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=4.35,
                color=COL["muted"],
            )
            style_profile_axis(ax)
            if row_index == 0:
                ax.set_title(rf"$n_P={state_p}$", fontsize=5.4, pad=3)
            if state_p == 0:
                ax.text(
                    -0.46,
                    0.52,
                    rf"$n_D={state_d}$",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=5.6,
                )

    fig.text(0.515, 0.07, "Punishment action, $a$", ha="center", va="center", fontsize=6.3)
    fig.text(0.035, 0.49, "scaled mean Q", rotation=90, ha="center", va="center", fontsize=6.3)
    save_all(fig, OUT_DIR / "model_F_punishment_q_bar_facets")


def draw_punishment_policy_map(pun):
    summary = punishment_summary(pun)
    fig, ax = plt.subplots(figsize=(105 * MM, 94 * MM), dpi=300)
    fig.subplots_adjust(left=0.16, right=0.86, bottom=0.14, top=0.85)
    cmap = LinearSegmentedColormap.from_list(
        "punishment_action", ["#EFF7F4", "#CBE7DD", "#84C9BC", "#3690C0", "#165A8A"]
    )
    norm = Normalize(vmin=0.1, vmax=0.5)

    ax.set_aspect("equal")
    ax.set_xlim(-0.04, 5.12)
    ax.set_ylim(-0.04, 5.12)
    for _, row in summary.iterrows():
        x = row["state_p"]
        y = row["state_d"]
        action = row["best_action"]
        q_max = row["max_q"]
        face = cmap(norm(action))
        ax.add_patch(Rectangle((x, y), 0.92, 0.92, facecolor=face, edgecolor="white", linewidth=0.9))
        txt = "white" if action >= 0.4 else COL["ink"]
        ax.text(x + 0.46, y + 0.56, rf"$\hat a={action:.1f}$", ha="center", va="center", fontsize=5.2, color=txt)
        ax.text(x + 0.46, y + 0.30, rf"$Q_{{max}}={q_max:.2f}$", ha="center", va="center", fontsize=4.6, color=txt)

    for x in range(5):
        for y in range(5):
            if x + y > 4:
                ax.add_patch(Rectangle((x, y), 0.92, 0.92, facecolor=COL["blank"], edgecolor="white", linewidth=0.9))

    ax.set_xticks(np.arange(5) + 0.46)
    ax.set_yticks(np.arange(5) + 0.46)
    ax.set_xticklabels([str(i) for i in range(5)])
    ax.set_yticklabels([str(i) for i in range(5)])
    ax.set_xlabel("Punisher neighbours, $n_P$")
    ax.set_ylabel("Defector neighbours, $n_D$")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.16, 0.955, "Punishment Q policy map", fontsize=7.4, fontweight="bold", ha="left")
    fig.text(0.16, 0.925, "Selected punishment action and raw maximum mean Q for each valid state.", fontsize=5.4, color=COL["muted"])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("selected $a$", labelpad=2)
    cbar.set_ticks([0.1, 0.3, 0.5])
    cbar.ax.tick_params(labelsize=5.2, length=2.2, width=0.5)
    save_all(fig, OUT_DIR / "model_F_punishment_q_policy_map")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inv = pd.read_csv(INVESTMENT_CSV)
    pun = pd.read_csv(PUNISHMENT_CSV)
    draw_investment_profiles(inv)
    draw_punishment_bar_facets(pun)
    draw_punishment_policy_map(pun)


if __name__ == "__main__":
    main()
