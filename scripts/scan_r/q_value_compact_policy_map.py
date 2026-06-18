from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import RegularPolygon


MM = 1 / 25.4
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "Q"
OUT_DIR = ROOT / "figures"
PUNISHMENT_CSV = DATA_DIR / "model_F_q_r2p1_L100_T10000_runs20_all_punishment_q.csv"
OUT_BASE = OUT_DIR / "model_F_punishment_q_policy_simplex"


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
    "axis": "#4B5563",
}


def punishment_summary(pun):
    rows = []
    for (state_d, state_p), group in pun.groupby(["state_d", "state_p"], sort=True):
        if state_d + state_p > 4:
            continue
        ranked = group.sort_values("mean_q", ascending=False).reset_index(drop=True)
        q_max = ranked.loc[0, "mean_q"]
        q_second = ranked.loc[1, "mean_q"]
        rows.append(
            {
                "state_d": int(state_d),
                "state_p": int(state_p),
                "preferred_a": float(ranked.loc[0, "action_value"]),
                "max_mean_q": float(q_max),
                "q_gap": float(q_max - q_second),
                "min_mean_q": float(group["mean_q"].min()),
            }
        )
    return pd.DataFrame(rows)


def save_all(fig, out_base):
    fig.savefig(f"{out_base}.svg", bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    fig.savefig(
        f"{out_base}.tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_policy_simplex(summary):
    cmap = LinearSegmentedColormap.from_list(
        "punishment_action",
        ["#EFF7F4", "#CBE7DD", "#84C9BC", "#3690C0", "#165A8A"],
    )
    norm = Normalize(vmin=0.1, vmax=0.5)
    row_step = np.sqrt(3) / 2

    fig, ax = plt.subplots(figsize=(103 * MM, 84 * MM), dpi=300)
    fig.subplots_adjust(left=0.10, right=0.83, bottom=0.16, top=0.84)
    ax.set_aspect("equal")
    ax.axis("off")

    for _, row in summary.iterrows():
        state_d = int(row["state_d"])
        state_p = int(row["state_p"])
        x = state_p + 0.5 * state_d
        y = state_d * row_step
        action = float(row["preferred_a"])
        q_max = float(row["max_mean_q"])
        face = cmap(norm(action))

        tile = RegularPolygon(
            (x, y),
            numVertices=6,
            radius=0.53,
            orientation=np.pi / 6,
            facecolor=face,
            edgecolor="white",
            linewidth=1.15,
            joinstyle="round",
        )
        ax.add_patch(tile)

        text_color = "white" if action >= 0.4 else COL["ink"]
        ax.text(
            x,
            y + 0.12,
            rf"$\hat a={action:.1f}$",
            ha="center",
            va="center",
            fontsize=5.45,
            color=text_color,
        )
        ax.text(
            x,
            y - 0.14,
            rf"$Q_{{max}}={q_max:.2f}$",
            ha="center",
            va="center",
            fontsize=4.75,
            color=text_color,
        )

    # Edge labels keep the state-space meaning without drawing invalid cells.
    for state_p in range(5):
        ax.text(
            state_p,
            -0.66,
            str(state_p),
            ha="center",
            va="center",
            fontsize=5.8,
            color=COL["axis"],
        )
    ax.text(
        2.0,
        -1.02,
        "Punisher neighbours, $n_P$",
        ha="center",
        va="center",
        fontsize=6.3,
        color=COL["ink"],
    )

    for state_d in range(5):
        x = 0.5 * state_d - 0.68
        y = state_d * row_step
        ax.text(
            x,
            y,
            str(state_d),
            ha="center",
            va="center",
            fontsize=5.8,
            color=COL["axis"],
        )
    ax.text(
        -0.92,
        1.74,
        "Defector neighbours, $n_D$",
        ha="center",
        va="center",
        rotation=60,
        fontsize=6.3,
        color=COL["ink"],
    )

    ax.set_xlim(-1.05, 4.78)
    ax.set_ylim(-1.10, 3.88)

    fig.text(
        0.10,
        0.955,
        "Punishment Q policy map",
        ha="left",
        va="top",
        fontsize=7.4,
        fontweight="bold",
        color=COL["ink"],
    )
    fig.text(
        0.10,
        0.920,
        r"Only feasible states are shown: $n_D+n_P\leq4$; each tile reports selected action and raw maximum mean Q.",
        ha="left",
        va="top",
        fontsize=5.35,
        color=COL["muted"],
    )

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.060, pad=0.035, shrink=0.80)
    cbar.set_label("selected $a$", labelpad=2)
    cbar.set_ticks([0.1, 0.3, 0.5])
    cbar.ax.tick_params(labelsize=5.2, length=2.2, width=0.5)

    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pun = pd.read_csv(PUNISHMENT_CSV)
    summary = punishment_summary(pun)
    summary.to_csv(OUT_DIR / "model_F_punishment_q_policy_simplex_summary.csv", index=False)
    fig = draw_policy_simplex(summary)
    save_all(fig, OUT_BASE)


if __name__ == "__main__":
    main()
