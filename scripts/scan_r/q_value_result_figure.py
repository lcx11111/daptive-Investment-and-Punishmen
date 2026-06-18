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
OUT_BASE = OUT_DIR / "model_F_q_value_policy_summary"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "dejavusans",
        "font.size": 6.2,
        "axes.linewidth": 0.55,
        "axes.labelsize": 6.4,
        "axes.titlesize": 7.0,
        "xtick.labelsize": 5.6,
        "ytick.labelsize": 5.6,
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
    "investment": "#0072B2",
    "investment_bar": "#F7D48A",
    "investment_best": "#F2A000",
    "investment_edge": "#4A3A16",
    "punish_edge": "#56616C",
    "line": "#2E3033",
    "table_head": "#F2F5F8",
    "table_alt": "#FBFCFD",
}


def fmt_action(x):
    return "1.0" if abs(float(x) - 1.0) < 1e-9 else f"{float(x):.1f}"


def compute_investment_summary(inv):
    rows = []
    for state, group in inv.groupby("state_cp", sort=True):
        ranked = group.sort_values("mean_q", ascending=False).reset_index(drop=True)
        q_max = ranked.loc[0, "mean_q"]
        q_second = ranked.loc[1, "mean_q"]
        q_min = group["mean_q"].min()
        rows.append(
            {
                "state_cp": int(state),
                "cooperation_level": int(state) / 4,
                "preferred_c": float(ranked.loc[0, "action_value"]),
                "max_mean_q": float(q_max),
                "q_gap": float(q_max - q_second),
                "min_mean_q": float(q_min),
            }
        )
    return pd.DataFrame(rows)


def compute_punishment_summary(pun):
    rows = []
    for (state_d, state_p), group in pun.groupby(["state_d", "state_p"], sort=True):
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


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.4, width=0.5, color=COL["line"], pad=1.5)
    ax.grid(axis="y", color=COL["grid"], linewidth=0.45, zorder=0)


def draw_investment_profiles(fig, spec, inv, inv_summary):
    sub = spec.subgridspec(5, 1, hspace=0.16)
    actions = sorted(inv["action_value"].unique())
    best_by_state = inv_summary.set_index("state_cp")

    axes = []
    for idx, state in enumerate(sorted(inv["state_cp"].unique())):
        ax = fig.add_subplot(sub[idx, 0])
        axes.append(ax)
        g = inv[inv["state_cp"] == state].sort_values("action_value").copy()
        q_min = g["mean_q"].min()
        q_range = g["mean_q"].max() - q_min
        g["q_scaled"] = 0 if q_range == 0 else (g["mean_q"] - q_min) / q_range
        best_c = best_by_state.loc[state, "preferred_c"]
        colors = [
            COL["investment_best"] if np.isclose(v, best_c) else COL["investment_bar"]
            for v in g["action_value"]
        ]

        ax.bar(
            g["action_value"],
            g["q_scaled"],
            width=0.075,
            color=colors,
            edgecolor=COL["investment_edge"],
            linewidth=0.38,
        )
        ax.set_ylim(0, 1.08)
        ax.set_xlim(0.04, 1.06)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["0", "1"] if idx == 2 else [])
        ax.text(
            -0.075,
            0.53,
            rf"$s_{{CP}}={int(state)}$",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=6.0,
            color=COL["ink"],
        )
        row = best_by_state.loc[state]
        ax.text(
            1.01,
            0.88,
            rf"$\hat c={fmt_action(row['preferred_c'])}$"
            + "\n"
            + rf"$Q_{{max}}={row['max_mean_q']:.2f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.35,
            color=COL["ink"],
            clip_on=False,
        )
        style_axis(ax)
        if idx < 4:
            ax.set_xticklabels([])
        else:
            ax.set_xticks(actions)
            ax.set_xticklabels([fmt_action(v) for v in actions], rotation=0)
            ax.set_xlabel("Investment action, $c$")
        if idx == 0:
            ax.set_title("Investment Q profiles by local cooperation", loc="left", pad=5, fontweight="bold")
            ax.text(
                -0.13,
                1.22,
                "a",
                transform=ax.transAxes,
                fontsize=8.0,
                fontweight="bold",
                va="bottom",
                color=COL["ink"],
            )

    axes[0].text(
        -0.08,
        1.02,
        "scaled Q",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=5.4,
        color=COL["muted"],
    )
    return axes


def draw_punishment_policy(ax, pun_summary):
    cmap = LinearSegmentedColormap.from_list(
        "punishment_action", ["#EFF7F4", "#CBE7DD", "#84C9BC", "#3690C0", "#165A8A"]
    )
    norm = Normalize(vmin=0.1, vmax=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(-0.04, 5.12)
    ax.set_ylim(-0.04, 5.12)

    for _, row in pun_summary.iterrows():
        x = row["state_p"]
        y = row["state_d"]
        action = row["preferred_a"]
        q_max = row["max_mean_q"]
        fc = cmap(norm(action))
        ax.add_patch(Rectangle((x, y), 0.92, 0.92, facecolor=fc, edgecolor="white", linewidth=0.9))
        txt_color = "white" if action >= 0.4 else COL["ink"]
        ax.text(
            x + 0.46,
            y + 0.56,
            rf"$\hat a={action:.1f}$",
            ha="center",
            va="center",
            fontsize=5.25,
            color=txt_color,
        )
        ax.text(
            x + 0.46,
            y + 0.30,
            rf"$Q_{{max}}={q_max:.2f}$",
            ha="center",
            va="center",
            fontsize=4.75,
            color=txt_color,
        )

    for x in range(5):
        for y in range(5):
            if x + y > 4:
                ax.add_patch(Rectangle((x, y), 0.92, 0.92, facecolor="#F7F8FA", edgecolor="white", linewidth=0.9))

    ax.set_xticks(np.arange(5) + 0.46)
    ax.set_yticks(np.arange(5) + 0.46)
    ax.set_xticklabels([str(i) for i in range(5)])
    ax.set_yticklabels([str(i) for i in range(5)])
    ax.set_xlabel("Punisher neighbours, $n_P$")
    ax.set_ylabel("Defector neighbours, $n_D$")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Punishment policy map", loc="left", pad=5, fontweight="bold")
    ax.text(
        -0.13,
        1.05,
        "b",
        transform=ax.transAxes,
        fontsize=8.0,
        fontweight="bold",
        va="bottom",
        color=COL["ink"],
    )
    ax.text(
        0.0,
        -0.16,
        "Each valid state shows selected action and raw maximum mean Q.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.2,
        color=COL["muted"],
    )
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("selected $a$", labelpad=2)
    cbar.set_ticks([0.1, 0.3, 0.5])
    cbar.ax.tick_params(labelsize=5.2, length=2.2, width=0.5)


def draw_table_block(ax, x0, y0, widths, rows, headers, title):
    total_w = sum(widths)
    row_h = 0.095
    header_h = 0.105
    ax.text(x0, y0 + header_h + row_h * len(rows) + 0.055, title, fontsize=6.5, fontweight="bold", va="bottom")
    ax.add_patch(Rectangle((x0, y0 + row_h * len(rows)), total_w, header_h, facecolor=COL["table_head"], edgecolor="none"))
    x = x0
    for h, w in zip(headers, widths):
        ax.text(x + 0.008, y0 + row_h * len(rows) + header_h / 2, h, va="center", ha="left", fontsize=5.6, fontweight="bold")
        x += w
    ax.plot([x0, x0 + total_w], [y0 + row_h * len(rows), y0 + row_h * len(rows)], color=COL["line"], lw=0.5)
    ax.plot([x0, x0 + total_w], [y0 + row_h * len(rows) + header_h, y0 + row_h * len(rows) + header_h], color=COL["line"], lw=0.5)

    for ridx, row in enumerate(rows):
        y = y0 + row_h * (len(rows) - 1 - ridx)
        if ridx % 2 == 1:
            ax.add_patch(Rectangle((x0, y), total_w, row_h, facecolor=COL["table_alt"], edgecolor="none"))
        x = x0
        for cell, w in zip(row, widths):
            ax.text(x + 0.008, y + row_h / 2, cell, va="center", ha="left", fontsize=5.55, color=COL["ink"])
            x += w
    ax.plot([x0, x0 + total_w], [y0, y0], color=COL["line"], lw=0.5)


def draw_summary_tables(ax, inv_summary, pun_summary):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 0.965, "c", transform=ax.transAxes, fontsize=8.0, fontweight="bold", va="top", color=COL["ink"])
    ax.text(0.035, 0.96, "Compact policy summary", transform=ax.transAxes, fontsize=7.0, fontweight="bold", va="top")

    inv_rows = []
    for _, row in inv_summary.iterrows():
        inv_rows.append(
            [
                rf"{int(row['state_cp'])}",
                f"{row['cooperation_level']:.2f}",
                fmt_action(row["preferred_c"]),
                f"{row['max_mean_q']:.2f}",
                f"{row['q_gap']:.3f}",
            ]
        )
    draw_table_block(
        ax,
        0.035,
        0.08,
        [0.06, 0.08, 0.08, 0.10, 0.08],
        inv_rows,
        [r"$s_{CP}$", "$q$", r"$\hat c$", "$Q_{max}$", "$gap$"],
        "Investment policy",
    )

    rule_rows = []
    for n_d in range(5):
        sub = pun_summary[pun_summary["state_d"] == n_d].sort_values("state_p")
        actions = sub["preferred_a"].round(1).tolist()
        q_min = sub["max_mean_q"].min()
        q_max = sub["max_mean_q"].max()
        if n_d == 0:
            state_label = r"$n_D=0$"
        elif n_d == 1:
            state_label = r"$n_D=1$"
        else:
            state_label = rf"$n_D={n_d}$"
        if len(set(actions)) == 1:
            action_label = f"{actions[0]:.1f}"
        else:
            action_label = ", ".join(f"{a:.1f}" for a in actions)
        rule_rows.append([state_label, action_label, f"{q_min:.2f}-{q_max:.2f}", f"{len(sub)} states"])
    draw_table_block(
        ax,
        0.52,
        0.08,
        [0.10, 0.15, 0.12, 0.10],
        rule_rows,
        [r"$n_D$ class", r"selected $\hat a$", r"$Q_{max}$", "states"],
        "Punishment policy",
    )
    ax.text(
        0.035,
        0.025,
        "Gap is the difference between the largest and second-largest mean Q within the same state.",
        transform=ax.transAxes,
        fontsize=5.2,
        color=COL["muted"],
        va="bottom",
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inv = pd.read_csv(INVESTMENT_CSV)
    pun = pd.read_csv(PUNISHMENT_CSV)
    inv_summary = compute_investment_summary(inv)
    pun_summary = compute_punishment_summary(pun)

    inv_summary.to_csv(OUT_DIR / "model_F_investment_q_policy_summary.csv", index=False)
    pun_summary.to_csv(OUT_DIR / "model_F_punishment_q_policy_summary.csv", index=False)

    fig = plt.figure(figsize=(183 * MM, 145 * MM), dpi=300)
    outer = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.18, 0.82],
        width_ratios=[1.32, 1.0],
        left=0.07,
        right=0.985,
        bottom=0.075,
        top=0.88,
        wspace=0.30,
        hspace=0.34,
    )
    fig.text(
        0.07,
        0.972,
        "Learned Q-value structure in the double Q-learning model",
        ha="left",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color=COL["ink"],
    )
    fig.text(
        0.07,
        0.948,
        "Model F; r=2.1, L=100, T=10000, 20 runs; all individuals",
        ha="left",
        va="top",
        fontsize=5.7,
        color=COL["muted"],
    )

    draw_investment_profiles(fig, outer[0, 0], inv, inv_summary)
    ax_b = fig.add_subplot(outer[0, 1])
    draw_punishment_policy(ax_b, pun_summary)
    ax_c = fig.add_subplot(outer[1, :])
    draw_summary_tables(ax_c, inv_summary, pun_summary)

    fig.savefig(f"{OUT_BASE}.svg", bbox_inches="tight")
    fig.savefig(f"{OUT_BASE}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_BASE}.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(f"{OUT_BASE}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
