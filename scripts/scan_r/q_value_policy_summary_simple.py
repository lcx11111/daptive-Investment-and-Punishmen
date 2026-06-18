from __future__ import annotations

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
OUT_BASE = OUT_DIR / "model_F_q_policy_selected_action_summary"
INV_OUT_BASE = OUT_DIR / "model_F_investment_policy_selected_action_map"
PUN_OUT_BASE = OUT_DIR / "model_F_punishment_policy_selected_action_map"
BAR_OUT_BASE = OUT_DIR / "model_F_q_selected_action_bars"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "dejavusans",
        "font.size": 6.2,
        "axes.labelsize": 6.4,
        "axes.titlesize": 7.0,
        "xtick.labelsize": 5.8,
        "ytick.labelsize": 5.8,
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
    "blank": "#F7F8FA",
    "edge": "white",
}


def fmt_action(value: float) -> str:
    return "1.0" if np.isclose(float(value), 1.0) else f"{float(value):.1f}"


def best_action_by_state(
    df: pd.DataFrame,
    state_columns: list[str],
    action_name: str,
) -> pd.DataFrame:
    rows = []
    for state, group in df.groupby(state_columns, sort=True):
        ranked = group.sort_values("mean_q", ascending=False).reset_index(drop=True)
        state_values = state if isinstance(state, tuple) else (state,)
        row = {column: int(value) for column, value in zip(state_columns, state_values)}
        row[action_name] = float(ranked.loc[0, "action_value"])
        row["max_mean_q"] = float(ranked.loc[0, "mean_q"])
        if len(ranked) > 1:
            row["q_gap"] = float(ranked.loc[0, "mean_q"] - ranked.loc[1, "mean_q"])
        else:
            row["q_gap"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def save_all(fig: plt.Figure, out_base: Path) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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


def draw_investment_policy(ax: plt.Axes, summary: pd.DataFrame) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "investment_action",
        ["#FDF6E3", "#F7D48A", "#F2A000", "#C57400"],
    )
    norm = Normalize(vmin=0.1, vmax=1.0)
    summary = summary.sort_values("state_cp").reset_index(drop=True)

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Investment policy", loc="left", pad=4, fontweight="bold")

    for idx, row in summary.iterrows():
        state_cp = int(row["state_cp"])
        action = float(row["preferred_c"])
        q_max = float(row["max_mean_q"])
        face = cmap(norm(action))

        ax.add_patch(
            Rectangle(
                (idx + 0.03, 0.18),
                0.94,
                0.62,
                facecolor=face,
                edgecolor=COL["edge"],
                linewidth=1.0,
            )
        )
        ax.text(
            idx + 0.50,
            0.68,
            rf"$s_{{CP}}={state_cp}$",
            ha="center",
            va="center",
            fontsize=5.9,
            color=COL["ink"],
        )
        ax.text(
            idx + 0.50,
            0.48,
            rf"$\hat c={fmt_action(action)}$",
            ha="center",
            va="center",
            fontsize=6.2,
            color=COL["ink"],
        )
        ax.text(
            idx + 0.50,
            0.30,
            rf"$Q_{{max}}={q_max:.2f}$",
            ha="center",
            va="center",
            fontsize=5.35,
            color=COL["ink"],
        )

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.050, pad=0.025)
    cbar.set_label(r"selected $\hat c$", labelpad=2)
    cbar.set_ticks([0.1, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=5.2, length=2.2, width=0.5)


def draw_punishment_policy(ax: plt.Axes, summary: pd.DataFrame) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "punishment_action",
        ["#EFF7F4", "#CBE7DD", "#84C9BC", "#3690C0", "#165A8A"],
    )
    norm = Normalize(vmin=0.1, vmax=0.5)
    summary = summary[summary["state_d"] + summary["state_p"] <= 4].copy()

    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 5.10)
    ax.set_ylim(-0.05, 5.10)
    ax.set_title("Punishment policy", loc="left", pad=4, fontweight="bold")

    for x in range(5):
        for y in range(5):
            if x + y > 4:
                ax.add_patch(
                    Rectangle(
                        (x, y),
                        0.92,
                        0.92,
                        facecolor=COL["blank"],
                        edgecolor=COL["edge"],
                        linewidth=0.9,
                    )
                )

    for _, row in summary.iterrows():
        state_d = int(row["state_d"])
        state_p = int(row["state_p"])
        action = float(row["preferred_a"])
        q_max = float(row["max_mean_q"])
        face = cmap(norm(action))

        ax.add_patch(
            Rectangle(
                (state_p, state_d),
                0.92,
                0.92,
                facecolor=face,
                edgecolor=COL["edge"],
                linewidth=0.9,
            )
        )
        text_color = "white" if action >= 0.4 else COL["ink"]
        ax.text(
            state_p + 0.46,
            state_d + 0.57,
            rf"$\hat a={action:.1f}$",
            ha="center",
            va="center",
            fontsize=5.35,
            color=text_color,
        )
        ax.text(
            state_p + 0.46,
            state_d + 0.31,
            rf"$Q_{{max}}={q_max:.2f}$",
            ha="center",
            va="center",
            fontsize=4.65,
            color=text_color,
        )

    ax.set_xticks(np.arange(5) + 0.46)
    ax.set_yticks(np.arange(5) + 0.46)
    ax.set_xticklabels([str(i) for i in range(5)])
    ax.set_yticklabels([str(i) for i in range(5)])
    ax.set_xlabel(r"Punisher neighbours, $n_P$")
    ax.set_ylabel(r"Defector neighbours, $n_D$")
    ax.tick_params(length=0, colors=COL["axis"])
    for spine in ax.spines.values():
        spine.set_visible(False)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.018)
    cbar.set_label(r"selected $\hat a$", labelpad=2)
    cbar.set_ticks([0.1, 0.3, 0.5])
    cbar.ax.tick_params(labelsize=5.2, length=2.2, width=0.5)


def add_state_table(
    ax: plt.Axes,
    rows: list[list[str]],
    row_labels: list[str],
    *,
    bbox: list[float],
) -> None:
    table = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        cellLoc="center",
        rowLoc="center",
        loc="bottom",
        bbox=bbox,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.6)
    for cell in table.get_celld().values():
        cell.set_edgecolor("#6A6A6A")
        cell.set_linewidth(0.55)
        cell.set_facecolor("white")


def style_selected_action_axis(ax: plt.Axes, y_max: float, y_ticks: list[float]) -> None:
    ax.set_ylim(0, y_max)
    ax.set_yticks(y_ticks)
    ax.grid(axis="y", color="#E7EAEE", linewidth=0.55, zorder=0)
    ax.tick_params(axis="both", direction="out", length=2.4, width=0.55, pad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)


def annotate_action_bars(ax: plt.Axes, x: np.ndarray, values: np.ndarray, y_offset: float) -> None:
    for xpos, value in zip(x, values):
        ax.text(
            xpos,
            float(value) + y_offset,
            fmt_action(float(value)),
            ha="center",
            va="bottom",
            fontsize=5.6,
            color=COL["ink"],
        )


def draw_investment_selected_action_bars(ax: plt.Axes, summary: pd.DataFrame) -> None:
    plot_df = summary.copy()
    plot_df["state_d"] = 4 - plot_df["state_cp"].astype(int)
    plot_df = plot_df.sort_values("state_d").reset_index(drop=True)

    x = np.arange(len(plot_df))
    values = plot_df["preferred_c"].to_numpy(dtype=float)
    colors = ["#F2A000" if value >= values.max() - 1e-12 else "#F7D48A" for value in values]

    ax.bar(
        x,
        values,
        width=0.68,
        color=colors,
        edgecolor="#4A3A16",
        linewidth=0.55,
        zorder=3,
    )
    annotate_action_bars(ax, x, values, 0.025)
    style_selected_action_axis(ax, 1.12, [0, 0.5, 1.0])
    ax.set_xlim(-0.60, len(plot_df) - 0.40)
    ax.set_xticks([])
    ax.set_ylabel(r"Selected investment action, $\hat c$")
    ax.set_title("(a)", fontsize=7.4, pad=4)

    add_state_table(
        ax,
        [[str(int(value)) for value in plot_df["state_d"]]],
        [r"$n_D$"],
        bbox=[0.0, -0.26, 1.0, 0.18],
    )


def draw_punishment_selected_action_bars(ax: plt.Axes, summary: pd.DataFrame) -> None:
    plot_df = summary[summary["state_d"] + summary["state_p"] <= 4].copy()
    plot_df = plot_df.sort_values(["state_d", "state_p"]).reset_index(drop=True)

    x = np.arange(len(plot_df))
    values = plot_df["preferred_a"].to_numpy(dtype=float)
    colors = ["#2E86C1" if value >= values.max() - 1e-12 else "#A7CFE8" for value in values]

    ax.bar(
        x,
        values,
        width=0.68,
        color=colors,
        edgecolor="#1F4E66",
        linewidth=0.52,
        zorder=3,
    )
    annotate_action_bars(ax, x, values, 0.012)
    style_selected_action_axis(ax, 0.58, [0, 0.25, 0.5])
    ax.set_xlim(-0.60, len(plot_df) - 0.40)
    ax.set_xticks([])
    ax.set_ylabel(r"Selected punishment action, $\hat a$")
    ax.set_title("(b)", fontsize=7.4, pad=4)

    for boundary in np.cumsum(plot_df.groupby("state_d", sort=True).size().to_numpy())[:-1]:
        ax.axvline(boundary - 0.5, color="#8A8A8A", linewidth=0.55, zorder=2)

    add_state_table(
        ax,
        [
            [str(int(value)) for value in plot_df["state_p"]],
            [str(int(value)) for value in plot_df["state_d"]],
        ],
        [r"$n_P$", r"$n_D$"],
        bbox=[0.0, -0.34, 1.0, 0.26],
    )


def draw_selected_action_bar_summary(inv_summary: pd.DataFrame, pun_summary: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(178 * MM, 64 * MM),
        dpi=300,
        gridspec_kw={"width_ratios": [0.85, 1.75]},
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.28, top=0.88, wspace=0.30)

    draw_investment_selected_action_bars(axes[0], inv_summary)
    draw_punishment_selected_action_bars(axes[1], pun_summary)
    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inv = pd.read_csv(INVESTMENT_CSV)
    pun = pd.read_csv(PUNISHMENT_CSV)

    inv_summary = best_action_by_state(inv, ["state_cp"], "preferred_c")
    pun_summary = best_action_by_state(pun, ["state_d", "state_p"], "preferred_a")
    inv_summary.to_csv(OUT_DIR / "model_F_investment_policy_selected_actions.csv", index=False)
    pun_summary.to_csv(OUT_DIR / "model_F_punishment_policy_selected_actions.csv", index=False)

    fig_inv, ax_inv_only = plt.subplots(figsize=(145 * MM, 32 * MM), dpi=300)
    fig_inv.subplots_adjust(left=0.035, right=0.955, bottom=0.12, top=0.80)
    draw_investment_policy(ax_inv_only, inv_summary)
    save_all(fig_inv, INV_OUT_BASE)

    fig_pun, ax_pun_only = plt.subplots(figsize=(92 * MM, 78 * MM), dpi=300)
    fig_pun.subplots_adjust(left=0.17, right=0.86, bottom=0.14, top=0.86)
    draw_punishment_policy(ax_pun_only, pun_summary)
    save_all(fig_pun, PUN_OUT_BASE)

    fig = plt.figure(figsize=(178 * MM, 118 * MM), dpi=300)
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.55, 1.45],
        left=0.06,
        right=0.96,
        bottom=0.10,
        top=0.91,
        hspace=0.34,
    )

    ax_inv = fig.add_subplot(gs[0, 0])
    draw_investment_policy(ax_inv, inv_summary)

    ax_pun = fig.add_subplot(gs[1, 0])
    draw_punishment_policy(ax_pun, pun_summary)

    save_all(fig, OUT_BASE)
    fig_bar = draw_selected_action_bar_summary(inv_summary, pun_summary)
    save_all(fig_bar, BAR_OUT_BASE)
    print("Saved simplified Q-policy summary:")
    for base in [OUT_BASE, INV_OUT_BASE, PUN_OUT_BASE, BAR_OUT_BASE]:
        for suffix in [".svg", ".pdf", ".png", ".tiff"]:
            print(f"{base}{suffix}")
    print(OUT_DIR / "model_F_investment_policy_selected_actions.csv")
    print(OUT_DIR / "model_F_punishment_policy_selected_actions.csv")


if __name__ == "__main__":
    main()
