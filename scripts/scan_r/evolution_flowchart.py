from pathlib import Path as FilePath

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MplPath


MM = 1 / 25.4
OUT_DIR = FilePath(__file__).resolve().parents[2] / "figures"
OUT_BASE = OUT_DIR / "double_q_evolution_flowchart"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 5.2,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


COLORS = {
    "ink": "#202124",
    "muted": "#60646C",
    "line": "#2E3033",
    "lane": "#FAFAFB",
    "spatial": "#F1F3F4",
    "state": "#EAF2FB",
    "action": "#EAF7F2",
    "payoff": "#FFF1DD",
    "reward": "#F3EEF9",
    "fermi": "#FDECEA",
    "update": "#FFF8E6",
    "blue": "#4C78A8",
    "teal": "#2A9D8F",
    "amber": "#E9A33A",
    "red": "#D6604D",
}


def add_box(ax, x, y, w, h, title, body, fill, edge=None, fs=5.0):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.45,rounding_size=1.6",
        linewidth=0.75,
        edgecolor=edge or COLORS["line"],
        facecolor=fill,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + 2.2,
        y + h - 3.4,
        title,
        ha="left",
        va="top",
        fontsize=5.7,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=4,
    )
    ax.text(
        x + 2.2,
        y + h - 8.4,
        body,
        ha="left",
        va="top",
        fontsize=fs,
        color=COLORS["ink"],
        linespacing=1.25,
        zorder=4,
    )


def add_arrow(ax, start, end, rad=0.0, label=None, label_xy=None, color=None):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=7.5,
        linewidth=0.75,
        color=color or COLORS["line"],
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2.5,
        shrinkB=2.5,
        zorder=3,
    )
    ax.add_patch(arrow)
    if label:
        ax.text(
            *(label_xy or ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)),
            label,
            ha="center",
            va="center",
            fontsize=4.6,
            color=COLORS["muted"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7, "alpha": 0.78},
            zorder=5,
        )


def add_bezier_arrow(ax, start, control, end, label=None, label_xy=None, color=None):
    path = MplPath([start, control, end], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
    arrow = FancyArrowPatch(
        path=path,
        arrowstyle="-|>",
        mutation_scale=7.5,
        linewidth=0.75,
        color=color or COLORS["line"],
        shrinkA=2.5,
        shrinkB=2.5,
        zorder=3,
    )
    ax.add_patch(arrow)
    if label:
        ax.text(
            *(label_xy or control),
            label,
            ha="center",
            va="center",
            fontsize=4.6,
            color=COLORS["muted"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7, "alpha": 0.78},
            zorder=5,
        )


def add_lattice(ax, x, y, cell=2.7):
    states = [
        ["C", "P", "C", "D", "C"],
        ["P", "C", "P", "D", "D"],
        ["C", "P", "i", "D", "C"],
        ["C", "C", "P", "P", "D"],
        ["D", "C", "C", "P", "C"],
    ]
    fill = {"D": COLORS["red"], "C": COLORS["teal"], "P": COLORS["amber"], "i": "#FFFFFF"}
    text = {"D": "white", "C": "white", "P": COLORS["ink"], "i": COLORS["ink"]}
    for row, values in enumerate(states):
        for col, value in enumerate(values):
            xx = x + col * cell
            yy = y + (4 - row) * cell
            lw = 0.85 if value == "i" else 0.25
            ax.add_patch(
                Rectangle(
                    (xx, yy),
                    cell,
                    cell,
                    facecolor=fill[value],
                    edgecolor=COLORS["line"],
                    linewidth=lw,
                    zorder=5,
                )
            )
            ax.text(
                xx + cell / 2,
                yy + cell / 2,
                value,
                ha="center",
                va="center",
                fontsize=4.2,
                fontweight="bold" if value == "i" else "normal",
                color=text[value],
                zorder=6,
            )


def add_spatial_box(ax, x, y, w, h):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.45,rounding_size=1.6",
        linewidth=0.75,
        edgecolor=COLORS["line"],
        facecolor=COLORS["spatial"],
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + 2.2,
        y + h - 3.4,
        "Spatial public-goods game",
        ha="left",
        va="top",
        fontsize=5.7,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=4,
    )
    add_lattice(ax, x + 2.6, y + 5.4, cell=2.55)
    ax.text(
        x + 18.2,
        y + h - 8.4,
        "$L\\times L$ periodic lattice\n"
        "$z_i(t)\\in\\{D,C,P\\}$\n"
        "Each agent joins five\n"
        "$G=5$ public-goods groups\n"
        "$Q_i^c,Q_i^a$ initialized to 0",
        ha="left",
        va="top",
        fontsize=4.8,
        linespacing=1.18,
        color=COLORS["ink"],
        zorder=4,
    )


def add_legend(ax):
    items = [
        ("D", "defector", COLORS["red"], "white"),
        ("C", "cooperator", COLORS["teal"], "white"),
        ("P", "punisher", COLORS["amber"], COLORS["ink"]),
    ]
    x0 = 9.5
    y0 = 5.0
    ax.text(x0, y0 + 1.2, "Strategy states", fontsize=4.8, color=COLORS["muted"], va="center")
    x = x0 + 24
    for symbol, label, fill, fg in items:
        ax.add_patch(
            Rectangle((x, y0), 4.2, 4.2, facecolor=fill, edgecolor=COLORS["line"], linewidth=0.35)
        )
        ax.text(x + 2.1, y0 + 2.1, symbol, ha="center", va="center", fontsize=4.0, color=fg)
        ax.text(x + 5.4, y0 + 2.1, label, ha="left", va="center", fontsize=4.8, color=COLORS["muted"])
        x += 25.5


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(183 * MM, 115 * MM), dpi=300)
    ax.set_xlim(0, 183)
    ax.set_ylim(0, 115)
    ax.axis("off")

    ax.add_patch(Rectangle((6, 64), 171, 43, facecolor=COLORS["lane"], edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((6, 10), 171, 50, facecolor=COLORS["lane"], edgecolor="none", zorder=0))
    ax.text(4.6, 86, "strategy / payoff", rotation=90, fontsize=4.8, color=COLORS["muted"], va="center")
    ax.text(4.6, 35, "behavior learning", rotation=90, fontsize=4.8, color=COLORS["muted"], va="center")

    ax.text(8, 110.5, "a", fontsize=8.0, fontweight="bold", color=COLORS["ink"], va="center")
    ax.text(
        16,
        110.5,
        "Evolutionary workflow of the double Q-learning adaptive investment-punishment model",
        fontsize=7.0,
        fontweight="bold",
        color=COLORS["ink"],
        va="center",
    )

    add_spatial_box(ax, 11, 75, 38, 28)
    add_box(
        ax,
        55,
        75,
        42,
        28,
        "Local state encoding",
        "Four nearest neighbors $\\mathcal{N}_i$\n"
        "$s_i^c(t)=n_{CP}^i(t)$\n"
        "$s_i^a(t)=(n_D^i(t),n_P^i(t))$\n"
        "$d_i(t)=n_D^i(t)/4$",
        COLORS["state"],
        COLORS["blue"],
        fs=4.9,
    )
    add_box(
        ax,
        104,
        75,
        43,
        28,
        "Payoff aggregation",
        "$C^g=\\sum_{j\\in g,z_j\\in\\{C,P\\}} c_j$\n"
        "$F^g=n_P^g[\\exp(\\beta_F\\bar a^g)-1]$\n"
        "$\\pi_{i,g}$ subtracts fine/costs\n"
        "$\\Pi_i=\\sum_{g\\in\\Omega_i}\\pi_{i,g}$",
        COLORS["payoff"],
        "#D99425",
        fs=4.55,
    )
    add_box(
        ax,
        153,
        72,
        27,
        31,
        "Fermi update",
        "Random neighbor $j$\n"
        "$W=[1+e^{-(\\Pi_j-\\Pi_i)/K}]^{-1}$\n"
        "$z_i(t+1)\\leftarrow z_j(t)$ with $W$\n"
        "Strategy only;\n"
        "Q tables not copied",
        COLORS["fermi"],
        COLORS["red"],
        fs=4.15,
    )

    add_box(
        ax,
        55,
        18,
        42,
        39,
        "Dual Q actions",
        "$\\epsilon$-greedy selection\n"
        "$Q_i^c(s_i^c,c)\\rightarrow c_i(t)\\in A_c$\n"
        "if $z_i\\in\\{C,P\\}$\n"
        "$Q_i^a(s_i^a,a)\\rightarrow a_i(t)\\in A_a$\n"
        "if $z_i=P$\n"
        "$D$: no investment or punishment",
        COLORS["action"],
        COLORS["teal"],
        fs=4.55,
    )
    add_box(
        ax,
        104,
        18,
        43,
        39,
        "Targets and rewards",
        "$q_i=n_{CP}^i/4$\n"
        "$c_i^*=c_{min}+(c_{max}-c_{min})q_i$\n"
        "$a_i^*=a_{min}$ inside stable clusters\n"
        "$a_i^*=a_{min}+(a_{max}-a_{min})d_i$\n"
        "when $n_D^i>0$ and $\\Pi_i<\\bar\\Pi_{\\mathcal{N}_i}$\n"
        "$R=1-|\\mathrm{action}-\\mathrm{target}|/\\mathrm{range}$",
        COLORS["reward"],
        "#8A63B2",
        fs=4.25,
    )
    add_box(
        ax,
        153,
        18,
        27,
        39,
        "Q update",
        "$Q\\leftarrow Q+\\eta\\,\\Delta Q$\n"
        "$\\Delta Q=R+\\gamma\\max Q'-Q$\n"
        "Update $Q_i^c$ if $C/P$\n"
        "Update $Q_i^a$ if $P$\n"
        "Only the experienced\n"
        "state-action pair changes\n"
        "$\\epsilon_{c,a}$ decay",
        COLORS["update"],
        "#C9A227",
        fs=4.2,
    )

    add_arrow(ax, (49, 89), (55, 89))
    add_arrow(ax, (97, 89), (104, 89))
    add_arrow(ax, (147, 89), (153, 89))
    add_arrow(ax, (76, 75), (76, 57), label="states", label_xy=(82, 66))
    add_arrow(ax, (94, 57), (106, 75), label="$c_i,a_i$", label_xy=(102, 64))
    add_arrow(ax, (125.5, 75), (125.5, 57), label="$\\Pi_i$", label_xy=(132, 66))
    add_arrow(ax, (147, 37.5), (153, 37.5))
    add_arrow(ax, (166.5, 72), (166.5, 57), label="$s(t+1)$", label_xy=(173, 64))
    add_arrow(
        ax,
        (168, 104.2),
        (30, 104.2),
        rad=0.0,
        label="next round: recompute local states from $z(t+1)$",
        label_xy=(99, 106.8),
    )
    add_bezier_arrow(
        ax,
        (166.5, 18),
        (121, 2),
        (76, 18),
        label="updated Q tables and exploration rates",
        label_xy=(121, 9.6),
    )
    add_legend(ax)

    fig.savefig(f"{OUT_BASE}.svg", bbox_inches="tight")
    fig.savefig(f"{OUT_BASE}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_BASE}.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(f"{OUT_BASE}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
