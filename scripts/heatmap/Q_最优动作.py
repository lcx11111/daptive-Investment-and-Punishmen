import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# 1. 基础路径设置
# =========================

DATA_DIR = Path("data") / "Q"

CSV_NAMES = [
    "model_F_q_r2p1_L100_T10000_runs20_all_investment_q.csv",
    "model_F_q_r2p1_L100_T10000_runs20_all_punishment_q.csv",
]


# =========================
# 2. 画投资 Q 表最优动作热图
# =========================

def plot_investment_optimal_action(df, out_path):
    """
    投资 Q 表：
    状态：state_cp = n_C + n_P
    动作：action_value = 投资成本 c
    Q值：mean_q

    对每个 state_cp，取 mean_q 最大的 action_value
    """

    idx = df.groupby("state_cp")["mean_q"].idxmax()
    best_df = df.loc[idx].copy()
    best_df = best_df.sort_values("state_cp")

    states = best_df["state_cp"].astype(int).to_numpy()
    best_actions = best_df["action_value"].astype(float).to_numpy()

    # 画成 1 × 状态数 的热图
    heatmap = best_actions.reshape(1, -1)

    action_min = df["action_value"].min()
    action_max = df["action_value"].max()

    plt.figure(figsize=(6, 2.2))

    im = plt.imshow(
        heatmap,
        aspect="auto",
        vmin=action_min,
        vmax=action_max
    )

    cbar = plt.colorbar(im)
    cbar.set_label("Optimal investment action, $c^*$")

    plt.xticks(range(len(states)), states)
    plt.yticks([0], [""])

    plt.xlabel("Investment state, $n_{CP}$")

    # 标注每个状态下的最优投资动作
    for j, value in enumerate(best_actions):
        plt.text(
            j,
            0,
            f"{value:.1f}",
            ha="center",
            va="center",
            color="black",
            fontsize=11
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"最优投资动作热图已保存到：{out_path}")


# =========================
# 3. 画惩罚 Q 表最优动作热图
# =========================

def plot_punishment_optimal_action(df, out_path):
    """
    惩罚 Q 表：
    状态：state_d = n_D, state_p = n_P
    动作：action_value = 惩罚成本 a
    Q值：mean_q

    对每个 (state_d, state_p)，取 mean_q 最大的 action_value
    """

    idx = df.groupby(["state_d", "state_p"])["mean_q"].idxmax()
    best_df = df.loc[idx].copy()

    max_d = int(df["state_d"].max())
    max_p = int(df["state_p"].max())

    heatmap = np.full((max_p + 1, max_d + 1), np.nan)

    for _, row in best_df.iterrows():
        n_D = int(row["state_d"])
        n_P = int(row["state_p"])
        best_action = float(row["action_value"])
        heatmap[n_P, n_D] = best_action

    action_min = df["action_value"].min()
    action_max = df["action_value"].max()

    plt.figure(figsize=(6, 5))

    im = plt.imshow(
        heatmap,
        origin="lower",
        aspect="equal",
        vmin=action_min,
        vmax=action_max
    )

    cbar = plt.colorbar(im)
    cbar.set_label("Optimal punishment action, $a^*$")

    plt.xticks(range(max_d + 1), range(max_d + 1))
    plt.yticks(range(max_p + 1), range(max_p + 1))

    plt.xlabel("$n_D$")
    plt.ylabel("$n_P$")

    # 标注每个状态下的最优惩罚动作
    for n_P in range(max_p + 1):
        for n_D in range(max_d + 1):
            if not np.isnan(heatmap[n_P, n_D]):
                plt.text(
                    n_D,
                    n_P,
                    f"{heatmap[n_P, n_D]:.1f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10
                )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"最优惩罚动作热图已保存到：{out_path}")


# =========================
# 4. 自动识别 Q 表类型并绘图
# =========================

def plot_optimal_action_heatmap(csv_name):
    csv_path = DATA_DIR / csv_name

    df = pd.read_csv(csv_path)

    stem = csv_path.stem

    # 自动识别投资 Q 表
    if "state_cp" in df.columns:
        out_path = DATA_DIR / f"{stem}_optimal_investment_action_heatmap.png"
        plot_investment_optimal_action(df, out_path)

    # 自动识别惩罚 Q 表
    elif "state_d" in df.columns and "state_p" in df.columns:
        out_path = DATA_DIR / f"{stem}_optimal_punishment_action_heatmap.png"
        plot_punishment_optimal_action(df, out_path)

    else:
        raise ValueError(
            "无法识别 Q 表类型。投资 Q 表需要 state_cp 列；惩罚 Q 表需要 state_d 和 state_p 列。"
        )


# =========================
# 5. 批量绘制
# =========================

for csv_name in CSV_NAMES:
    plot_optimal_action_heatmap(csv_name)