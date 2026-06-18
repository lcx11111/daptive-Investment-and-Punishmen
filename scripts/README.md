# 代码分类

建议始终在项目根目录运行脚本，例如：

```powershell
python scripts\heatmap\beta_wc.py
```

这样所有脚本的默认输出仍会写到根目录下的 `data/` 或 `figures/`。

## scan_r

合作率随公共品倍增因子 `r` 扫描，以及合作率对比图：

- `A.py`, `B.py`, `C.py`, `D.py`, `E.py`
- `A_D.py`, `A_E.py`, `B_E.py`, `C_E.py`, `D_E.py`

## heatmap

二维参数热图和 Q 表热图：

- `a_a.py`, `alpha_gamma_a.py`, `alpha_gamma_c.py`
- `beta_r.py`, `beta_wc.py`
- `Q.py`, `Q_最优动作.py`

## evolution

策略、模型 E、成本、beta、权重等随轮次 `t` 的演化图：

- `evo.py`, `evo_beta.py`, `evo_cost.py`
- `A_evo.py`, `B_evo.py`, `C_evo.py`, `D_evo.py`
- `evo_abcd_common.py`

## payoff_r

平均收益随 `r` 扫描，以及平均收益对比图：

- `A_payoff_r.py`, `B_payoff_r.py`, `C_payoff_r.py`, `D_payoff_r.py`, `E_payoff_r.py`
- `A_C_payoff.py`, `A_D_payoff.py`, `A_E_payoff.py`
- `B_E_payoff.py`, `C_E_payoff.py`, `D_E_payoff.py`
- `payoff.py`, `payoff_compare_common.py`, `payoff_r_common.py`

## snapshot

空间策略快照图：

- `snap.py`
