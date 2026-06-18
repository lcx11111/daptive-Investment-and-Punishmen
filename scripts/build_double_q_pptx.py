from __future__ import annotations

from datetime import datetime
from pathlib import Path
import html
import json
import shutil
import zipfile
import xml.etree.ElementTree as ET

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "result" / "texfig_latex"
OUT = ROOT / "output"
ASSET_DIR = OUT / "assets" / "figures"
PPTX_PATH = OUT / "final_presentation_cn.pptx"
QA_PATH = OUT / "qa_report.md"
MANIFEST_PATH = OUT / "asset_manifest.md"
SCRIPT_PATH = OUT / "ppt_script_cn_with_figures.md"

EMU_PER_INCH = 914400
SLIDE_W = int(13.333333 * EMU_PER_INCH)
SLIDE_H = int(7.5 * EMU_PER_INCH)
NOTES_W = int(7.5 * EMU_PER_INCH)
NOTES_H = int(10 * EMU_PER_INCH)

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

COLORS = {
    "ink": "1B1F23",
    "muted": "5B6570",
    "line": "D7DCE2",
    "teal": "236B6B",
    "orange": "B95F2A",
}


def inch(value: float) -> int:
    return int(round(value * EMU_PER_INCH))


def esc(text: object) -> str:
    return html.escape(str(text), quote=False)


def srgb(color: str) -> str:
    return color.replace("#", "").upper()


def fig(name: str) -> Path:
    path = SRC_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def run_xml(text: str, size: float = 18, color: str | None = None, bold: bool = False) -> str:
    attrs = [f'lang="zh-CN"', f'sz="{int(size * 100)}"']
    if bold:
        attrs.append('b="1"')
    color = color or COLORS["ink"]
    return (
        f'<a:r><a:rPr {" ".join(attrs)} dirty="0">'
        f'<a:solidFill><a:srgbClr val="{srgb(color)}"/></a:solidFill>'
        '<a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/>'
        '<a:cs typeface="Arial"/></a:rPr>'
        f"<a:t>{esc(text)}</a:t></a:r>"
    )


def p_xml(
    text: str = "",
    size: float = 18,
    color: str | None = None,
    bold: bool = False,
    align: str = "l",
    bullet: bool = False,
    space_after: int = 6,
) -> str:
    mar = inch(0.18) if bullet else 0
    indent = -inch(0.13) if bullet else 0
    bullet_xml = '<a:buChar char="•"/>' if bullet else ""
    ppr = (
        f'<a:pPr algn="{align}" marL="{mar}" indent="{indent}">'
        f'<a:spcAft><a:spcPts val="{space_after * 100}"/></a:spcAft>{bullet_xml}</a:pPr>'
    )
    return f"<a:p>{ppr}{run_xml(text, size=size, color=color, bold=bold)}</a:p>"


def text_box(
    shape_id: int,
    name: str,
    x: int,
    y: int,
    w: int,
    h: int,
    paragraphs: list[str],
    margin: float = 0.04,
    ph_type: str | None = None,
) -> str:
    placeholder = f'<p:ph type="{ph_type}"/>' if ph_type else ""
    body = "".join(paragraphs)
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{shape_id}" name="{esc(name)}"/><p:cNvSpPr txBox="1"/><p:nvPr>{placeholder}</p:nvPr></p:nvSpPr>
      <p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>
      <p:txBody><a:bodyPr wrap="square" anchor="t" lIns="{inch(margin)}" rIns="{inch(margin)}" tIns="{inch(margin)}" bIns="{inch(margin)}"/><a:lstStyle/>{body}</p:txBody>
    </p:sp>"""


def rect(shape_id: int, name: str, x: int, y: int, w: int, h: int, fill: str) -> str:
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{shape_id}" name="{esc(name)}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
      <p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="{srgb(fill)}"/></a:solidFill><a:ln><a:noFill/></a:ln></p:spPr>
      <p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody>
    </p:sp>"""


def picture(shape_id: int, rel_id: str, name: str, x: int, y: int, w: int, h: int) -> str:
    return f"""
    <p:pic>
      <p:nvPicPr><p:cNvPr id="{shape_id}" name="{esc(name)}"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>
      <p:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
      <p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>
    </p:pic>"""


def fit(path: Path, x: int, y: int, max_w: int, max_h: int) -> tuple[int, int, int, int]:
    with Image.open(path) as image:
        iw, ih = image.size
    scale = min(max_w / iw, max_h / ih)
    w, h = int(iw * scale), int(ih * scale)
    return x + (max_w - w) // 2, y + (max_h - h) // 2, w, h


def make_slides() -> list[dict]:
    return [
        {
            "title": "双 Q-learning 自适应投资–惩罚空间公共物品博弈",
            "subtitle": "模型机制与数值结果汇报",
            "bullets": [
                "策略 D/C/P 由 Fermi 社会模仿更新",
                "投资成本与惩罚成本由两张 Q 表分别学习",
                "目标：提高低收益区间的合作形成能力，同时降低长期惩罚负担",
            ],
            "images": [{"file": fig("res_03_evolution_flow.png"), "box": (6.2, 1.25, 6.65, 3.7), "label": "模型演化流程"}],
            "takeaway": "把“是否合作”的策略竞争，与“投入多少、惩罚多强”的行为调节分开建模。",
            "notes": "开场先强调模型的两个层次：策略层仍是空间公共物品博弈中的 D、C、P 竞争；行为层则通过双 Q 表学习投资与惩罚强度。",
        },
        {
            "title": "固定投资与固定惩罚难以同时兼顾启动合作和控制成本",
            "bullets": [
                "低 r 条件下，公共品收益不足，合作者容易被背叛者剥削。",
                "固定高投资会在背叛环境中放大损失；固定高惩罚会让惩罚者承担持续成本。",
                "关键问题不是“惩罚是否存在”，而是“何时提高投资、何时提高惩罚”。",
            ],
            "images": [{"file": fig("res_02_model_table.png"), "box": (5.7, 1.3, 7.0, 2.8), "label": "A–E 对照模型设置"}],
            "takeaway": "模型 E 的出发点是把成本强度从固定参数改成局部状态依赖的学习结果。",
            "notes": "固定惩罚模型常见的问题是强度选择敏感：太弱无法保护合作，太强会降低惩罚者收益。",
        },
        {
            "title": "模型 E：策略更新和行为强度学习被明确分离",
            "bullets": [
                "空间结构：二维方格网络，周期边界，每个个体参与 5 人公共物品小组。",
                "策略集合：D 不投资不惩罚；C 投资公共品；P 投资并惩罚背叛者。",
                "行为学习：Qᶜ 学习投资成本 c，Qᵃ 学习惩罚成本 a。",
            ],
            "images": [{"file": fig("res_02_model_table.png"), "box": (0.85, 3.05, 11.7, 2.6), "label": "Source: res_02_model_table.png"}],
            "takeaway": "完整模型 E 的特征是同时具备自适应投资和自适应惩罚。",
            "notes": "D、C、P 不是 Q-learning 学出来的策略类别，而是通过 Fermi 规则模仿更新；Q-learning 决定的是 C 和 P 的投资成本，以及 P 的惩罚成本。",
        },
        {
            "title": "非线性罚金函数让惩罚者数量与平均惩罚强度共同作用",
            "bullets": [
                "背叛者罚金：Fᵍ = n_Pᵍ[exp(β_F · āᵍ) − 1]。",
                "n_Pᵍ 越多，惩罚者形成的集体压力越强。",
                "β_F 控制惩罚成本向罚金转化的非线性强度。",
            ],
            "images": [{"file": fig("res_01_fine_formula.png"), "box": (0.8, 1.35, 7.25, 4.85), "label": "罚金函数示意"}],
            "takeaway": "惩罚不只是惩罚者的线性成本，也是对背叛者的非线性集体抑制。",
            "notes": "惩罚者支付 a，但背叛者受到的罚金还取决于惩罚者数量和平均惩罚成本，因此在空间团簇边界会出现集体防御效应。",
        },
        {
            "title": "每轮演化包含一条策略模仿通道和两条 Q 表更新通道",
            "bullets": [
                "Fermi 规则只改变 D/C/P 策略，不复制邻居的 Q 表。",
                "投资 Q 表状态：邻域中 C/P 邻居数 n_CP。",
                "惩罚 Q 表状态：背叛邻居数 n_D 与惩罚邻居数 n_P。",
            ],
            "images": [{"file": fig("res_03_evolution_flow.png"), "box": (0.7, 1.35, 12.0, 4.75), "label": "Source: res_03_evolution_flow.png"}],
            "takeaway": "模型的关键耦合发生在收益和局部状态，而不是直接复制学习表。",
            "notes": "流程图可按状态观测、动作选择、收益计算、策略更新、Q 表更新五步讲。",
        },
        {
            "title": "投资 Q 表学到的是条件性投资，而不是固定高投资",
            "bullets": [
                "状态 S_c = {0,1,2,3,4} 表示邻域合作水平。",
                "目标投资 c* 随 n_CP/4 线性提高。",
                "Q 值结果显示：合作邻居越多，最优投资越高。",
            ],
            "images": [{"file": fig("res_24_investment_Q_optimized.png"), "box": (0.75, 1.25, 7.4, 5.25), "label": "投资 Q 表优化表达"}],
            "takeaway": "低合作环境中降低被剥削损失，高合作环境中放大公共品收益。",
            "notes": "投资学习不是鼓励所有人满额投资，而是在局部合作稳定时提高投资；当周围背叛较多时，较低投资是保护性选择。",
        },
        {
            "title": "惩罚 Q 表把惩罚集中在背叛压力存在的空间边界",
            "bullets": [
                "二维状态 (n_D, n_P) 同时刻画背叛压力和惩罚者局部结构。",
                "n_D=0 时最低惩罚通常最优，避免合作团簇内部无效成本。",
                "n_D 增加时，最优动作转向中高惩罚，形成边界防御。",
            ],
            "images": [
                {"file": fig("res_26_punishment_Q_policy.png"), "box": (0.75, 1.18, 5.35, 5.55), "label": "最优惩罚动作策略图"},
                {"file": fig("res_27_punishment_Q_facets.png"), "box": (6.25, 1.35, 6.55, 3.95), "label": "不同局部状态下的 Q 值分面"},
            ],
            "takeaway": "惩罚学习的目标是“按需防御”，不是长期维持全局高惩罚。",
            "notes": "没有背叛者时，高惩罚没有目标，只会消耗惩罚者；有背叛者且自身相对收益受损时，惩罚成本才随局部背叛比例提高。",
        },
        {
            "title": "消融对比显示：双自适应机制使合作跃迁更早出现",
            "grid": True,
            "images": [
                {"file": fig("res_04_ablation_AC.png"), "box": (0.55, 1.1, 4.0, 2.25), "label": "A vs C"},
                {"file": fig("res_05_ablation_AD.png"), "box": (4.68, 1.1, 4.0, 2.25), "label": "A vs D"},
                {"file": fig("res_06_ablation_AE.png"), "box": (8.82, 1.1, 4.0, 2.25), "label": "A vs E"},
                {"file": fig("res_07_ablation_BE.png"), "box": (0.55, 3.75, 4.0, 2.25), "label": "B vs E"},
                {"file": fig("res_08_ablation_CE.png"), "box": (4.68, 3.75, 4.0, 2.25), "label": "C vs E"},
                {"file": fig("res_09_ablation_DE.png"), "box": (8.82, 3.75, 4.0, 2.25), "label": "D vs E"},
            ],
            "takeaway": "模型 E 相比固定、随机、单通道自适应设置，在低 r 区间更容易进入高合作状态。",
            "notes": "这一页快速扫六组对照。重点是随机成本不能替代学习，仅自适应投资或仅自适应惩罚都不如二者耦合稳定。",
        },
        {
            "title": "模型 E 提升合作后并未表现为长期收益损失",
            "bullets": [
                "D–E 对比将焦点放在自适应惩罚相对于固定惩罚的增益。",
                "平均收益结果说明，模型 E 的惩罚成本主要用于短期边界防御。",
                "合作建立后，惩罚成本下降，系统收益没有被持续高惩罚吞噬。",
            ],
            "images": [{"file": fig("res_10_ablation_DE_payoff.png"), "box": (0.8, 1.35, 7.05, 4.75), "label": "模型 D 与模型 E 平均收益对比"}],
            "takeaway": "自适应惩罚的价值在于调节惩罚位置和时机，而不是简单加大惩罚。",
            "notes": "模型 E 不是靠长期高成本硬压背叛者，而是在背叛者仍存在的空间边界提高惩罚，等合作区稳定后降低惩罚负担。",
        },
        {
            "title": "r=2.1 临界附近：惩罚者扩张推动合作建立",
            "bullets": [
                "早期背叛者上升，随后惩罚者逐渐扩张并成为主导。",
                "合作率在中后期跃升，说明局部团簇越过临界阈值。",
                "投资成本随合作环境稳定而提高，惩罚成本在背叛消退后下降。",
            ],
            "images": [
                {"file": fig("res_16_r2p1_strategy.png"), "box": (0.55, 1.25, 3.8, 2.35), "label": "策略密度"},
                {"file": fig("res_17_r2p1_cooperation.png"), "box": (4.45, 1.25, 3.8, 2.35), "label": "合作率"},
                {"file": fig("res_18_r2p1_cost.png"), "box": (0.55, 3.95, 3.8, 2.35), "label": "平均成本"},
                {"file": fig("res_22_r2p1_beta.png"), "box": (4.45, 3.95, 3.8, 2.35), "label": "β_F 对比"},
            ],
            "takeaway": "r=2.1 展示了双 Q 表从“防御背叛”过渡到“放大合作”的完整过程。",
            "notes": "按时间顺序讲：早期竞争、惩罚者团簇建立、中后期合作率上升、最终惩罚成本降低。",
        },
        {
            "title": "空间快照显示惩罚者团簇沿边界向外推进",
            "bullets": [
                "红色 D 在早期仍可扩散，说明系统不是瞬时全合作。",
                "黄色 P 先形成局部团簇，再通过边界防御扩大合作区。",
                "最终 P/C 占据主体，空间结构成为合作扩张的载体。",
            ],
            "images": [{"file": fig("res_19_r2p1_snapshots.png"), "box": (0.55, 2.15, 12.25, 2.7), "label": "r=2.1 空间快照：红 D，蓝 C，黄 P"}],
            "takeaway": "合作不是均匀增长，而是通过空间团簇的局部筛选和边界推进形成。",
            "notes": "上一页看时间序列，这一页看空间过程。重点说清楚惩罚者在边界上的功能。",
        },
        {
            "title": "动作分布说明：高投资与低/中等惩罚可以同时出现",
            "bullets": [
                "投资动作逐渐向高投资集中，反映合作环境稳定后收益放大。",
                "惩罚动作没有长期集中到最高值，说明系统避免了全局高惩罚。",
                "这解释了模型 E 高合作率与较低长期惩罚负担可以并存。",
            ],
            "images": [
                {"file": fig("res_20_r2p1_investment_distribution.png"), "box": (0.65, 1.35, 5.7, 4.25), "label": "投资动作分布"},
                {"file": fig("res_21_r2p1_punishment_distribution.png"), "box": (6.6, 1.35, 5.85, 4.25), "label": "惩罚动作分布"},
            ],
            "takeaway": "双 Q 表把投资和惩罚拆成两种不同的局部调节逻辑。",
            "notes": "投资分布代表合作收益端，惩罚分布代表风险控制端。合作稳定后投资上升，但惩罚不必一直维持高位。",
        },
        {
            "title": "r=1.8 低收益区间：双 Q-learning 也存在启动边界",
            "bullets": [
                "公共品放大效应过低时，合作需要更长空间筛选过程。",
                "惩罚更像局部保护层，帮助合作簇抵抗背叛入侵。",
                "投资调节则在不稳定环境中降低被剥削损失。",
            ],
            "images": [{"file": fig("res_14_r1p8_snapshots.png"), "box": (0.55, 2.05, 12.25, 2.65), "label": "r=1.8 空间快照"}],
            "takeaway": "模型 E 的优势不是在任意低收益条件下消灭背叛，而是在可形成团簇时放大合作优势。",
            "notes": "这一页用于保持结论边界。低 r 下仍然需要局部合作团簇先出现，Q-learning 才能通过投资和惩罚调节扩大优势。",
        },
        {
            "title": "较高 r 下，合作更快进入稳定区",
            "bullets": [
                "r 提高后，公共品收益本身为合作提供更强基础。",
                "背叛者更难长期维持，合作/惩罚者团簇更快扩张。",
                "此时双 Q-learning 的作用更多是稳定合作并降低不必要成本。",
            ],
            "images": [
                {"file": fig("res_31_r3_snapshots.png"), "box": (0.75, 1.35, 11.7, 1.25), "label": "r=3"},
                {"file": fig("res_36_r4_snapshots.png"), "box": (0.75, 3.0, 11.7, 1.25), "label": "r=4"},
                {"file": fig("res_39_r5_snapshots.png"), "box": (0.75, 4.65, 11.7, 1.25), "label": "r=5"},
            ],
            "takeaway": "低 r 侧重“启动合作”，高 r 侧重“稳定合作”。",
            "notes": "用三个空间条带展示参数从困难区到稳定区的变化，强调趋势即可。",
        },
        {
            "title": "r–β_F 参数平面揭示公共品收益与惩罚效率的互补",
            "bullets": [
                "高合作区由较大的 r 或较高 β_F 共同支撑。",
                "平均投资成本的高值区域与高合作区域基本重合。",
                "平均惩罚成本整体较低，仅在困难区域或边界附近更活跃。",
            ],
            "images": [
                {"file": fig("res_41_beta_r_cooperation.png"), "box": (0.65, 1.3, 3.75, 3.45), "label": "合作水平"},
                {"file": fig("res_42_beta_r_investment.png"), "box": (4.75, 1.3, 3.75, 3.45), "label": "平均投资成本"},
                {"file": fig("res_43_beta_r_punishment.png"), "box": (8.85, 1.3, 3.75, 3.45), "label": "平均惩罚成本"},
            ],
            "takeaway": "当 r 较低时需要更高惩罚效率补偿；当 r 较高时，对惩罚敏感性的依赖降低。",
            "notes": "把单点演化推广到参数面。重点讲相变边界和互补关系：r 是收益基础，β_F 是惩罚效率。",
        },
        {
            "title": "核心结论：双 Q 表把合作维持转化为局部调节问题",
            "bullets": [
                "投资 Q 表：在可靠合作邻域提高投入，在背叛环境中降低被剥削风险。",
                "惩罚 Q 表：在背叛边界提高惩罚，在合作内部降低无效成本。",
                "消融结果：模型 E 比固定、随机或单通道自适应设置更稳定。",
                "展望：可扩展到异质网络、异步更新和连续动作空间。",
            ],
            "images": [{"file": fig("res_03_evolution_flow.png"), "box": (7.65, 1.4, 4.9, 2.6), "label": "机制摘要"}],
            "takeaway": "模型 E 的实质是让个体学习空间局部环境中的合适投资与惩罚强度。",
            "notes": "最后用三句话收束：策略更新与行为强度学习分离；两张 Q 表对应不同空间功能；结果支持双自适应机制提高合作扩张能力并减少长期惩罚负担。",
        },
    ]


def rels_xml(relationships: list[tuple[str, str, str]]) -> str:
    body = "".join(
        f'<Relationship Id="{rid}" Type="{rtype}" Target="{target}"/>'
        for rid, rtype, target in relationships
    )
    return f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="{REL_NS}">{body}</Relationships>'


def group_prefix() -> str:
    return (
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
        '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
    )


def build_slide_xml(slide: dict, slide_index: int, media_map: dict[Path, str]) -> tuple[str, str, str, str]:
    shape_id = 1
    shapes: list[str] = []
    shapes.append(rect(shape_id, "Accent bar", 0, 0, SLIDE_W, inch(0.065), COLORS["teal"]))
    shape_id += 1

    title_size = 30 if slide_index == 1 else 25
    title_w = inch(5.6 if slide_index == 1 else 12.1)
    title_h = inch(1.1 if slide_index == 1 else 0.58)
    shapes.append(
        text_box(
            shape_id,
            "Slide title",
            inch(0.55),
            inch(0.25),
            title_w,
            title_h,
            [p_xml(slide["title"], size=title_size, bold=True, color=COLORS["ink"], space_after=0)],
            margin=0.0,
        )
    )
    shape_id += 1

    if slide_index == 1:
        shapes.append(
            text_box(
                shape_id,
                "Subtitle",
                inch(0.6),
                inch(1.35),
                inch(5.25),
                inch(0.55),
                [p_xml(slide["subtitle"], size=18, color=COLORS["teal"], bold=True, space_after=0)],
                margin=0,
            )
        )
        shape_id += 1
        shapes.append(
            text_box(
                shape_id,
                "Cover bullets",
                inch(0.7),
                inch(2.05),
                inch(4.85),
                inch(1.8),
                [p_xml(b, size=16, color=COLORS["ink"], bullet=True, space_after=10) for b in slide["bullets"]],
                margin=0.02,
            )
        )
        shape_id += 1
        shapes.append(rect(shape_id, "Takeaway line", inch(0.7), inch(5.8), inch(0.08), inch(0.55), COLORS["orange"]))
        shape_id += 1
        shapes.append(
            text_box(
                shape_id,
                "Cover takeaway",
                inch(0.9),
                inch(5.72),
                inch(11.2),
                inch(0.75),
                [p_xml(slide["takeaway"], size=17, color=COLORS["ink"], bold=True, space_after=0)],
                margin=0.02,
            )
        )
        shape_id += 1
    elif slide.get("grid"):
        shapes.append(rect(shape_id, "Takeaway line", inch(0.55), inch(6.35), inch(0.08), inch(0.42), COLORS["orange"]))
        shape_id += 1
        shapes.append(
            text_box(
                shape_id,
                "Takeaway",
                inch(0.75),
                inch(6.25),
                inch(12.0),
                inch(0.6),
                [p_xml(slide["takeaway"], size=14.5, color=COLORS["ink"], bold=True, space_after=0)],
                margin=0.02,
            )
        )
        shape_id += 1
    else:
        if slide_index in [3, 5, 10, 11, 12, 13, 15]:
            yb = inch(5.25 if slide_index in [11, 12, 13] else 1.08)
            hb = inch(1.15 if slide_index in [11, 12, 13] else 1.65)
            xb = inch(0.65 if slide_index in [11, 12, 13] else 8.65)
            wb = inch(12.1 if slide_index in [11, 12, 13] else 3.8)
        else:
            xb, yb, wb, hb = inch(8.55), inch(1.25), inch(4.1), inch(3.55)

        shapes.append(rect(shape_id, "Interpretation accent", xb - inch(0.14), yb + inch(0.05), inch(0.055), min(hb, inch(2.2)), COLORS["teal"]))
        shape_id += 1
        shapes.append(
            text_box(
                shape_id,
                "Bullets",
                xb,
                yb,
                wb,
                hb,
                [p_xml(b, size=15.0 if slide_index == 16 else 15.2, color=COLORS["ink"], bullet=True, space_after=8) for b in slide["bullets"]],
                margin=0.01,
            )
        )
        shape_id += 1
        shapes.append(rect(shape_id, "Footer rule", inch(0.55), inch(6.72), inch(12.25), inch(0.018), COLORS["line"]))
        shape_id += 1
        shapes.append(
            text_box(
                shape_id,
                "Takeaway",
                inch(0.65),
                inch(6.78),
                inch(12.0),
                inch(0.38),
                [p_xml("Takeaway: " + slide["takeaway"], size=10.8, color=COLORS["muted"], space_after=0)],
                margin=0,
            )
        )
        shape_id += 1

    relationships = [("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout", "../slideLayouts/slideLayout1.xml")]
    rel_index = 2
    for image in slide.get("images", []):
        src = Path(image["file"])
        x, y, max_w, max_h = [inch(v) for v in image["box"]]
        fx, fy, fw, fh = fit(src, x, y, max_w, max_h)
        rid = f"rId{rel_index}"
        relationships.append((rid, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", f"../media/{media_map[src]}"))
        shapes.append(picture(shape_id, rid, src.name, fx, fy, fw, fh))
        shape_id += 1
        label = f"{image.get('label', src.name)} | {src.name}"
        shapes.append(
            text_box(
                shape_id,
                f"Image label {shape_id}",
                fx,
                fy + fh + inch(0.04),
                fw,
                inch(0.18),
                [p_xml(label, size=7.5, color=COLORS["muted"], space_after=0)],
                margin=0,
            )
        )
        shape_id += 1
        rel_index += 1

    relationships.append((f"rId{rel_index}", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide", f"../notesSlides/notesSlide{slide_index}.xml"))
    shapes.append(text_box(shape_id, "Slide number", inch(12.35), inch(7.12), inch(0.55), inch(0.2), [p_xml(str(slide_index), size=8, color=COLORS["muted"], align="r", space_after=0)], margin=0))

    slide_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="{NS["a"]}" xmlns:r="{NS["r"]}" xmlns:p="{NS["p"]}">
  <p:cSld><p:spTree>{group_prefix()}{"".join(shapes)}</p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>'''

    note_parts = [p_xml(slide["title"], size=16, bold=True, color=COLORS["ink"], space_after=12)]
    note_parts.extend(p_xml(line.strip(), size=12, color=COLORS["ink"], space_after=6) for line in slide["notes"].split("。") if line.strip())
    note_shape = text_box(2, "Speaker Notes", inch(0.65), inch(1.05), inch(6.2), inch(7.8), note_parts, margin=0.05, ph_type="body")
    notes_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:notes xmlns:a="{NS["a"]}" xmlns:r="{NS["r"]}" xmlns:p="{NS["p"]}">
  <p:cSld><p:spTree>{group_prefix()}{note_shape}</p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:notes>'''
    notes_rel = rels_xml([
        ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesMaster", "../notesMasters/notesMaster1.xml"),
        ("rId2", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide", f"../slides/slide{slide_index}.xml"),
    ])
    return slide_xml, rels_xml(relationships), notes_xml, notes_rel


def static_parts(slide_count: int) -> dict[str, str]:
    overrides = [
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Default Extension="png" ContentType="image/png"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
        '<Override PartName="/ppt/notesMasters/notesMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesMaster+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
        '<Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>',
        '<Override PartName="/ppt/viewProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"/>',
        '<Override PartName="/ppt/tableStyles.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml"/>',
    ]
    for i in range(1, slide_count + 1):
        overrides.append(f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>')
        overrides.append(f'<Override PartName="/ppt/notesSlides/notesSlide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"/>')

    now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    slide_ids = "".join(f'<p:sldId id="{255 + i}" r:id="rId{1 + i}"/>' for i in range(1, slide_count + 1))
    pres_rels = [("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", "slideMasters/slideMaster1.xml")]
    pres_rels.extend((f"rId{1 + i}", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide", f"slides/slide{i}.xml") for i in range(1, slide_count + 1))
    base = slide_count + 2
    pres_rels.extend([
        (f"rId{base}", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/presProps", "presProps.xml"),
        (f"rId{base + 1}", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/viewProps", "viewProps.xml"),
        (f"rId{base + 2}", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/tableStyles", "tableStyles.xml"),
    ])
    return {
        "[Content_Types].xml": f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">{"".join(overrides)}</Types>',
        "_rels/.rels": rels_xml([
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument", "ppt/presentation.xml"),
            ("rId2", "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties", "docProps/core.xml"),
            ("rId3", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties", "docProps/app.xml"),
        ]),
        "docProps/core.xml": f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>双 Q-learning 自适应投资--惩罚模型汇报</dc:title><dc:creator>Codex</dc:creator><cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>''',
        "docProps/app.xml": f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex OpenXML generator</Application><PresentationFormat>On-screen Show (16:9)</PresentationFormat><Slides>{slide_count}</Slides><Notes>{slide_count}</Notes><HiddenSlides>0</HiddenSlides><MMClips>0</MMClips><ScaleCrop>false</ScaleCrop><Company></Company><LinksUpToDate>false</LinksUpToDate><SharedDoc>false</SharedDoc><HyperlinksChanged>false</HyperlinksChanged><AppVersion>16.0000</AppVersion>
</Properties>''',
        "ppt/presentation.xml": f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="{NS["a"]}" xmlns:r="{NS["r"]}" xmlns:p="{NS["p"]}" saveSubsetFonts="1">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:sldIdLst>{slide_ids}</p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/><p:notesSz cx="{NOTES_W}" cy="{NOTES_H}"/>
  <p:defaultTextStyle><a:defPPr><a:defRPr lang="zh-CN"><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/></a:defRPr></a:defPPr></p:defaultTextStyle>
</p:presentation>''',
        "ppt/_rels/presentation.xml.rels": rels_xml(pres_rels),
        "ppt/slideMasters/slideMaster1.xml": f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="{NS["a"]}" xmlns:r="{NS["r"]}" xmlns:p="{NS["p"]}">
  <p:cSld><p:spTree>{group_prefix()}</p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>''',
        "ppt/slideMasters/_rels/slideMaster1.xml.rels": rels_xml([
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout", "../slideLayouts/slideLayout1.xml"),
            ("rId2", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme", "../theme/theme1.xml"),
        ]),
        "ppt/slideLayouts/slideLayout1.xml": f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="{NS["a"]}" xmlns:r="{NS["r"]}" xmlns:p="{NS["p"]}" type="blank" preserve="1"><p:cSld name="Blank"><p:spTree>{group_prefix()}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>''',
        "ppt/slideLayouts/_rels/slideLayout1.xml.rels": rels_xml([
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", "../slideMasters/slideMaster1.xml")
        ]),
        "ppt/notesMasters/notesMaster1.xml": f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:notesMaster xmlns:a="{NS["a"]}" xmlns:r="{NS["r"]}" xmlns:p="{NS["p"]}">
  <p:cSld><p:spTree>{group_prefix()}</p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:notesStyle><a:lvl1pPr marL="0" algn="l"><a:defRPr sz="1200"><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/></a:defRPr></a:lvl1pPr></p:notesStyle>
</p:notesMaster>''',
        "ppt/notesMasters/_rels/notesMaster1.xml.rels": rels_xml([
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme", "../theme/theme1.xml")
        ]),
        "ppt/theme/theme1.xml": f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="{NS["a"]}" name="DoubleQ Nature Light"><a:themeElements>
  <a:clrScheme name="DoubleQ"><a:dk1><a:srgbClr val="1B1F23"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="263238"/></a:dk2><a:lt2><a:srgbClr val="F7F8FA"/></a:lt2><a:accent1><a:srgbClr val="236B6B"/></a:accent1><a:accent2><a:srgbClr val="B95F2A"/></a:accent2><a:accent3><a:srgbClr val="365F91"/></a:accent3><a:accent4><a:srgbClr val="2F7D4E"/></a:accent4><a:accent5><a:srgbClr val="7A5C9E"/></a:accent5><a:accent6><a:srgbClr val="C4A000"/></a:accent6><a:hlink><a:srgbClr val="236B6B"/></a:hlink><a:folHlink><a:srgbClr val="5B6570"/></a:folHlink></a:clrScheme>
  <a:fontScheme name="Office"><a:majorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/><a:cs typeface="Arial"/></a:majorFont><a:minorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/><a:cs typeface="Arial"/></a:minorFont></a:fontScheme>
  <a:fmtScheme name="Clean"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
</a:themeElements><a:objectDefaults/><a:extraClrSchemeLst/></a:theme>''',
        "ppt/presProps.xml": f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:presentationPr xmlns:a="{NS["a"]}" xmlns:r="{NS["r"]}" xmlns:p="{NS["p"]}"/>',
        "ppt/viewProps.xml": f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:viewPr xmlns:a="{NS["a"]}" xmlns:r="{NS["r"]}" xmlns:p="{NS["p"]}"><p:normalViewPr><p:restoredLeft sz="15620"/><p:restoredTop sz="94660"/></p:normalViewPr><p:slideViewPr><p:cSldViewPr><p:cViewPr varScale="1"><p:scale><a:sx n="100" d="100"/><a:sy n="100" d="100"/></p:scale><p:origin x="0" y="0"/></p:cViewPr><p:guideLst/></p:cSldViewPr></p:slideViewPr><p:notesTextViewPr><p:cViewPr><p:scale><a:sx n="1" d="1"/><a:sy n="1" d="1"/></p:scale><p:origin x="0" y="0"/></p:cViewPr></p:notesTextViewPr><p:gridSpacing cx="72008" cy="72008"/></p:viewPr>',
        "ppt/tableStyles.xml": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:tblStyleLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" def="{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"/>',
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    slides = make_slides()
    used: list[Path] = []
    for slide in slides:
        for image in slide.get("images", []):
            src = Path(image["file"])
            if src not in used:
                used.append(src)
                shutil.copy2(src, ASSET_DIR / src.name)

    media_map = {src: f"image{index}{src.suffix.lower()}" for index, src in enumerate(used, start=1)}

    manifest = ["# Asset Manifest", "", "| Asset | Source file | Used on slides | Placement / role |", "|---|---|---|---|"]
    for src in used:
        placements = []
        for slide_index, slide in enumerate(slides, start=1):
            for image in slide.get("images", []):
                if Path(image["file"]) == src:
                    placements.append(f"{slide_index}: {image.get('label', 'figure')}")
        manifest.append(f"| `{src.name}` | `{src.relative_to(ROOT)}` | {', '.join(placements)} | copied original PNG; no scientific data altered |")
    MANIFEST_PATH.write_text("\n".join(manifest) + "\n", encoding="utf-8")

    script = ["# 双 Q-learning 自适应投资--惩罚模型 PPT 讲稿", ""]
    slide_xmls = []
    for slide_index, slide in enumerate(slides, start=1):
        slide_xmls.append(build_slide_xml(slide, slide_index, media_map))
        script.extend([f"## Slide {slide_index}. {slide['title']}", "", f"- 核心句：{slide['takeaway']}"])
        if slide.get("bullets"):
            script.append("- 页面要点：")
            script.extend(f"  - {bullet}" for bullet in slide["bullets"])
        if slide.get("images"):
            script.append("- 使用图片：" + ", ".join(Path(image["file"]).name for image in slide["images"]))
        script.append(f"- 讲稿提示：{slide['notes']}")
        script.append("")
    SCRIPT_PATH.write_text("\n".join(script), encoding="utf-8")

    with zipfile.ZipFile(PPTX_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in static_parts(len(slides)).items():
            archive.writestr(name, content)
        for slide_index, (slide_xml, slide_rel, notes_xml, notes_rel) in enumerate(slide_xmls, start=1):
            archive.writestr(f"ppt/slides/slide{slide_index}.xml", slide_xml)
            archive.writestr(f"ppt/slides/_rels/slide{slide_index}.xml.rels", slide_rel)
            archive.writestr(f"ppt/notesSlides/notesSlide{slide_index}.xml", notes_xml)
            archive.writestr(f"ppt/notesSlides/_rels/notesSlide{slide_index}.xml.rels", notes_rel)
        for src, media in media_map.items():
            archive.write(src, f"ppt/media/{media}")

    problems = []
    with zipfile.ZipFile(PPTX_PATH, "r") as archive:
        names = set(archive.namelist())
        slide_parts = [name for name in names if name.startswith("ppt/slides/slide") and name.endswith(".xml")]
        media_parts = [name for name in names if name.startswith("ppt/media/")]
        note_parts = [name for name in names if name.startswith("ppt/notesSlides/notesSlide") and name.endswith(".xml")]
        for name in sorted(name for name in names if name.endswith(".xml") or name.endswith(".rels")):
            try:
                ET.fromstring(archive.read(name))
            except Exception as exc:  # pragma: no cover - CLI validation path
                problems.append(f"XML parse error in {name}: {exc}")

    qa = [
        "# QA Report",
        "",
        f"- PPTX: `{PPTX_PATH.relative_to(ROOT)}`",
        f"- Creation status: {'OK' if not problems else 'Created with warnings'}",
        f"- Slide count: {len(slide_parts)}",
        f"- Embedded unique media files: {len(media_parts)}",
        f"- Speaker notes XML parts: {len(note_parts)}",
        f"- Figure assets copied to: `{ASSET_DIR.relative_to(ROOT)}`",
        "- Verification method: ZIP package reopen + XML well-formedness parse + required-part count.",
        "- Rendered preview: not performed; no local `python-pptx` or LibreOffice renderer was available.",
    ]
    if problems:
        qa.append("- Known issues:")
        qa.extend(f"  - {problem}" for problem in problems)
    else:
        qa.append("- Known issues: none detected by lightweight structural checks.")
    qa.append("- Manual follow-up: open the PPTX in PowerPoint/WPS to visually confirm final font substitution and slide rendering.")
    QA_PATH.write_text("\n".join(qa) + "\n", encoding="utf-8")

    print(json.dumps({
        "pptx": str(PPTX_PATH),
        "slides": len(slide_parts),
        "media": len(media_parts),
        "notes": len(note_parts),
        "warnings": problems,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
