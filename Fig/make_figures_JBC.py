"""
make_figures_JBC.py —— Springer LNCS / CCIS 主图统一生成脚本
=============================================================
输出：Table 1（CSV + Markdown）、Fig 2 / 3 / 4（正文主图）、Fig S1（补充材料）
      以及 alt_text.md（Springer Sect. 3 要求的替代文本初稿）

排版口径
  · 正文版心 122 mm、补充材料版心 160 mm，图按该宽度 1:1 设计。
    插入 Word 后把图宽显式设为对应毫米数即可，脚本内的 pt 数就是印刷 pt 数。
  · 图内最小字号受 FS_MIN 约束（Springer 硬性下限 6 pt，这里取 6.5 pt）。
    全局放大用 FS_SCALE，一处改动即可整体缩放所有字号。
  · 以颜色区分类别处一律叠加 hatch 纹理或不同 marker，保证黑白印刷可读。
  · savefig.bbox=None：输出画布严格等于 figsize，避免二次缩放。

运行
    python make_figures_JBC.py                      # 全部
    python make_figures_JBC.py --fig 2              # 只生成 Fig 2
    python make_figures_JBC.py --fig 2 3 4 S1       # 任意子集
    python make_figures_JBC.py --formats pdf png    # 指定输出格式
"""

import argparse
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════════════════════
#  PART 1 —— 尺寸、字号与配色常量
# ═════════════════════════════════════════════════════════════
MM = 1 / 25.4

TEXT_W_MM = 122.0          # 正文版心宽度
ESM_W_MM = 160.0           # 补充材料版心宽度
W_TEXT = TEXT_W_MM * MM
W_ESM = ESM_W_MM * MM

# 图高。字号整体放大后行距随之变紧，这里同步给出补偿后的高度；
# 若需回到更紧凑的排版，把下面四个值各减 4~6 mm 即可。
H_MAIN_MM = 100.0          # Fig 2：2×2 四面板
H_FIG3_MM = 96.0           # Fig 3：上排跨栏 + 下排两面板
H_FIG4_MM = 102.0          # Fig 4：Panel C 的 GO 条目占下排大半高度
H_ESM_MM = 156.0           # Fig S1：五面板

FIG_MAIN = (W_TEXT, H_MAIN_MM * MM)
FIG_FIG3 = (W_TEXT, H_FIG3_MM * MM)
FIG_FIG4 = (W_TEXT, H_FIG4_MM * MM)
FIG_ESM = (W_ESM, H_ESM_MM * MM)

# ── 字号 ────────────────────────────────────────────────────
# FS_SCALE：全局放大系数，作用于脚本中每一处经 fs() 的字号以及 rcParams。
#           1.00 = 原始设定；当前 1.15 即整体放大约一档。
# FS_MIN  ：放大后仍强制不低于此值（Springer 硬性下限 6 pt）。
FS_SCALE = 1.15
FS_MIN = 6.5


def fs(size):
    """把硬编码字号统一缩放并收敛到下限之上。"""
    return max(float(size) * FS_SCALE, FS_MIN)


# ── 黑白可读性 ──────────────────────────────────────────────
HATCH_3 = ["", "/////", "....."]        # 三个 holdout
MARKER_4 = ["o", "s", "^", "D"]         # 四个象限

# ── 色盲友好配色（Wong 2011）────────────────────────────────
C = {
    # 消融链 D→L→B→T→C
    "D": "#AECDE8", "L": "#56B4E9", "B": "#0072B2",
    "T": "#CC79A7", "C": "#D55E00",
    # 四象限
    "Q1": "#0072B2", "Q2": "#D55E00",
    "Q3": "#009E73", "Q4": "#BBBBBB",
    # 3 holdout
    "main": "#D55E00", "chr3_4": "#0072B2", "chr1_2": "#009E73",
    # 辅助
    "gray": "#888888", "ref": "#CCCCCC", "bg": "#F7F7F7",
}

CMAP_COORD = LinearSegmentedColormap.from_list(
    "cmap_coord", ["#FFFFFF", "#E69F00", "#7F3300"])


# ═════════════════════════════════════════════════════════════
#  PART 2 —— 全局样式
# ═════════════════════════════════════════════════════════════
def apply_springer_style():
    """应用 Springer LNCS / CCIS 规范的全局 rcParams。"""
    mpl.rcParams.update({
        # 字体（fallback 链，避免 findfont 警告）
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans",
                            "Liberation Sans"],
        "font.size": fs(7),
        "axes.titlesize": fs(8),
        "axes.labelsize": fs(7.5),
        "xtick.labelsize": fs(6.5),
        "ytick.labelsize": fs(6.5),
        "legend.fontsize": fs(6.5),
        "figure.titlesize": fs(8),
        # 线条：恒定线宽，全部 ≥ 0.4 pt
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,
        "lines.linewidth": 0.9,
        "patch.linewidth": 0.4,
        "grid.linewidth": 0.4,
        "hatch.linewidth": 0.4,
        # 坐标轴
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelpad": 2.0,
        # 图例
        "legend.frameon": False,
        "legend.handlelength": 1.1,
        "legend.handleheight": 0.7,
        "legend.borderaxespad": 0.2,
        "legend.labelspacing": 0.25,
        "legend.columnspacing": 0.8,
        # 紧凑布局
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        "figure.constrained_layout.hspace": 0.03,
        "figure.constrained_layout.wspace": 0.03,
        # 保存
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.use14corefonts": False,
        "ps.useafm": False,
        "svg.fonttype": "none",
        "savefig.dpi": 1200,
        "savefig.bbox": None,
        "savefig.transparent": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "image.cmap": "viridis",
        "axes.prop_cycle": mpl.cycler(color=[
            "#0072B2", "#E69F00", "#009E73", "#D55E00",
            "#56B4E9", "#CC79A7", "#F0E442", "#000000",
        ]),
    })


def label_panel(ax, letter, x=-0.12, y=1.06):
    """子图角落 panel 字母（A/B/C/D/E），供 caption 引用。"""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=fs(9), fontweight="bold", va="top", ha="left")


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ═════════════════════════════════════════════════════════════
#  PART 3 —— 路径与 I/O
# ═════════════════════════════════════════════════════════════
BASE = Path("../53_filtered_79")
CHR34 = Path("../53_filtered_79_holdout_chr3_chr4")
CHR12 = Path("../53_filtered_79_holdout_chr1_chr2")
GO_DIR = Path("../GO/result")
OUT_DIR = Path("figures_JBC")
OUT_DIR.mkdir(exist_ok=True)

# 输出参数，由 main() 按命令行设置
SAVE_FORMATS = ("pdf", "png")
SAVE_DPI = 1200
GRAY_PREVIEW = False


def _col(df, candidates, default=None):
    """宽容地查找列名。"""
    for c in candidates:
        if c in df.columns:
            return c
    return default


def _save(fig, path):
    """按 SAVE_FORMATS / SAVE_DPI 保存图像。

    rcParams["savefig.bbox"] 为 None，输出画布严格等于 figsize，
    因此 PNG 像素宽 = figsize_inch × dpi，Word 中按版心毫米数设宽即 1:1。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fmts = []
    for f in SAVE_FORMATS:
        f = str(f).lower().lstrip(".")
        f = "tiff" if f in ("tif", "tiff") else f
        if f not in fmts:
            fmts.append(f)

    for f in fmts:
        out = p.with_suffix(f".{f}")
        kw = dict(dpi=SAVE_DPI)
        if f == "tiff":
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(str(out), **kw)
        w_in, h_in = fig.get_size_inches()
        note = (f"  [{int(w_in * SAVE_DPI)}×{int(h_in * SAVE_DPI)} px "
                f"@ {SAVE_DPI} dpi]" if f in ("png", "tiff") else "")
        print(f"  ✓ {f.upper():4s} → {out}"
              f"  ({w_in * 25.4:.0f}×{h_in * 25.4:.0f} mm){note}")

    if GRAY_PREVIEW:
        try:
            from PIL import Image
            src = p.with_suffix(".png")
            if not src.exists():
                fig.savefig(str(src), dpi=300)
            gout = p.parent / (p.stem + "_gray.png")
            Image.open(src).convert("L").save(gout)
            print(f"  ✓ GRAY → {gout}  （自查黑白印刷可读性）")
        except Exception as e:
            print(f"  ⚠ 灰度预览失败：{e}")


# ═════════════════════════════════════════════════════════════
#  PART 4 —— Table 1：Benchmark
# ═════════════════════════════════════════════════════════════
def make_table1():
    """基线对比数据导出为 table1_benchmark.csv + .md（后者可直接贴入 Word）。

    数据来源：
      · model_comparison_matrix/comparison_matrix_full.csv —— 6 基线 × 3 特征集
      · multiseed_C/multiseed_C_summary.csv                —— 本方法 Full (5-seed)
      · multiseed_B/multiseed_B_summary.csv                —— 本方法 +Activity (5-seed)
    """
    print("\n" + "=" * 60)
    print("  生成 Table 1: Benchmark ...")
    print("=" * 60)

    df_comp = pd.read_csv(
        BASE / "model_comparison_matrix/comparison_matrix_full.csv")
    df_ms_c = pd.read_csv(BASE / "multiseed_C/multiseed_C_summary.csv")
    df_ms_b = pd.read_csv(BASE / "multiseed_B/multiseed_B_summary.csv")

    MODEL_ORDER = ["Ridge", "Lasso", "XGBoost", "LightGBM",
                   "GatedMLP", "FT-Transformer", "ours"]
    FS_COLS = ["lm_only", "lm_act", "full"]
    FS_LABEL = {"lm_only": "LM-only", "lm_act": "+Activity", "full": "Full"}

    def _mean_std(df):
        if "pearson_mean" in df.columns:
            return (float(df["pearson_mean"].iloc[0]),
                    float(df["pearson_std"].iloc[0]))
        return float(df["pearson"].mean()), float(df["pearson"].std(ddof=1))

    c_mean, c_std = _mean_std(df_ms_c)
    b_mean, b_std = _mean_std(df_ms_b)
    for fset, (mu, sd) in {"full": (c_mean, c_std),
                           "lm_act": (b_mean, b_std)}.items():
        mask = (df_comp["model"] == "ours") & (df_comp["feature_set"] == fset)
        if mask.any():
            df_comp.loc[mask, "pearson"] = mu
            df_comp.loc[mask, "pearson_std"] = sd
        else:
            df_comp = pd.concat([df_comp, pd.DataFrame([{
                "model": "ours", "feature_set": fset,
                "pearson": mu, "pearson_std": sd}])], ignore_index=True)

    pivot_m = (df_comp.groupby(["model", "feature_set"])["pearson"]
               .mean().unstack("feature_set")
               .reindex(index=MODEL_ORDER, columns=FS_COLS))
    if "pearson_std" in df_comp.columns:
        pivot_s = (df_comp.groupby(["model", "feature_set"])["pearson_std"]
                   .mean().unstack("feature_set")
                   .reindex(index=MODEL_ORDER, columns=FS_COLS))
    else:
        pivot_s = pd.DataFrame(np.nan, index=MODEL_ORDER, columns=FS_COLS)

    # FUN-PROSE 参考（见正文 §3.1）
    FP_BIND_MEAN, FP_BIND_STD = 0.7897, 0.0068
    FP_EXPR_MEAN, FP_EXPR_STD = 0.5961, 0.0188

    # ─ CSV（机器可读完整版）
    rows = []
    for m in MODEL_ORDER:
        row = {"Model": "Ours" if m == "ours" else m}
        for fset in FS_COLS:
            row[f"{FS_LABEL[fset]}_mean"] = pivot_m.loc[m, fset]
            row[f"{FS_LABEL[fset]}_std"] = pivot_s.loc[m, fset]
        rows.append(row)
    for name, mu, sd in [
        ("FUN-PROSE (binding, steelman upper bound)", FP_BIND_MEAN, FP_BIND_STD),
        ("FUN-PROSE (expression, faithful to original)", FP_EXPR_MEAN, FP_EXPR_STD),
    ]:
        rows.append({"Model": name,
                     "LM-only_mean": np.nan, "LM-only_std": np.nan,
                     "+Activity_mean": np.nan, "+Activity_std": np.nan,
                     "Full_mean": mu, "Full_std": sd})
    csv_path = OUT_DIR / "table1_benchmark.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  ✓ CSV → {csv_path}")

    # ─ Markdown（直接粘贴到 Word）
    def _fmt(mu, sd):
        if not np.isfinite(mu):
            return "—"
        return f"{mu:.4f} ± {sd:.4f}" if np.isfinite(sd) and sd > 0 else f"{mu:.4f}"

    md = ["# Table 1. Benchmark comparison: this method vs. 6 baselines × 3 feature sets",
          "",
          "| Model | LM-only | +Activity | Full |",
          "|---|---|---|---|"]
    for m in MODEL_ORDER:
        lbl = "**Ours**" if m == "ours" else m
        cells = [lbl]
        for fset in FS_COLS:
            txt = _fmt(pivot_m.loc[m, fset], pivot_s.loc[m, fset])
            if m == "ours" and fset == "full":
                txt = f"**{txt}**"
            cells.append(txt)
        md.append("| " + " | ".join(cells) + " |")
    md += ["",
           ("Note: All values are test Pearson r on the main holdout "
            "(chr15/16, n = 1,108). Our method's Full and +Activity columns are "
            "5-seed paired means ± std over seeds [42, 123, 456, 789, 2024]. "
            "Baselines use the same seed protocol as released in "
            "comparison_matrix_full.csv. For reference, FUN-PROSE (binding mode, "
            f"steelman upper bound): {FP_BIND_MEAN:.4f} ± {FP_BIND_STD:.4f}; "
            "FUN-PROSE (expression mode, faithful to original): "
            f"{FP_EXPR_MEAN:.4f} ± {FP_EXPR_STD:.4f}. "
            "See Additional file 1: §S1 for FUN-PROSE reproduction details.")]
    md_path = OUT_DIR / "table1_benchmark.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"  ✓ MD  → {md_path}")
    print("\n  ── 复制下方表格到 Word（表 1）──\n")
    print("\n".join(md))
    print("\n  Table 1 done.")


# ═════════════════════════════════════════════════════════════
#  PART 5 —— Fig 2：Architecture Ablation
# ═════════════════════════════════════════════════════════════
def make_fig2():
    """Fig 2，四面板。

    面板布置：
      左上 A —— 消融链 D→L→B→T→CITRA
      右上 B —— 2×2 因子分解热图
      左下 C —— fusion variant Δ（3 holdout 分组条形）
      右下 D —— 4 变体 × 3 holdout 一致性热图

    B 与 C 相对上一版已互换位置，字母按阅读顺序 A→B→C→D 重新指派，
    故 Word 中 Fig 2 的 figure legend 需同步对调 (B) 与 (C) 两段描述。
    """
    print("\n" + "=" * 60)
    print("  生成 Fig 2: Architecture Ablation ...")
    print("=" * 60)

    p1 = pd.read_csv(BASE / "paper_protocol1_single_seed42.csv")
    p2 = pd.read_csv(BASE / "paper_protocol2_paired_multiseed.csv")
    fa_m = pd.read_csv(BASE / "fusion_ablation/fusion_ablation_summary.csv")
    fa_34 = pd.read_csv(CHR34 / "fusion_ablation/fusion_ablation_summary.csv")
    fa_12 = pd.read_csv(CHR12 / "fusion_ablation/fusion_ablation_summary.csv")

    CHAIN = ["D", "L", "B", "T", "C"]
    # 单行短名：122 mm 版心下每个面板仅约 61 mm，长标签必然压盖，
    # 各阶段完整含义在图注中说明。
    CHAIN_LBL = {"D": "D", "L": "L", "B": "B", "T": "T", "C": "CITRA"}
    VARIANTS = ["C_concat", "C_noCrossW", "C_noLM", "C_tss_weighted"]
    VAR_LBL = {"C_concat": "concat", "C_noCrossW": "noCrossW",
               "C_noLM": "noLM", "C_tss_weighted": "tss_w"}
    HOLDOUTS = [("Main\n(chr15/16)", fa_m, C["main"]),
                ("chr3/4", fa_34, C["chr3_4"]),
                ("chr1/2", fa_12, C["chr1_2"])]

    def _p1(model):
        row = p1[p1["model"] == model]
        if row.empty:
            return np.nan
        col = _col(row, ["pearson_seed42", "pearson"])
        return float(row[col].values[0]) if col else np.nan

    def _p2(model):
        row = p2[p2["model"] == model]
        if row.empty:
            return np.nan, np.nan
        mu = (float(row["pearson_mean"].values[0])
              if "pearson_mean" in row.columns else np.nan)
        sd = (float(row["pearson_std"].values[0])
              if "pearson_std" in row.columns else 0.0)
        return mu, sd

    def _fa_delta(fa_df, variant):
        row = fa_df[fa_df["variant"] == variant]
        if row.empty:
            return np.nan, np.nan
        d_col = _col(row, ["delta_vs_C_mean", "delta_mean"])
        s_col = _col(row, ["delta_vs_C_std", "delta_std"])
        d = float(row[d_col].values[0]) if d_col else np.nan
        s = float(row[s_col].values[0]) if s_col else 0.0
        return d, s

    fig = plt.figure(figsize=FIG_MAIN, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.0])
    ax_chain = fig.add_subplot(gs[0, 0])    # A
    ax_2x2 = fig.add_subplot(gs[0, 1])      # B（原 C）
    ax_var = fig.add_subplot(gs[1, 0])      # C（原 B）
    ax_cons = fig.add_subplot(gs[1, 1])     # D

    # ─ Panel A：消融链
    ax = ax_chain
    x = np.arange(len(CHAIN))
    for i, m in enumerate(CHAIN):
        col = C[m]
        r1 = _p1(m)
        r2, r2s = _p2(m)
        if np.isfinite(r1):
            ax.bar(i, r1, width=0.63, color=col, alpha=0.28,
                   edgecolor="none", zorder=2)
        if np.isfinite(r2):
            ax.bar(i, r2, width=0.55, color=col, alpha=0.92,
                   edgecolor="white", linewidth=0.4, zorder=3)
            if r2s > 0:
                ax.errorbar(i, r2, yerr=r2s, fmt="none", ecolor="#333333",
                            elinewidth=1.2, capsize=3, capthick=1.2, zorder=4)
            # 柱顶四位小数在 5 柱面板内相邻很近，取字号下限以保住间隙
            ax.text(i, r2 + (r2s if r2s > 0 else 0) + 0.018, f"{r2:.4f}",
                    ha="center", fontsize=fs(5.5), color=col, fontweight="bold")
        elif np.isfinite(r1):
            ax.text(i, r1 + 0.018, f"{r1:.4f}", ha="center",
                    fontsize=fs(5.5), color=col, fontweight="bold")
    # Δ 箭头：交错两行，避免与柱顶数值和相邻标注互相接触
    for i in range(1, len(CHAIN)):
        rp = _p2(CHAIN[i - 1])[0]
        if not np.isfinite(rp):
            rp = _p1(CHAIN[i - 1])
        rc = _p2(CHAIN[i])[0]
        if not np.isfinite(rc):
            rc = _p1(CHAIN[i])
        if np.isfinite(rp) and np.isfinite(rc):
            lad_y = 0.90 if (i % 2) else 0.99
            ax.annotate("", xy=(i - 0.08, lad_y), xytext=(i - 0.92, lad_y),
                        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.7))
            ax.text(i - 0.5, lad_y + 0.015, f"{rc - rp:+.3f}",
                    ha="center", va="bottom", fontsize=fs(6.5), color="#444444")
    ax.set_xticks(x)
    ax.set_xticklabels([CHAIN_LBL[m] for m in CHAIN], fontsize=fs(7))
    ax.set_ylabel("Test Pearson r", fontsize=fs(7))
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.20, lw=0.5)
    despine(ax)
    label_panel(ax, "A")

    # ─ Panel B：2×2 因子分解（上一版的 Panel C）
    ax = ax_2x2
    vals_2x2 = np.array([[0.8185, 0.8247],
                         [0.8469, 0.8482]])
    row_lbl = ["tss_w\nencoder", "learnable\nencoder"]
    col_lbl = ["concat\nfusion", "residual\ncross-modal"]
    cmap_c = LinearSegmentedColormap.from_list(
        "c2x2", ["#fff7ec", "#fc8d59", "#7F0000"])
    im = ax.imshow(vals_2x2, cmap=cmap_c, vmin=0.810, vmax=0.855, aspect="auto")
    for i in range(2):
        for j in range(2):
            v = vals_2x2[i, j]
            ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                    fontsize=fs(8), fontweight="bold",
                    color="white" if v > 0.838 else "#222222")
    ax.set_xticks(range(2))
    ax.set_xticklabels(col_lbl, fontsize=fs(7))
    ax.set_yticks(range(2))
    ax.set_yticklabels(row_lbl, fontsize=fs(7))
    cbar = plt.colorbar(im, ax=ax, shrink=0.65, pad=0.04, aspect=15)
    cbar.set_label("Test Pearson r", fontsize=fs(7))
    cbar.ax.tick_params(labelsize=fs(6.5))
    despine(ax)
    label_panel(ax, "B", x=-0.26)

    # ─ Panel C：fusion variant Δ（上一版的 Panel B）
    ax = ax_var
    n_v, n_h = len(VARIANTS), len(HOLDOUTS)
    total_w = 0.72
    bar_w = total_w / n_h
    offsets = np.linspace(-total_w / 2 + bar_w / 2,
                          total_w / 2 - bar_w / 2, n_h)
    x_v = np.arange(n_v)
    for hi, (hlbl, hdf, hcol) in enumerate(HOLDOUTS):
        ds, es = zip(*[_fa_delta(hdf, v) for v in VARIANTS])
        ds, es = np.array(ds), np.array(es)
        ax.bar(x_v + offsets[hi], ds, width=bar_w * 0.88, color=hcol,
               alpha=0.85, edgecolor="white", linewidth=0.4,
               hatch=HATCH_3[hi], label=hlbl, zorder=3)
        valid = np.isfinite(ds) & np.isfinite(es)
        if valid.any():
            ax.errorbar((x_v + offsets[hi])[valid], ds[valid], yerr=es[valid],
                        fmt="none", ecolor="#333333", elinewidth=0.8,
                        capsize=2, capthick=0.8, zorder=4)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x_v)
    ax.set_xticklabels([VAR_LBL[v] for v in VARIANTS], fontsize=fs(6.5),
                       rotation=25, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Δ Pearson (CITRA − variant)", fontsize=fs(7))
    ax.set_ylim(-0.025, 0.10)
    # 显式两位小数刻度：自动刻度会给出 0.025 步长的 5 字符标签，
    # 放大字号后会把 ylabel 挤出画布
    ax.set_yticks(np.arange(-0.02, 0.101, 0.02))
    ax.set_yticklabels([f"{v:.2f}" for v in np.arange(-0.02, 0.101, 0.02)])
    ax.legend(fontsize=fs(6.5), ncol=1, loc="upper left",
              title="holdout", title_fontsize=fs(6.5))
    ax.grid(axis="y", alpha=0.20, lw=0.5)
    despine(ax)
    label_panel(ax, "C", x=-0.20, y=1.13)

    # ─ Panel D：3-holdout 一致性热图
    ax = ax_cons
    hnames = ["chr15/16", "chr3/4", "chr1/2"]
    delta_mat = np.full((len(VARIANTS), 3), np.nan)
    for hi, (_, hdf, _) in enumerate(HOLDOUTS):
        for vi, v in enumerate(VARIANTS):
            delta_mat[vi, hi] = _fa_delta(hdf, v)[0]
    norm_d = TwoSlopeNorm(vmin=-0.015, vcenter=0.0, vmax=0.075)
    cmap_d = LinearSegmentedColormap.from_list(
        "diverg", [C["chr3_4"], "#FFFFFF", C["Q2"]])
    im_d = ax.imshow(delta_mat, cmap=cmap_d, norm=norm_d, aspect="auto")
    for vi in range(len(VARIANTS)):
        for hi in range(3):
            v = delta_mat[vi, hi]
            if np.isfinite(v):
                ax.text(hi, vi, f"{v:+.3f}", ha="center", va="center",
                        fontsize=fs(6.5), fontweight="bold",
                        color="white" if abs(v) > 0.04 else "#222222")
    ax.set_xticks(range(3))
    ax.set_xticklabels(hnames, fontsize=fs(6.5))
    ax.set_yticks(range(len(VARIANTS)))
    ax.set_yticklabels([VAR_LBL[v] for v in VARIANTS], fontsize=fs(7))
    cbar_d = plt.colorbar(im_d, ax=ax, shrink=0.65, pad=0.04, aspect=15)
    cbar_d.set_label("Δ Pearson (CITRA − variant)", fontsize=fs(6.5))
    cbar_d.ax.tick_params(labelsize=fs(6.5))
    despine(ax)
    label_panel(ax, "D", x=-0.26)

    _save(fig, OUT_DIR / "fig2_ablation")
    plt.close(fig)
    print("  Fig 2 done.")


# ═════════════════════════════════════════════════════════════
#  PART 6 —— Fig 3：Biological Signatures
# ═════════════════════════════════════════════════════════════
# Panel C 中需要在纵轴刻度上高亮的 TF：颜色与散点四象限配色一致，
# 橙 = Q2（trans-repressed 端），绿 = Q3（trans-activated 端）。
TF_HIGHLIGHT = {"STE12": C["Q2"], "RAP1": C["Q3"]}


def make_fig3():
    """Fig 3，三面板。

      Panel A（上排跨栏）—— 四象限散点图
      Panel B（下排左）  —— 3-holdout 系统性偏置并排条形
      Panel C（下排右）  —— Q2_HC/Q3_HC 差分注意力 Top-TF
    """
    print("\n" + "=" * 60)
    print("  生成 Fig 3: Biological Signatures ...")
    print("=" * 60)

    gq = pd.read_csv(BASE / "analysis/quadrant_gene_list.csv")
    hd = pd.read_csv(BASE / "analysis/HC_Q2vsQ3_TF_attn_differential.csv")

    QUADS = ["Q1: Sequence-concordant (high)", "Q2: Trans-repressed",
             "Q3: Trans-activated", "Q4: Sequence-concordant (low)"]
    QSHORT = ["Q1", "Q2", "Q3", "Q4"]
    QCOLS = [C["Q1"], C["Q2"], C["Q3"], C["Q4"]]

    fig = plt.figure(figsize=FIG_FIG3, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15],
                          width_ratios=[0.94, 1.06])
    ax_scatter = fig.add_subplot(gs[0, :])
    ax_bias = fig.add_subplot(gs[1, 0])
    ax_tf = fig.add_subplot(gs[1, 1])

    # ─ Panel A：四象限散点图
    ax = ax_scatter
    test = gq[gq["split"] == "test"].copy()
    x_col = "seq_only_pred" if "seq_only_pred" in test.columns else "seq_activity"
    test["_x"] = test[x_col] - float(test[x_col].median())
    for qi, (q_full, qs, qcol) in enumerate(zip(QUADS, QSHORT, QCOLS)):
        sub = test[test["quadrant"] == q_full]
        # marker 形状 + 颜色双重编码，黑白印刷下四象限仍可区分
        ax.scatter(sub["_x"], sub["tpm_norm"], c=qcol, s=3.0, alpha=0.40,
                   rasterized=True, linewidths=0, marker=MARKER_4[qi],
                   label=f"{qs} n={len(sub)}")
    ax.axhline(0, color=C["ref"], lw=0.6, ls="--", zorder=1)
    ax.axvline(0, color=C["ref"], lw=0.6, ls="--", zorder=1)
    for txt, xy in [("Q1", (0.94, 0.90)), ("Q2", (0.94, 0.04)),
                    ("Q3", (0.03, 0.62)), ("Q4", (0.03, 0.04))]:
        ax.text(*xy, txt, transform=ax.transAxes, fontsize=fs(7.5),
                fontweight="bold", color=QCOLS[QSHORT.index(txt)],
                alpha=0.7, ha="center", va="center")
    ax.set_xlabel("Seq-only pred (B, median-centred)", fontsize=fs(7))
    ax.set_ylabel("TPM z-score", fontsize=fs(7))
    ax.legend(fontsize=fs(6.5), loc="upper left", ncol=2, markerscale=2.5,
              handletextpad=0.2, bbox_to_anchor=(0.0, 1.02))
    despine(ax)
    label_panel(ax, "A", x=-0.06, y=1.02)

    # ─ Panel B：3-holdout Bias 并排条形
    ax = ax_bias
    BIAS_DATA = {
        "Q2": {"Main": (+0.571, 0.037), "chr3/4": (+0.672, 0.055),
               "chr1/2": (+0.620, 0.034)},
        "Q3": {"Main": (-0.256, 0.020), "chr3/4": (-0.219, 0.022),
               "chr1/2": (-0.219, 0.020)},
    }
    Q_BIAS_ORDER = ["Q2", "Q3"]
    HOLDOUT_STYLE = [("Main", C["main"]), ("chr3/4", C["chr3_4"]),
                     ("chr1/2", C["chr1_2"])]
    bar_w_c = 0.22
    x_c = np.arange(len(Q_BIAS_ORDER))
    offsets_c = np.array([-bar_w_c, 0, +bar_w_c])
    for hi, (hlbl, hcol) in enumerate(HOLDOUT_STYLE):
        bs = [BIAS_DATA[q][hlbl][0] for q in Q_BIAS_ORDER]
        es = [BIAS_DATA[q][hlbl][1] for q in Q_BIAS_ORDER]
        ax.bar(x_c + offsets_c[hi], bs, width=bar_w_c * 0.88, color=hcol,
               alpha=0.85, edgecolor="white", linewidth=0.4,
               hatch=HATCH_3[hi], label=hlbl, zorder=3)
        ax.errorbar(x_c + offsets_c[hi], bs, yerr=es, fmt="none",
                    ecolor="#333333", elinewidth=0.8, capsize=2.5,
                    capthick=0.8, zorder=4)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x_c)
    # 三行折行 + 降一档：两组标签在 57 mm 面板内不再相接
    ax.set_xticklabels(["Q2\ntrans-repressed\n(over-prediction)",
                        "Q3\ntrans-activated\n(under-prediction)"],
                       fontsize=fs(6))
    ax.set_ylabel("Prediction bias (mean ± std)", fontsize=fs(7))
    ax.legend(fontsize=fs(6.5), loc="upper right", ncol=1, frameon=False)
    ax.grid(axis="y", alpha=0.20, lw=0.5)
    despine(ax)
    label_panel(ax, "B", x=-0.20)

    # ─ Panel C：HC-Q2/Q3 差分注意力 Top-TF
    ax = ax_tf
    delta_col = _col(hd, ["delta_Q3_minus_Q2", "delta"])
    tf_col = _col(hd, ["tf", "TF", "tf_name"])
    if delta_col is None:
        # 回退：由两列注意力均值现算差分
        a2c = _col(hd, ["attn_Q2_HC", "mean_attn_Q2"])
        a3c = _col(hd, ["attn_Q3_HC", "mean_attn_Q3"])
        tf_col = tf_col or hd.columns[0]
        if a2c and a3c:
            hd = hd.assign(_delta=hd[a3c] - hd[a2c])
            delta_col = "_delta"

    if delta_col and tf_col:
        show = pd.concat([hd.nsmallest(8, delta_col).iloc[::-1],
                          hd.nlargest(7, delta_col)])
        deltas, tfs = show[delta_col].values, show[tf_col].values
    else:
        deltas, tfs = np.zeros(1), np.array(["N/A"])

    y_pos_d = np.arange(len(tfs))
    ax.barh(y_pos_d, deltas,
            color=[C["Q2"] if v < 0 else C["Q3"] for v in deltas],
            alpha=0.85, edgecolor="white", linewidth=0.3, height=0.65, zorder=3)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y_pos_d)
    ax.set_yticklabels([str(t)[:10] for t in tfs], fontsize=fs(7))

    # 关键 TF 的刻度标签着色：STE12 橙（Q2 端）、RAP1 绿（Q3 端）
    for tick, name in zip(ax.get_yticklabels(), tfs):
        key = str(name).strip().upper()
        for target, colr in TF_HIGHLIGHT.items():
            if key == target or key.startswith(target):
                tick.set_color(colr)
                tick.set_fontweight("bold")
                break

    ax.set_xlabel("Δ attention (Q3_HC − Q2_HC)", fontsize=fs(6.5))
    ax.locator_params(axis="x", nbins=5)   # 放大字号后减少刻度数，避免相接
    # 顶端留一条空带专门放方向引导，两条标签缩短后不再互相压盖；
    # Q2/Q3 的归属由颜色承担，完整表述见图注。
    ax.set_ylim(-0.8, len(tfs) + 0.3)
    ax.text(0.02, 0.99, "← repressors", transform=ax.transAxes,
            fontsize=fs(6.5), color=C["Q2"], va="top", ha="left",
            fontweight="bold")
    ax.text(0.98, 0.99, "activators →", transform=ax.transAxes,
            fontsize=fs(6.5), color=C["Q3"], va="top", ha="right",
            fontweight="bold")
    despine(ax)
    label_panel(ax, "C", x=-0.20)

    _save(fig, OUT_DIR / "fig3_biological")
    plt.close(fig)
    print("  Fig 3 done.")


# ═════════════════════════════════════════════════════════════
#  PART 7 —— Fig 4：c-axis Interpretability
# ═════════════════════════════════════════════════════════════
def make_fig4():
    """Fig 4，四面板。

      Panel A —— cross_weight c 的分布
      Panel B —— canonical growth / meiosis 基因的 c 对比
      Panel C —— c 与 GO 条目的 rank-biserial 相关（lollipop）
      Panel D —— c 与 Brauer 生长速率特征的相关（3 holdout）

    Panel B 的分组优先读 goc.py 产出的 GO_canonical_gene_signature.csv
    （已做 name→ORF 解析）；该文件缺失时才回落到按标准基因名匹配。
    """
    print("\n" + "=" * 60)
    print("  生成 Fig 4: c-axis Interpretability ...")
    print("=" * 60)

    iaa_df = pd.read_csv(BASE / "analysis/dirC_crossweight_iaa_summary.csv")
    go_mw_df = pd.read_csv(GO_DIR / "GO_mw_continuous_enrichment.csv")

    cw_col = _col(iaa_df, ["cross_weight", "c", "cross_w"])
    gene_col = _col(iaa_df, ["gene", "gene_name", "name"])

    # goc.py 实际写出的相关列名为 rank_biserial，须优先命中
    go_term = _col(go_mw_df, ["term", "name", "GO_term", "description", "Term"])
    go_r = _col(go_mw_df, ["rank_biserial", "spearman_r", "r", "correlation", "rho"])
    go_p = _col(go_mw_df, ["p_fdr", "fdr", "FDR", "pval", "p_value", "p_two", "p"])
    go_ns = _col(go_mw_df, ["namespace", "ns"])

    canon_path = GO_DIR / "GO_canonical_gene_signature.csv"
    if canon_path.exists():
        canon = pd.read_csv(canon_path)
        c_col = _col(canon, ["cross_weight_c", "cross_weight", "c"])
        cw_growth = (canon.loc[canon["group"] == "growth", c_col].dropna().values
                     if c_col else np.array([]))
        cw_meiosis = (canon.loc[canon["group"] == "sporulation", c_col].dropna().values
                      if c_col else np.array([]))
        print(f"  [Panel B] canonical CSV loaded  growth n={len(cw_growth)}  "
              f"sporulation n={len(cw_meiosis)}")
    else:
        print(f"  [Panel B] ⚠ {canon_path} 不存在，回退到标准名查找（可能为空）")
        GROWTH_GENES = ["TEF1", "RPL6A", "TEF2", "PGK1", "TPI1", "CDC19", "ENO2"]
        MEIOSIS_GENES = ["IME1", "REC8", "SPO11", "DIT1", "HOP1"]

        def _get_cw(gene_list):
            if gene_col and cw_col:
                return iaa_df[iaa_df[gene_col].isin(gene_list)][cw_col].dropna().values
            return np.array([])

        cw_growth = _get_cw(GROWTH_GENES)
        cw_meiosis = _get_cw(MEIOSIS_GENES)

    BRAUER_DATA = {"chr15/16": 0.439, "chr3/4": 0.230, "chr1/2": 0.620}

    fig = plt.figure(figsize=FIG_FIG4, constrained_layout=True)
    # 上下行各自独立布局：A/B 不受 C（GO 长标签）的列宽约束
    sfig_top, sfig_bot = fig.subfigures(2, 1, height_ratios=[0.95, 1.25],
                                        hspace=0.04)
    gs_top = sfig_top.add_gridspec(1, 2)
    gs_bot = sfig_bot.add_gridspec(1, 2, width_ratios=[1.75, 1.0])
    ax_a = sfig_top.add_subplot(gs_top[0, 0])
    ax_b = sfig_top.add_subplot(gs_top[0, 1])
    ax_c = sfig_bot.add_subplot(gs_bot[0, 0])
    ax_d = sfig_bot.add_subplot(gs_bot[0, 1])

    # ─ Panel A：cross_weight 分布
    ax = ax_a
    cw_vals = iaa_df[cw_col].dropna().values if cw_col else np.array([])
    if len(cw_vals) > 0:
        ax.hist(cw_vals, bins=50, color=C["C"], alpha=0.75,
                edgecolor="white", linewidth=0.3)
        mu = float(np.mean(cw_vals))
        p5 = float(np.percentile(cw_vals, 5))
        p95 = float(np.percentile(cw_vals, 95))
        ax.axvline(mu, color="#333333", lw=1.5, ls="-", label=f"mean={mu:.3f}")
        ax.axvline(p5, color=C["Q1"], lw=1.0, ls=":", label=f"p5={p5:.3f}")
        ax.axvline(p95, color=C["Q2"], lw=1.0, ls=":", label=f"p95={p95:.3f}")
        ax.set_xlabel("cross_weight  c", fontsize=fs(7))
        ax.set_ylabel("Count", fontsize=fs(7))
        ax.legend(fontsize=fs(6.5), loc="upper left", frameon=True,
                  framealpha=0.92, edgecolor="none", borderpad=0.3)
    despine(ax)
    label_panel(ax, "A")

    # ─ Panel B：canonical 基因分组（显式 ylim，保证标注全部内移）
    ax = ax_b
    rng = np.random.default_rng(42)
    for vals, col, yi in [(cw_growth, C["Q3"], 1), (cw_meiosis, C["Q1"], 0)]:
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(vals, np.full(len(vals), yi) + jitter, c=col, s=9,
                   alpha=0.85, edgecolors="white", linewidth=0.3,
                   marker=("o" if yi == 1 else "s"), zorder=3)
        med = float(np.median(vals))
        ax.plot([med, med], [yi - 0.18, yi + 0.18], color=col, lw=2.0, zorder=4)
        ax.text(med, yi - 0.30, f"med={med:.2f}", fontsize=fs(6.5), color=col,
                ha="center", va="top", fontweight="bold")
    if len(cw_growth) >= 3 and len(cw_meiosis) >= 3:
        _, p = mannwhitneyu(cw_growth, cw_meiosis, alternative="two-sided")
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
        ax.text(0.02, 0.98, f"MWU p = {p:.1e} {stars}", transform=ax.transAxes,
                fontsize=fs(6.5), color="#222222", ha="left", va="top",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=C["ref"], lw=0.5))
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Sporulation/\nMeiosis", "Growth/\nBiosynthesis"],
                       fontsize=fs(7))
    ax.set_xlabel("cross_weight  c", fontsize=fs(7))
    ax.set_xlim(0.25, 1.05)
    ax.set_ylim(-0.6, 1.55)
    ax.grid(axis="x", alpha=0.20, lw=0.5)
    despine(ax)
    label_panel(ax, "B", x=-0.22)

    # ─ Panel C：GO rank-biserial lollipop
    ax = ax_c
    if go_r and go_term and len(go_mw_df) > 0:
        go_plot = go_mw_df.dropna(subset=[go_r, go_term]).copy()
        go_plot["_r"] = pd.to_numeric(go_plot[go_r], errors="coerce")
        go_plot = go_plot.dropna(subset=["_r"])
        # 优先取 FDR<0.05 的显著条目，不足 6 条则回到全集
        if go_p:
            go_sig = go_plot[pd.to_numeric(go_plot[go_p], errors="coerce") < 0.05]
            if len(go_sig) >= 6:
                go_plot = go_sig
        # 优先 BP namespace，使解读更聚焦
        if go_ns:
            go_bp = go_plot[go_plot[go_ns].astype(str).str.lower()
                            .str.contains("biological_process|bp", na=False)]
            if len(go_bp) >= 6:
                go_plot = go_bp
        # 122 mm 版心下上下各 6 条即为行距上限，再多会互相压盖
        show = pd.concat([go_plot.nsmallest(6, "_r").iloc[::-1],
                          go_plot.nlargest(6, "_r")]).reset_index(drop=True)
        rs = show["_r"].values
        terms = show[go_term].values
        y_pos = np.arange(len(rs))
        loll_colors = [C["Q1"] if r > 0 else C["Q2"] for r in rs]
        ax.hlines(y_pos, 0, rs, color=loll_colors, alpha=0.65, lw=1.5)
        ax.scatter(rs, y_pos, c=loll_colors, s=20, zorder=4,
                   edgecolors="white", linewidth=0.4)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_yticks(y_pos)
        # 字号放大后可容纳的字符数下调至 26
        ax.set_yticklabels([str(t)[:26] + ("…" if len(str(t)) > 26 else "")
                            for t in terms], fontsize=fs(6.5))
        ax.set_xlabel("rank-biserial r (c vs GO)", fontsize=fs(7))
        # 方向引导放在两个空角
        ax.text(0.02, 0.99, "← low-c\n(meiosis)", transform=ax.transAxes,
                fontsize=fs(6.5), color=C["Q2"], va="top", ha="left",
                fontweight="bold")
        ax.text(0.98, 0.01, "high-c →\n(ribosome)", transform=ax.transAxes,
                fontsize=fs(6.5), color=C["Q1"], va="bottom", ha="right",
                fontweight="bold")
    else:
        ax.text(0.5, 0.5, f"GO MW columns:\n{list(go_mw_df.columns[:8])}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=fs(6.5))
    ax.grid(axis="x", alpha=0.20, lw=0.5)
    despine(ax)
    label_panel(ax, "C")

    # ─ Panel D：Brauer 3-holdout
    ax = ax_d
    h_names = list(BRAUER_DATA)
    r_vals = [BRAUER_DATA[h] for h in h_names]
    colors_d = [C["main"], C["chr3_4"], C["chr1_2"]]
    for i, (r, col) in enumerate(zip(r_vals, colors_d)):
        ax.bar(i, r, color=col, alpha=0.85, edgecolor="white", linewidth=0.4,
               width=0.55, hatch=HATCH_3[i], zorder=3)
        ax.text(i, r + 0.012, f"r={r:.3f}", ha="center", va="bottom",
                fontsize=fs(6.5), color=col, fontweight="bold")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(len(h_names)))
    ax.set_xticklabels(h_names, fontsize=fs(6.5), rotation=25, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("Pearson r (Brauer growth-up)", fontsize=fs(7))
    ax.set_ylim(-0.05, 0.90)
    ax.grid(axis="y", alpha=0.20, lw=0.5)
    despine(ax)
    label_panel(ax, "D", x=-0.22)

    _save(fig, OUT_DIR / "fig4_c_axis")
    plt.close(fig)
    print("  Fig 4 done.")


# ═════════════════════════════════════════════════════════════
#  PART 8 —— Fig S1：TF–TF Coordination (NMF)，补充材料
# ═════════════════════════════════════════════════════════════
def make_figS1():
    """Fig S1，五面板（补充材料版心 160 mm）。"""
    print("\n" + "=" * 60)
    print("  生成 Fig S1: TF–TF Coordination (NMF) ...")
    print("=" * 60)

    W_df = pd.read_csv(BASE / "analysis/fig5_tf_tf_nmf_loadings.csv", index_col=0)
    clu_df = pd.read_csv(BASE / "analysis/fig5_tf_tf_clusters.csv")
    M_bg_df = pd.read_csv(
        BASE / "analysis/fig5_tf_tf_coordination_bg_corrected.csv", index_col=0)
    # 未减背景的原始对称矩阵，仅 Panel E 的 top-15 使用
    M_raw_df = pd.read_csv(BASE / "analysis/fig5_tf_tf_coordination.csv", index_col=0)
    enr_df = pd.read_csv(BASE / "analysis/fig5_tf_tf_cluster_enrichment.csv")

    tf_names = list(W_df.index)
    N_TF, n_k = len(tf_names), W_df.shape[1]

    nmf_col = _col(clu_df, ["nmf_cluster", "nmf_label"])
    tf_col = _col(clu_df, ["tf", "TF", "tf_name"])

    if nmf_col and tf_col:
        order_df = clu_df.sort_values(nmf_col)
        ordered_tfs = [t for t in order_df[tf_col].tolist() if t in tf_names]
        ordered_tfs += [t for t in tf_names if t not in ordered_tfs]
    else:
        ordered_tfs = tf_names
    ord_idx = [tf_names.index(t) for t in ordered_tfs if t in tf_names]
    W_ord = W_df.iloc[ord_idx].values
    M_ord = M_bg_df.values.astype(float)[np.ix_(ord_idx, ord_idx)]

    if nmf_col and tf_col:
        clu_ordered = order_df[nmf_col].values[:len(ord_idx)]
        boundaries = np.where(np.diff(clu_ordered))[0] + 0.5
    else:
        boundaries = []

    fig = plt.figure(figsize=FIG_ESM, constrained_layout=True)
    # 上下行各自独立布局：D/E 不受上排 colorbar 结构约束
    sfig_top, sfig_bot = fig.subfigures(2, 1, height_ratios=[1.2, 1.3], hspace=0.04)
    gs_top = sfig_top.add_gridspec(1, 3, width_ratios=[1.0, 1.1, 1.8])
    gs_bot = sfig_bot.add_gridspec(1, 2, width_ratios=[3.0, 2.0])
    ax_a = sfig_top.add_subplot(gs_top[0, 0])
    ax_b = sfig_top.add_subplot(gs_top[0, 1])
    ax_c = sfig_top.add_subplot(gs_top[0, 2])
    ax_d = sfig_bot.add_subplot(gs_bot[0, 0])
    ax_e = sfig_bot.add_subplot(gs_bot[0, 1])

    # ─ Panel A：NMF loadings
    ax = ax_a
    W_norm = W_ord / (W_ord.max(axis=1, keepdims=True) + 1e-12)
    cmap_a = LinearSegmentedColormap.from_list(
        "nmf", ["#f7fbff", "#0072B2", "#00264d"])
    im_a = ax.imshow(W_norm, aspect="auto", cmap=cmap_a, vmin=0, vmax=1)
    for b in boundaries:
        ax.axhline(b, color="white", lw=0.6, alpha=0.8)
    ax.set_xticks(range(n_k))
    ax.set_xticklabels([f"k{j + 1}" for j in range(n_k)], fontsize=fs(7))
    ax.set_yticks([])
    ax.set_ylabel(f"TFs (n={N_TF})", fontsize=fs(7))
    cbar_a = plt.colorbar(im_a, ax=ax, shrink=0.6, pad=0.04, aspect=20)
    cbar_a.set_label("Norm. loading", fontsize=fs(6.5))
    cbar_a.ax.tick_params(labelsize=fs(6))
    despine(ax)
    label_panel(ax, "A", x=-0.06)

    # ─ Panel B：TF-TF 协调矩阵
    ax = ax_b
    im_b = ax.imshow(np.log1p(np.maximum(M_ord, 0) * 5e3),
                     aspect="auto", cmap=CMAP_COORD)
    for b in boundaries:
        ax.axhline(b, color="#009E73", lw=0.7, alpha=0.9)
        ax.axvline(b, color="#009E73", lw=0.7, alpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar_b = plt.colorbar(im_b, ax=ax, shrink=0.6, pad=0.03, aspect=20)
    cbar_b.set_label("log1p(coord.)", fontsize=fs(6.5))
    cbar_b.ax.tick_params(labelsize=fs(6))
    ax.text(0.02, 0.02,
            "Pairwise ARI: 0.413 / 0.332 / 0.504\n(mean 0.416 ± 0.070, exploratory)",
            transform=ax.transAxes, fontsize=fs(6), color="white", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="#00000099", ec="none"))
    despine(ax)
    label_panel(ax, "B", x=-0.03)

    # ─ Panel C：簇均衡性
    ax = ax_c
    if nmf_col and tf_col:
        counts = clu_df[nmf_col].value_counts().sort_index()
        uniform = N_TF / n_k
        ax.bar(counts.index, counts.values, color=C["B"], alpha=0.80,
               edgecolor="white", linewidth=0.3, width=0.7)
        ax.axhline(uniform, color=C["ref"], lw=1.0, ls="--",
                   label=f"uniform={uniform:.1f}")
        for xi, cnt in zip(counts.index, counts.values):
            ax.text(xi, cnt + 0.4, str(cnt), ha="center", fontsize=fs(6.5))
        ax.legend(fontsize=fs(6.5), loc="upper right", frameon=False)
    ax.set_xlabel("NMF cluster k", fontsize=fs(7.5))
    ax.set_ylabel("n TFs", fontsize=fs(7.5))
    ax.grid(axis="y", alpha=0.20, lw=0.5)
    despine(ax)
    label_panel(ax, "C", x=-0.10)

    # ─ Panel D：Cluster × known-complex enrichment
    ax = ax_d
    enr_sorted = (enr_df.sort_values("cluster")
                  if "cluster" in enr_df.columns else enr_df)
    clu_enr_col = _col(enr_sorted, ["cluster", "cluster_id"],
                       default=enr_sorted.columns[0])
    top_cplx_col = _col(enr_sorted, ["top_complex", "top_complex_name"])
    frac_col = _col(enr_sorted, ["top_complex_frac", "frac"])
    n_in_col = _col(enr_sorted, ["n_in", "n_total"])
    n_top_col = _col(enr_sorted, ["n_top_complex", "n_top"])

    if top_cplx_col and frac_col:
        palette = ["#E69F00", "#56B4E9", "#009E73", "#D55E00",
                   "#CC79A7", "#F0E442", "#0072B2", "#000000", "#888888"]
        cplx_color = {cplx: palette[i % len(palette)]
                      for i, cplx in enumerate(enr_sorted[top_cplx_col].unique())}
        for _, row in enr_sorted.iterrows():
            k_id = int(row[clu_enr_col])
            frac = float(row[frac_col])
            cplx = str(row[top_cplx_col])
            ax.bar(k_id, frac, color=cplx_color.get(cplx, "#888888"),
                   edgecolor="white", linewidth=0.3, alpha=0.85, label=cplx)
            n_in = int(row[n_in_col]) if n_in_col else ""
            n_top = int(row[n_top_col]) if n_top_col else ""
            ax.text(k_id, frac + 0.025,
                    f"{cplx[:10]}\n{n_top}/{n_in}" if n_in else cplx[:10],
                    ha="center", va="bottom", fontsize=fs(6))
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), fontsize=fs(6), ncol=4,
                  loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
        ax.set_xlabel("NMF cluster k", fontsize=fs(7.5))
        ax.set_ylabel("Top-complex fraction", fontsize=fs(7.5))
        ax.set_ylim(0, 1.30)
        ax.grid(axis="y", alpha=0.20, lw=0.5)
        despine(ax)
    else:
        ax.text(0.5, 0.5, "enrichment data unavailable", ha="center",
                va="center", transform=ax.transAxes, fontsize=fs(8))
        ax.axis("off")
    label_panel(ax, "D")

    # ─ Panel E：Top-15 协调对
    ax = ax_e
    ax.axis("off")
    M_sq = M_raw_df.reindex(index=tf_names, columns=tf_names).values.astype(float).copy()
    np.fill_diagonal(M_sq, 0)
    triu_i, triu_j = np.triu_indices(N_TF, k=1)
    vals_pairs = M_sq[triu_i, triu_j]
    top15_idx = np.argsort(vals_pairs)[::-1][:15]

    rows_txt = ["{:<8} {:<8} {:>7}".format("TF_a", "TF_b", "coord"), "─" * 27]
    for idx in top15_idx:
        ia, ib = triu_i[idx], triu_j[idx]
        rows_txt.append("{:<8} {:<8} {:>7.4f}".format(
            str(tf_names[ia])[:8], str(tf_names[ib])[:8], vals_pairs[idx]))
    ax.text(0.0, 0.97, "\n".join(rows_txt), transform=ax.transAxes,
            fontsize=fs(6.5), va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc=C["bg"], ec=C["ref"], lw=0.5))
    label_panel(ax, "E", x=-0.05)

    _save(fig, OUT_DIR / "figS1_tftf_nmf")
    plt.close(fig)
    print("  Fig S1 done.")


# ═════════════════════════════════════════════════════════════
#  PART 9 —— Alt Text（Springer Sect. 3，EU Accessibility Act / WCAG）
# ═════════════════════════════════════════════════════════════
ALT_TEXT = {
    "Fig. 2": (
        "Four-panel chart of architecture ablation. Panel A is a bar chart of "
        "test Pearson r across five model stages, rising from DNA-CNN through "
        "language-model and promoter-activity features to the TF branch and the "
        "full CITRA model. Panel B is a two-by-two heat map decomposing the gain "
        "into a learnable-TF-encoder effect and a residual-cross-modal-fusion "
        "effect. Panel C is a grouped bar chart of the Pearson drop caused by "
        "four fusion variants on three chromosome hold-out sets. Panel D is a "
        "heat map of the same drop for four variants across the three hold-out "
        "sets, all agreeing in sign."),
    "Fig. 3": (
        "Three-panel chart of biological signatures. Panel A is a scatter plot "
        "of measured expression against sequence-only prediction, with genes "
        "assigned to four quadrants. Panel B is a grouped bar chart showing that "
        "the over-prediction of trans-repressed genes and the under-prediction "
        "of trans-activated genes are reproduced on all three hold-out sets. "
        "Panel C is a horizontal bar chart ranking transcription factors by the "
        "difference in attention between the two high-confidence subsets, with "
        "STE12 at the negative extreme and RAP1 at the positive extreme."),
    "Fig. 4": (
        "Four-panel chart of the scalar fusion gate c. Panel A is a histogram of "
        "c on the primary test set, unimodal and shifted towards high values. "
        "Panel B is a strip plot contrasting c for canonical growth genes and "
        "canonical meiosis genes. Panel C is a lollipop chart of rank-biserial "
        "correlation between c and Gene Ontology membership, with ribosomal and "
        "translation terms at the high-c end and meiosis terms at the low-c end. "
        "Panel D is a bar chart of the correlation between c and an external "
        "growth-rate signature on the three hold-out sets."),
    "Fig. S1": (
        "Five-panel exploratory analysis of transcription-factor cooperation. "
        "Panel A is a heat map of non-negative matrix factorisation loadings. "
        "Panel B is a heat map of the background-corrected cooperation matrix "
        "with cluster boundaries marked. Panel C is a bar chart of cluster sizes "
        "compared with hierarchical clustering. Panel D is a bar chart of the "
        "fraction of each cluster accounted for by its top known complex. "
        "Panel E is a table of the fifteen highest-scoring cooperation pairs."),
}


def write_alt_text(keys):
    """导出 Alt Text 初稿到 figures_chip/alt_text.md。"""
    out = OUT_DIR / "alt_text.md"
    lines = ["# Alternative text (Alt Text) for figures", "",
             "Springer Computer Science Proceedings, Sect. 3 "
             "(EU Accessibility Act / WCAG).",
             "Each entry describes what the figure shows, without repeating "
             "the caption verbatim.", ""]
    for k in keys:
        if k in ALT_TEXT:
            lines += [f"## {k}", "", ALT_TEXT[k], ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ ALT  → {out}")


# ═════════════════════════════════════════════════════════════
#  PART 10 —— main
# ═════════════════════════════════════════════════════════════
TASKS = {
    "1": ("Table 1", "table1_benchmark", make_table1),
    "2": ("Fig 2", "fig2_ablation", make_fig2),
    "3": ("Fig 3", "fig3_biological", make_fig3),
    "4": ("Fig 4", "fig4_c_axis", make_fig4),
    "S1": ("Fig S1", "figS1_tftf_nmf", make_figS1),
}
ORDER = ["1", "2", "3", "4", "S1"]


def main():
    global SAVE_FORMATS, SAVE_DPI, GRAY_PREVIEW

    parser = argparse.ArgumentParser(
        description="生成 Springer LNCS/CCIS 规范主图 Fig 2-4 + 补图 S1 + 表 1")
    parser.add_argument("--fig", nargs="*", type=str, default=ORDER,
                        help="要生成的编号，默认全部。1=Table 1，2/3/4=主图，S1=补充图。")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"],
                        choices=["pdf", "eps", "tiff", "png", "svg"],
                        help="输出格式，默认 PDF + PNG（仅作用于图；表 1 恒为 CSV+MD）。")
    parser.add_argument("--dpi", type=int, default=1200,
                        help="栅格输出分辨率，仅作用于 png/tiff，默认 1200。")
    parser.add_argument("--gray-preview", action="store_true",
                        help="额外导出 *_gray.png，自查黑白印刷可读性。")
    args = parser.parse_args()

    apply_springer_style()
    SAVE_FORMATS = tuple(args.formats)
    SAVE_DPI = args.dpi
    GRAY_PREVIEW = args.gray_preview

    print("=" * 60)
    print("  Springer LNCS / CCIS 主图生成")
    print(f"  输出格式: {' + '.join(f.upper() for f in SAVE_FORMATS)}"
          f"   栅格 dpi: {SAVE_DPI}")
    print(f"  设计宽度: 正文 {TEXT_W_MM:.0f} mm | 补充材料 {ESM_W_MM:.0f} mm（均 1:1）")
    print(f"  图高:     Fig 2 {H_MAIN_MM:.0f} | Fig 3 {H_FIG3_MM:.0f} | "
          f"Fig 4 {H_FIG4_MM:.0f} | Fig S1 {H_ESM_MM:.0f} mm")
    print(f"  字号:     缩放 ×{FS_SCALE:.2f}，下限 {FS_MIN} pt")
    print("=" * 60)

    def _norm(t):
        s = str(t).strip().upper()
        return s if s.startswith("S") else s.lstrip("0")

    requested = []
    for f in args.fig:
        nf = _norm(f)
        if nf not in TASKS:
            print(f"  ⚠ 未知编号 {f}（应为 1 / 2 / 3 / 4 / S1），跳过")
        elif nf not in requested:
            requested.append(nf)

    results = {}
    for key in ORDER:
        if key not in requested:
            continue
        label, _, func = TASKS[key]
        try:
            func()
            results[key] = "✅"
        except Exception as e:
            results[key] = f"❌ {e}"
            print(f"\n  ⚠ {label} 失败:")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("  生成结果汇总")
    print("=" * 60)
    for key in ORDER:
        if key not in results:
            continue
        label, prefix, _ = TASKS[key]
        files = sorted(OUT_DIR.glob(f"{prefix}.*"))
        out_name = (f"{prefix}.{{{','.join(p.suffix.lstrip('.') for p in files)}}}"
                    if files else "(未找到输出)")
        print(f"  {label:<8} {results[key]}  →  {out_name}")

    all_ok = results and all(v == "✅" for v in results.values())
    print(f"\n  {'✅ 全部成功' if all_ok else '⚠ 部分失败，请检查上方错误信息'}")

    fig_keys = [f"Fig. {k}" for k in ("2", "3", "4", "S1") if k in results]
    if fig_keys:
        write_alt_text(fig_keys)

    print(f"\n  输出目录: {OUT_DIR.resolve()}")
    print(f"\n  ℹ 使用说明："
          f"\n    · Word 中插入 PNG 后把图宽设为 {TEXT_W_MM:.0f} mm"
          f"（补充材料 {ESM_W_MM:.0f} mm），此时为 1:1，"
          f"图内最小字号不低于 {FS_MIN} pt。"
          f"\n    · 图注置于图下方、9 pt；表注置于表上方。"
          f"\n    · .pdf 为矢量主文件，随源文件一并提交。"
          f"\n    · alt_text.md 为替代文本初稿，按 Sect. 3 备查。"
          f"\n    · 表 1 的 .md 可直接粘贴到 Word 替换占位表格。")


if __name__ == "__main__":
    main()
