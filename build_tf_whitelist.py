#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_tf_whitelist.py
================================================================
构造筛 TF 后重训用的白名单 —— 三源并集 + 复合体补完 - 有害剔除

逻辑 (按优先级)
------------
源 1 (因果 / ΔMSE 主要):   per_tf_ablation_ranking.csv 的 top-N (默认 60)
源 2 (HC Q2/Q3 标注):       HC 分析发现的 Q2/Q3 relevant TFs
源 3 (复合体补完):          若某复合体 ≥1 个成员已入选, 补齐其它未被硬剔除成员

硬剔除规则
----------
ΔMSE < HARD_EXCLUDE_THRESHOLD (默认 -0.001) 的 TF 直接踢出
—— 这些"越加越糟"的 TF 即使在复合体里也不留, 避免注入已知噪声

输入
----
  result/analysis/per_tf_ablation_ranking.csv    (per_tf_ablation_ranking.py 产出)

输出
----
  result/analysis/tf_whitelist.csv         最终白名单 (带来源标注和 ΔMSE)
  result/analysis/tf_excluded.csv          被硬剔除的 TF 列表 (带原因)
  result/analysis/tf_whitelist_summary.txt 人类可读的构造报告

用法
----
  cd ~/projects/Yeast
  python3 build_tf_whitelist.py
  # 或自定义阈值
  TOP_N=50 HARD_EXCLUDE_THRESHOLD=-0.0005 python3 build_tf_whitelist.py
"""

from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================================
# ★ 配置
# ============================================================================
PATH_CONFIG = dict(
    ablation_csv = "./data/analysis/per_tf_ablation_ranking.csv",
    out_dir      = "./data/analysis",
)

TOP_N                   = int(os.environ.get("TOP_N", 60))
HARD_EXCLUDE_THRESHOLD  = float(os.environ.get("HARD_EXCLUDE_THRESHOLD", -0.001))

# HC Q2/Q3 标注 —— 和 per_tf_ablation_ranking.py 里的 Q{2,3}_HC_TOP 一致
HC_Q2_TOP = {"SPN1", "EAF6", "EAF5", "EPL1", "RIF1", "DIG1", "TOS8"}
HC_Q3_TOP = {"THI3", "HAP4", "RBA50", "SUM1", "OPI1", "PPR1", "SFP1"}

# 已知酵母转录复合体 —— 和 new40.py / diagnose_input_tf_redundancy.py 一致
KNOWN_TF_COMPLEXES = {
    'SAGA':        ['GCN5', 'SPT3', 'SPT7', 'SPT8', 'SPT20', 'ADA2', 'ADA3',
                    'NGG1', 'TAF5', 'TAF6', 'TAF9', 'TAF10', 'TAF12', 'SGF29',
                    'SGF73', 'UBP8', 'SUS1', 'TRA1'],
    'NuA4/HAT':    ['ESA1', 'EPL1', 'YNG2', 'EAF1', 'EAF3', 'EAF5', 'EAF6',
                    'EAF7', 'VID21', 'YAF9', 'TRA1'],
    'Mediator':    ['MED1', 'MED2', 'MED4', 'MED6', 'MED7', 'MED8', 'MED11',
                    'MED14', 'MED15', 'MED17', 'MED18', 'MED19', 'MED20',
                    'MED21', 'MED22', 'MED31', 'SRB2', 'SRB4', 'SRB5', 'SRB6',
                    'CDK8', 'CYC8', 'SSN2', 'SSN3', 'SSN8'],
    'TFIID':       ['TAF1', 'TAF2', 'TAF3', 'TAF4', 'TAF5', 'TAF6', 'TAF7',
                    'TAF8', 'TAF9', 'TAF10', 'TAF11', 'TAF12', 'TAF13', 'TAF14',
                    'SPT15', 'TBP'],
    'Ribi/TOR':    ['SFP1', 'IFH1', 'FHL1', 'RAP1', 'HMO1', 'ABF1', 'CRF1'],
    'Stress':      ['MSN2', 'MSN4', 'YAP1', 'SKN7', 'HSF1', 'GIS1', 'RPH1',
                    'XBP1'],
    'Carbon/Resp': ['HAP1', 'HAP2', 'HAP3', 'HAP4', 'HAP5', 'CAT8', 'SIP4',
                    'ADR1', 'MIG1', 'MIG2', 'RGT1', 'GAL4', 'GAL80'],
    'Cell-cycle':  ['MBP1', 'SWI4', 'SWI6', 'FKH1', 'FKH2', 'NDD1', 'MCM1',
                    'ACE2', 'SWI5', 'HCM1', 'STB1', 'XBP1'],
}


def main():
    PC = PATH_CONFIG
    out = Path(PC["out_dir"]); out.mkdir(parents=True, exist_ok=True)

    # ── 1. 加载消融结果 ─────────────────────────────────────────────────────
    abl_path = Path(PC["ablation_csv"])
    if not abl_path.exists():
        print(f"[致命] 找不到 {abl_path}"); print(f"       先跑 per_tf_ablation_ranking.py"); sys.exit(1)

    df = pd.read_csv(abl_path)
    df["tf_upper"] = df["tf_name"].str.upper()
    all_tfs_u     = set(df["tf_upper"])
    tf2dmse       = dict(zip(df["tf_upper"], df["delta_mse"]))
    n_total       = len(df)
    print(f"[输入] 加载 {abl_path} — {n_total} 个 TF")
    print(f"[参数] TOP_N={TOP_N}  HARD_EXCLUDE_THRESHOLD={HARD_EXCLUDE_THRESHOLD}")

    # ── 2. 硬剔除集合 ─────────────────────────────────────────────────────────
    hard_exclude = set(df.loc[df["delta_mse"] < HARD_EXCLUDE_THRESHOLD, "tf_upper"])
    print(f"\n[硬剔除] ΔMSE < {HARD_EXCLUDE_THRESHOLD} 的 TF: {len(hard_exclude)}")
    worst15 = df.nsmallest(15, "delta_mse")[["tf_name", "delta_mse"]]
    print("  最差 15 个 (供审查):")
    for _, r in worst15.iterrows():
        flag = " ✗剔除" if r["tf_name"].upper() in hard_exclude else ""
        print(f"    {r['tf_name']:<14} ΔMSE={r['delta_mse']:+.5f}{flag}")

    # ── 3. 源 1: ΔMSE Top-N ─────────────────────────────────────────────────
    src1 = set(df.nlargest(TOP_N, "delta_mse")["tf_upper"])
    src1_filtered = src1 - hard_exclude
    print(f"\n[源 1] ΔMSE Top-{TOP_N}: {len(src1)} → 硬剔除后 {len(src1_filtered)}")

    # ── 4. 源 2: HC Q2/Q3 ────────────────────────────────────────────────────
    hc_all       = HC_Q2_TOP | HC_Q3_TOP
    hc_in_data   = hc_all & all_tfs_u
    hc_missing   = hc_all - all_tfs_u
    src2_filtered = hc_in_data - hard_exclude
    print(f"\n[源 2] HC Q2/Q3 共 {len(hc_all)} 个 — 数据中存在 {len(hc_in_data)} 个  "
          f"→ 硬剔除后 {len(src2_filtered)}")
    if hc_missing:
        print(f"  ⚠ 不在 ChEC-seq 数据里的: {sorted(hc_missing)}")

    # 合并源 1 + 源 2 作为"锚点集合"
    anchor = src1_filtered | src2_filtered
    print(f"\n[锚点集合 = 源1 ∪ 源2] {len(anchor)} 个 TF")

    # ── 5. 源 3: 复合体补完 ─────────────────────────────────────────────────
    print(f"\n[源 3] 复合体补完 (≥1 成员已入选 → 补齐其余非剔除成员)")
    complex_members_added = set()
    complex_membership    = {}                    # tf_upper -> list of complex names
    for cplx_name, members in KNOWN_TF_COMPLEXES.items():
        m_upper = {m.upper() for m in members} & all_tfs_u        # 只看数据里有的
        in_anchor = m_upper & anchor
        if len(in_anchor) == 0:                                    # 无 anchor, 不触发补完
            continue
        to_add = (m_upper - anchor) - hard_exclude                # 新增且非剔除
        # 记录复合体成员身份 (已入选的也标注)
        for t in (m_upper & anchor) | to_add:
            complex_membership.setdefault(t, []).append(cplx_name)
        print(f"  {cplx_name:<14}  data={len(m_upper):>2}  "
              f"anchor_hits={len(in_anchor):>2}  to_add={len(to_add):>2}  "
              f"(+{','.join(sorted(to_add)) if to_add else '–'})")
        complex_members_added |= to_add

    # ── 6. 最终并集 ──────────────────────────────────────────────────────────
    whitelist = anchor | complex_members_added
    print(f"\n[最终] 白名单: {len(whitelist)} 个 TF  "
          f"= 源1({len(src1_filtered)}) ∪ 源2新增({len(src2_filtered - src1_filtered)}) "
          f"∪ 复合体补完({len(complex_members_added)})")
    print(f"       硬剔除: {len(hard_exclude)} 个 TF (ΔMSE 有害)")
    print(f"       被丢弃的中间态 TF (非有害但无信息): "
          f"{n_total - len(whitelist) - len(hard_exclude)} 个")

    # ── 7. 构造输出表 ────────────────────────────────────────────────────────
    def _src_label(tfu):
        tags = []
        if tfu in src1:                tags.append("top60_ablation")
        if tfu in HC_Q2_TOP:           tags.append("HC_Q2")
        if tfu in HC_Q3_TOP:           tags.append("HC_Q3")
        if tfu in complex_members_added: tags.append("complex_add")
        return "|".join(tags) if tags else "?"

    rows = []
    for tfu in sorted(whitelist, key=lambda t: -tf2dmse.get(t, 0)):
        r = df[df["tf_upper"] == tfu].iloc[0]
        rows.append({
            "tf_name":    r["tf_name"],
            "tf_idx":     int(r["tf_idx"]),
            "delta_mse":  float(r["delta_mse"]),
            "delta_pearson": float(r["delta_pearson"]),
            "sources":    _src_label(tfu),
            "complexes":  ",".join(complex_membership.get(tfu, [])),
        })
    wl_df = pd.DataFrame(rows)
    wl_path = out / "tf_whitelist.csv"
    wl_df.to_csv(wl_path, index=False)
    print(f"\n[输出] 白名单 → {wl_path}")

    # 被硬剔除 CSV
    ex_rows = []
    for tfu in sorted(hard_exclude, key=lambda t: tf2dmse.get(t, 0)):
        r = df[df["tf_upper"] == tfu].iloc[0]
        ex_rows.append({
            "tf_name":   r["tf_name"], "tf_idx": int(r["tf_idx"]),
            "delta_mse": float(r["delta_mse"]),
            "reason":    f"delta_mse<{HARD_EXCLUDE_THRESHOLD}",
        })
    ex_df = pd.DataFrame(ex_rows)
    ex_path = out / "tf_excluded.csv"
    ex_df.to_csv(ex_path, index=False)
    print(f"       硬剔除 → {ex_path}")

    # ── 8. 摘要 TXT (人类可读) ──────────────────────────────────────────────
    summary = out / "tf_whitelist_summary.txt"
    with open(summary, "w") as f:
        f.write(f"TF 白名单构造报告\n{'='*60}\n")
        f.write(f"输入: {abl_path}  (共 {n_total} 个 TF)\n")
        f.write(f"参数: TOP_N={TOP_N}  HARD_EXCLUDE_THRESHOLD={HARD_EXCLUDE_THRESHOLD}\n\n")
        f.write(f"源 1 (ΔMSE Top-{TOP_N}):         {len(src1_filtered)}\n")
        f.write(f"源 2 (HC Q2/Q3):               {len(src2_filtered)}\n")
        f.write(f"源 1 ∪ 源 2 (锚点):             {len(anchor)}\n")
        f.write(f"源 3 (复合体补完新增):           {len(complex_members_added)}\n")
        f.write(f"硬剔除 (ΔMSE 有害):             {len(hard_exclude)}\n\n")
        f.write(f"最终白名单: {len(whitelist)} / {n_total} "
                f"({len(whitelist)/n_total*100:.1f}%)\n\n")
        f.write(f"白名单 (按 ΔMSE 降序):\n")
        f.write(wl_df.to_string(index=False))
        f.write(f"\n\n硬剔除列表:\n")
        f.write(ex_df.to_string(index=False))
    print(f"       摘要  → {summary}")

    # ── 9. 下一步提示 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  ▶ 下一步: 参照 new40_whitelist_patch.md 修改 new40.py main block")
    print(f"    重训命令:  python3 new40.py   # 会自动检测 {wl_path.name}")
    print(f"    输出将写到: ./result_filtered_{len(whitelist)}/")
    print("=" * 62)


if __name__ == "__main__":
    main()
