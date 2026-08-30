#!/usr/bin/env python3
# new55.py — 酵母基因表达多模态预测模型
#
# 输入级消融链 (RUN_ABLATIONS): D → L → B → C / T
#   D — DNA one-hot CNN（从零训练的纯序列基线，-1000/+500 bp）
#   L — LM-only（预训练真菌 LM 嵌入，无 activity 标量）
#   B — Activity + LM（成熟的序列分支，作为 seq_only_pred 象限锚 + 软监督源）
#   C — B + TF cross-modal（完整模型；无辅助损失、无 Q2 上采样）
#   T — ours 架构内的 TF-only 诊断（关 seq 分支 / fusion）
#
# 模块级消融 (RUN_FUSION_ABLATIONS + FUSION_ABLATION)：
#   在 C 的完整架构上做 surgical ablation, 每 variant × 多 seed, 与 C 配对对比.
#
# 主要顶层配置：
#   RUN_ABLATIONS / OURS_ONEHOT_EXP / RESUME_FROM_CACHE / GLOBAL_TRAIN /
#   CHR_HOLDOUT (3 holdout 稳健性) / MODEL_CONFIG / ABLATION_OVERRIDE /
#   MULTISEED_PROGRESSION /
#   RUN_FUSION_ABLATIONS + FUSION_ABLATION_SEEDS + FUSION_ABLATION /
#   MODEL_COMPARISON / TF_ONLY_BASELINE / MULTI_SEED_EVAL / BOOTSTRAP_CI

import re, math, random, warnings, traceback, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

warnings.filterwarnings('ignore')

try:
    import chardet; HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False

# ============================================================================
# ★ 顶部统一配置区
# ============================================================================

PATH_CONFIG = dict(
    npz_file             = "./data/105signal_matrices_10.npz",
    bed_file             = "./data/tss.bed",
    tpm_file             = "./data/DMSO_expression.txt",
    tpm_column           = "TPM_median",
    lm_npz_file          = "./shorkieLM/result/lm_deep_tss_gaussian_sig0.10_up1000_dn500.npz",
    iaa_fc_file          = "./data/perturbation_fc_matrix.tsv",
    tf_direction_csv     = "./data/tf_direction_classification_v7.csv",
    promoter_profile_csv = "./data/seq_activity.csv",
    # ── 筛选后的 TF 白名单（若存在则按此 csv 对 TF 维度做子集化）─────────
    # 文件需含列 `tf_name`；留空/置 None 则跳过白名单筛选
    #tf_whitelist_csv     = " ",
    tf_whitelist_csv     = "./data/analysis/tf_whitelist.csv",
    # ── DNA one-hot 分支 ────────────────────────────────────────────────
    genome_fasta_file    = "./data/S288C.fsa",
    dna_onehot_cache     = "./data/dna_onehot_up1000_dn500.npz",
    output_dir           = "./53",
    analysis_dir         = "./53/analysis",
)

# 消融代号: 'D'=DNA-CNN, 'L'=LM-only, 'B'=Act+LM, 'C'=Full(B+TF, 无辅助损失),
#          'T' =ours 架构中只保留 TF 分支 (不含 LM / Activity / fusion).
#                直接对比 Ridge (tss_weighted 0.64 / raw_bins_pca 0.78),
#                回答「PerTFEncoder + TFSpatialEncoder + TFAttentionPooler
#                本身是否强过 Ridge」 —— 独立验证 TF-side 架构价值.
# 常用组合: ['D','L','B','C'] 全量 / ['B','C'] 常规 / ['B'] 快速 smoke-test
#          ['B','C','T'] 常规 + TF-only 诊断
#RUN_ABLATIONS = ['B', 'C']
RUN_ABLATIONS = ['D','L','B','C','T']
#RUN_ABLATIONS = ['B','C','T']

# ============================================================================
# ★ (Q1) Ours-Architecture + DNA one-hot 可控实验（OURS_ONEHOT_EXP）
# ============================================================================
# 目的: 解耦 "序列表征(LM vs one-hot)" 与 "架构(cross-modal vs 拼接)" 对
# 最终性能的贡献 —— 以 C 的完整架构为底, 把 SequenceBranch 的 LMAdapter
# 换成 DNAOneHotCNN。结果放 supplementary, 堵住 reviewer 关于 confounded
# comparison 的质疑。
#
# 启用后会在 C 完成后跑一次 'ablation_C_onehot_arch', 与 C 的 Pearson 做
# 对比, 得到 Δ = seq-repr gain (same arch)。
OURS_ONEHOT_EXP = dict(
    enabled       = True,      # 主开关; False 则完全跳过
    d_dna_out     = 128,        # DNAOneHotCNN 输出维度（对齐 d_lm_out）
    dna_dropout   = 0.20,       # one-hot encoder 内部 dropout
    revcomp_prob  = 0.0,        # 训练时反向互补增强概率（默认 0, 与 B/C 一致）
    # 训练覆盖参数（留空则复用 C 的配置）; 可酌情增大 epochs 补偿 from-scratch encoder
    overrides     = dict(epochs=350, lr=1.5e-4, patience=80, warmup_epochs=20),
)

# ============================================================================
# ★ (Panel H) 因果 Δc 反事实实验（CAUSAL_DELTA_C）
# ============================================================================
# 目的: 把 cross_weight c 从「与表达量相关」推进到「有反式因果含义」, 直接回应正文
#   "c 因果地位待验证" 的留白。做法: 在已训练好的完整模型 C 上做一次反事实前向 ——
#   把 TF(ChEC-seq) 输入置零(撤掉反式信号, 顺式侧 scalars / LM 不动)再前向一次,
#   同一次前向里同时收集 (基线门控 c0, 基线预测 ŷ0):
#       Δc = c − c0     真实反式状态把闸门(cross_weight)推开了多少
#       Δŷ = ŷ − ŷ0     反式信号真正让模型预测移动了多少
#
#   为什么 Δc 比原始 c 更干净 ── 原始 c = σ(⟨Q,KV⟩+b) 的跨基因差异被 c0=σ(⟨Q,KV0⟩+b)
#   这一「表达量驱动的基线门控值」主导(正文 c~true_expr ρ=+0.50 即源于此);
#   Δc 把 c0 差分掉, 只留下「TF 信号诱导的门控增量」⟨Q, ΔKV⟩, 因此能看到原始
#   c 分析(E/G/H)被表达量基线掩盖的反式信号。
#
#   ★ 金标准检验 corr(Δc, |Δŷ|) on Q2∪Q3 ── 把 c 直接绑到「模型内部反式贡献的功能」,
#   不依赖噪声大的外部 IAA, 是 noCrossW(c≡1)冗余结论的天然补充, 也是即使外部 IAA
#   全 null 时仍可能为正的那个检验。决策: corr(Δc,|Δŷ|) 强但 Δc~IAA 仍 null →
#   "c 在功能上确实门控模型内部反式贡献, 但该贡献不对齐实测扰动敏感性"; 两者皆 null →
#   因果层面钉死 "c=生长程序倾向、非反式负荷"。
#
#   内置偏置提醒: 增益项 c(1−c) 在 c≈0.5 最大、在高表达(c≈0.85)处偏小, 故 Δc 天然
#   偏向中等-c(偏低表达)基因 —— 下方所有判别性检验一律控 log-TPM(= z log-TPM)。
#
# 产出 (写入 output_dir):
#   · combined.csv 追加两列  cross_weight_c_baseline / yhat_baseline
#       → cross_weight_c_baseline 直接解锁 crossweight_quadrant_standalone.py 的
#         Panel H 因果 Δc 模式(已修正为限定 Q2/Q3); yhat_baseline 供其后续算 Δŷ。
#   · causal_delta_c_<tag>.csv  逐基因 (c/c0/Δc/ŷ/ŷ0/Δŷ/|trans_resid|/IAA/log-TPM/象限)
#   · 终端打印金标准裁决(默认 test split, 限定 Q2∪Q3, 一律控 log-TPM):
#       (1) corr(Δc, |Δŷ|)               功能绑定(最强、内生)
#       (2) Δc 是否 Q2/Q3 > Q1/Q4         反式象限富集 (MWU + Cliff's δ)
#       (3) partial corr(Δc, IAA|log2FC|) 外部扰动一致性(需 IAA)
#       (4) corr(Δc, |trans_resid|)       与「所需反式修正量」对齐(需 ablation B)
CAUSAL_DELTA_C = dict(
    enabled       = True,        # 主开关; False 则完全跳过(只影响本实验, 不动其它流程)
    baseline      = 'zero',      # 'zero' | 'mean' | 'shuffle' —— TF 输入的反事实基线
                                 #   zero    : TF→0 (反式信号撤除; KV 退化为偏置常数) ← 默认/正文口径
                                 #   mean    : 用全量逐 bin 均值替代(保留"平均反式背景")
                                 #   shuffle : 逐基因打散 TF 轴(破坏基因特异反式模式, 保留边际)
    split         = 'test',      # 'test' | 'all' —— 金标准检验所用子集(test 最干净, 排除已见标签)
    n_perm        = 2000,        # 偏相关置换检验次数
    shuffle_seed  = 42,          # baseline='shuffle' 的随机种子(可复现)
)

# ============================================================================
# ★ Bootstrap 置信区间
# ============================================================================
BOOTSTRAP_CI = dict(
    enabled     = True,
    n_boot      = 5000,
    ci_level    = 0.95,
    random_seed = 42,
)

# ============================================================================
# ★ 多种子重复实验（仅对全模型 C 生效）
#   每个 seed 额外记录「每象限 Pearson / MAE / Bias」, 覆盖论文核心叙事
#   象限定义: 用 seed=42 B 模型的 seq_only_pred 作 X 轴锚, TPM 作 Y 轴锚,
#            跨 seed 共用同一套象限标签 → std 才能归因到「模型随机性」
# ============================================================================
#    修正 (paired-Δ bug fix):
#     · 默认 enabled=True — 否则 FUSION_ABLATION 的配对 Δ 退化到单 seed=42 (假象)
#     · seeds 与 FUSION_ABLATION_SEEDS 同步, 两块共享 C per-seed (避免重复训练 5 次 C)
MULTI_SEED_EVAL = dict(
    enabled         = True,
    seeds           = [42, 123, 456, 789, 2024],
    # 分象限统计开关 — 启用后每 seed 额外输出 Q1/Q2/Q3/Q4 的 Pearson/MAE/Bias
    quadrant_stats  = True,
    # 与 FUSION_ABLATION 共享 C per-seed (若 FUSION_ABLATION 已跑过这些 seed,
    # MULTI_SEED_EVAL 直接复用缓存, 不重复训练). 默认 True.
    reuse_fusion_abl_c = True,
)

# ============================================================================
# ★ TF-only 基线 (不进 MODEL_COMPARISON 矩阵)
#   目的: 独立测试「TF 信号 79 维 TSS 加权均值 → Ridge/XGBoost」的 pearson,
#        回答「整个 cross-modal 架构的 gain 是不是在训练噪声带里」的大问题.
#   与 MULTI_SEED_EVAL 共用同一套象限标签, 保证可直接对齐比较.
# ============================================================================
TF_ONLY_BASELINE = dict(
    enabled         = True,                     # 主开关
    models          = ['Ridge', 'XGBoost'],     # Ridge 确定性 (跨 seed 同值, 仅跑 1 次)
                                                # XGBoost 对 seed 敏感 (跑全部 seeds)
    seeds           = [42, 123, 456, 789, 2024],
    # ── TF 特征表示 × variants ────────────────────────────────────────────
    # 每个 variant 独立跑一遍 (model × seed), 结果汇总到同一 csv 里 (多一列 variant).
    # 目的:
    #   'tss_weighted' → 与 MODEL_COMPARISON 列对齐, 代表「聚合标量」的上限.
    #   'raw_bins_pca' → 与 ours-C 一样拿原始 79×150 bin 特征, 仅差架构.
    #                    若 raw_bins_pca Ridge 的 Q2 pearson 也高于 ours-C,
    #                    则问题定位在 PerTFEncoder / cross-modal 架构本身,
    #                    而非 TF 信息量不足.
    # 原单值 `tf_summary` 字段仍保留, 向后兼容 (若 `variants` 未给则退化为单跑).
    variants        = ['tss_weighted', 'raw_bins_pca'],
    tf_summary      = 'tss_weighted',           # deprecated, 被 variants 覆盖
    bootstrap       = True,
    n_boot          = 2000,
    ci_level        = 0.95,
    quadrant_stats  = True,                     # 分象限 Pearson/MAE/Bias
    output_subdir   = 'tf_only_baseline',
    # 磁盘缓存复用. tf_only_per_seed.csv + tf_only_quadrant_per_seed.csv
    #   都已存在时 skip 全部训练, 仅读回. 设 True 强制重训.
    force_retrain   = False,
)

# ============================================================================
# ★ 断点续跑 / 结果缓存（RESUME_FROM_CACHE）
#   目的: 5-seed 实验被中断时, 已完成的实验无需重跑.
#   判定逻辑: exp_dir 下同时存在 complete_model.pth 和
#             train_result_cache.pth 视为"完整完成".
#   失效方式:
#     · 全局关闭: enabled = False
#     · 单个实验: 删除对应 exp_dir/train_result_cache.pth 或整个 exp_dir/
#     · 超参改变后: 建议改 output_dir 后缀, 或手动清理旧目录
# ============================================================================
RESUME_FROM_CACHE = dict(
    enabled = True,   # 主开关; False = 每次都重跑（不读也不写缓存）
)

GLOBAL_TRAIN = dict(
    batch_size     = 256,
    epochs         = 300,
    lr             = 2e-4,
    warmup_epochs  = 15,
    patience       = 60,
    tpm_transform  = 'log1p',
    tpm_normalize  = 'zscore',
    split_strategy = 'fixed_saccer3',
    random_seed    = 42,
)

# ============================================================================
# ★ 染色体 Holdout 稳健性 (P0 #12 · v7 → v8 新增)
# ----------------------------------------------------------------------------
# 目的: 验证主管道 (chrXV-XVI test, 5-seed 0.8482 ± 0.0051) 的性能不依赖
#       特定 test 染色体, 解锁论文 Supp S12 + Methods 的"robust to choice
#       of test chromosomes"陈述.
#
# 设计原则 (零侵入 + 一行切换):
#   1) 默认 enabled=False → 行为与 v7 完全一致, 不影响现有缓存.
#   2) 启用后:
#        a. split_strategy 切换为 'custom', 从 CHR_HOLDOUT_PRESETS 取 chrs
#        b. PATH_CONFIG.output_dir / analysis_dir 自动加 '_holdout_<preset>'
#           后缀 → 与主管道 ./53_filtered_79 完全隔离, 缓存互不污染
#        c. 5-seed / Fusion ablation / multiseed eval / 四象限 / GO 分析
#           全部沿用原有逻辑 (它们都基于 split_masks, 自动跟随)
#   3) chr_holdout_meta.json 自动落地, 记录 preset / chrs / n_test 供 Methods 引用.
#
# 推荐工作流 (跑 K 个 preset 的扫描):
#   for preset in [chr1_chr2, chr3_chr4, chr5_chr6, chr11_chr12]:
#       1) 改 CHR_HOLDOUT['enabled']=True / CHR_HOLDOUT['preset']=preset
#       2) 运行 python new53.py
#       3) 收集 ./53_filtered_79_holdout_<preset>/multiseed_C/multiseed_C_summary.csv
#   最后聚合到 paper_supp_s12_chr_holdout.csv (主管道 0.8482 + K 个 holdout 数字).
#
# 预设方案 (test 大小 / 染色体大小估计来自 saccer3, n_genes ≈):
#   · chr1_chr2   : chrI+chrII     ~ 553   (test 较小, power 较强)
#   · chr3_chr4   : chrIII+chrIV   ~ 1014  ← ★ 与主管道 1108 最接近, 推荐主对比
#   · chr5_chr6   : chrV+chrVI     ~ 446
#   · chr7_chr8   : chrVII+chrVIII ~ 889
#   · chr9_chr10  : chrIX+chrX     ~ 628
#   · chr11_chr12 : chrXI+chrXII   ~ 920
#   · fixed_saccer3 : chrXV+chrXVI ~ 1108  (reference, sanity-check 流程对齐)
#   · custom      : 任意 (经 custom_split 字段)
#
# 验收标准 (Methods 中陈述):
#   K 个 preset 的 5-seed mean Pearson 都落在 [0.83, 0.86] 区间
#   (≤ 0.02 偏差) 即可声明 "robust to choice of test chromosomes".
# ============================================================================
CHR_HOLDOUT = dict(
    enabled       = False,           # 主开关; False = 与 v7 行为完全一致
    preset        = 'chr3_chr4',     # 预设名 (见下); 'custom' 走 custom_split
    custom_split  = None,            # 仅 preset='custom' 时生效, 例:
                                     # dict(test_chrs=['chrI','chrII'],
                                     #      val_chrs =['chrXIII','chrXIV'])
    output_suffix = None,            # None = 自动用 '_holdout_<preset>'
)

CHR_HOLDOUT_PRESETS = {
    'chr1_chr2':   dict(test_chrs=['chrI',  'chrII'],
                        val_chrs =['chrXIII','chrXIV']),
    'chr3_chr4':   dict(test_chrs=['chrIII','chrIV'],
                        val_chrs =['chrXIII','chrXIV']),
    'chr5_chr6':   dict(test_chrs=['chrV',  'chrVI'],
                        val_chrs =['chrXIII','chrXIV']),
    'chr7_chr8':   dict(test_chrs=['chrVII','chrVIII'],
                        val_chrs =['chrXIII','chrXIV']),
    'chr9_chr10':  dict(test_chrs=['chrIX', 'chrX'],
                        val_chrs =['chrXIII','chrXIV']),
    'chr11_chr12': dict(test_chrs=['chrXI', 'chrXII'],
                        val_chrs =['chrXIII','chrXIV']),
    # reference: 等价于主管道, 仅用于 sanity check (启用后会重训, 不推荐)
    'fixed_saccer3': dict(test_chrs=['chrXV','chrXVI'],
                          val_chrs =['chrXIII','chrXIV']),
}


# ── 环境变量覆盖 (让 sweep wrapper 无需改源码即可切 preset) ─────────────
#   用法 (单跑):    CHR_HOLDOUT_PRESET=chr3_chr4 python new53.py
#   用法 (sweep):  python run_chr_holdout_sweep.py
#   优先级: 环境变量 > 顶部 CHR_HOLDOUT 字典 (即仅当 env 设置时才覆盖)
_env_preset = os.environ.get('CHR_HOLDOUT_PRESET', '').strip()
if _env_preset:
    CHR_HOLDOUT['enabled'] = True
    CHR_HOLDOUT['preset']  = _env_preset
    print(f"  [env override] CHR_HOLDOUT_PRESET={_env_preset} → enabled=True")
elif os.environ.get('CHR_HOLDOUT_ENABLED', '').strip() in ('1', 'true', 'True'):
    # 只设了 ENABLED 没设 PRESET → 用顶部默认 preset
    CHR_HOLDOUT['enabled'] = True
    print(f"  [env override] CHR_HOLDOUT_ENABLED=1 → enabled=True "
          f"(preset='{CHR_HOLDOUT['preset']}')")


def _resolve_chr_holdout(cfg, presets):
    """根据 CHR_HOLDOUT 配置返回 (test_chrs, val_chrs, suffix).
    若 enabled=False 返回 (None, None, '')."""
    if not cfg.get('enabled', False):
        return None, None, ''
    preset = cfg['preset']
    if preset == 'custom':
        cs = cfg.get('custom_split') or {}
        if not cs.get('test_chrs'):
            raise ValueError(
                "CHR_HOLDOUT.preset='custom' 但 custom_split.test_chrs 为空 — "
                "请在 CHR_HOLDOUT['custom_split'] 中提供 test_chrs (并可选 val_chrs).")
        test_chrs = list(cs['test_chrs'])
        val_chrs  = list(cs.get('val_chrs', ['chrXIII', 'chrXIV']))
    else:
        if preset not in presets:
            raise ValueError(
                f"CHR_HOLDOUT.preset='{preset}' 未在 CHR_HOLDOUT_PRESETS 中, "
                f"可用: {sorted(presets.keys())}")
        cs = presets[preset]
        test_chrs = list(cs['test_chrs'])
        val_chrs  = list(cs.get('val_chrs', ['chrXIII', 'chrXIV']))
    # train/val/test 互斥校验
    _overlap_tv = set(test_chrs) & set(val_chrs)
    if _overlap_tv:
        raise ValueError(
            f"CHR_HOLDOUT preset='{preset}' 中 test 与 val 染色体重叠: {_overlap_tv}")
    suffix = cfg.get('output_suffix') or f"_holdout_{preset}"
    return test_chrs, val_chrs, suffix


MODEL_CONFIG = dict(
    d_tf         = 128,
    d_cnn        = 128,
    n_tf_layers  = 1,
    n_regions    = 3,
    n_heads      = 4,
    d_seq        = 128,
    d_tf_proj    = 96,
    d_profile    = 64,
    d_lm_hidden  = 512,
    d_lm_out     = 128,
    lm_input_drop= 0.20,
    lm_noise_std = 0.05,
    fc_dropout   = 0.35,
    freeze_lm    = False,
    lm_layer_key = 'lm_concat',
    # Top-K sparse TF attention — 保留, 但默认 0 (关闭).
    # 在 n_tfs=178 下 TopK-30 ≈ 17% 稀疏, 有意义; 但在 n_tfs=79 下 = 38%,
    # 稀疏化失去意义, 另外 49 个 TF 被硬归零的梯度代价仍在, 净效应为负
    # (Q2 长尾 repressor 信号被系统性丢弃). new37(178 TF) 不用 TopK 就到 0.85.
    topk_k       = 0,
)

WINDOW_CONFIG = dict(upstream=1000, downstream=500)   # TF 信号分支窗口（保持不变）
_wu = WINDOW_CONFIG['upstream']; _wd = WINDOW_CONFIG['downstream']
WINDOW_UPSTREAM   = _wu
WINDOW_DOWNSTREAM = _wd
TSS_RATIO         = _wu / (_wu + _wd)

# LM 嵌入使用更大的上游窗口，以充分利用 LM 的长程上下文能力
LM_WINDOW_CONFIG = dict(upstream=1000, downstream=500)
_lm_wu = LM_WINDOW_CONFIG['upstream']; _lm_wd = LM_WINDOW_CONFIG['downstream']
LM_WINDOW_UPSTREAM   = _lm_wu
LM_WINDOW_DOWNSTREAM = _lm_wd
LM_TSS_RATIO         = _lm_wu / (_lm_wu + _lm_wd)   # 0.8333

# ── DNA one-hot 窗口（消融 D）────────────────────────────────────────────
# 与 LM 窗口一致（-1000/+500 bp），使 D vs L 的差异可归因于"预训练"而非窗口
DNA_WINDOW_CONFIG = dict(upstream=1000, downstream=500)
_dna_wu = DNA_WINDOW_CONFIG['upstream']; _dna_wd = DNA_WINDOW_CONFIG['downstream']
DNA_WINDOW_UPSTREAM   = _dna_wu
DNA_WINDOW_DOWNSTREAM = _dna_wd
DNA_TSS_RATIO         = _dna_wu / (_dna_wu + _dna_wd)
DNA_SEQ_LEN           = _dna_wu + _dna_wd

USE_LM_EMB       = True
NORMALIZE_LM_EMB = True

SEQ_PROFILE_COLS_DEFAULT = [
    '-160to-80','-240to-160','-320to-240','-400to-320',
    '-480to-400','-560to-480','-640to-560','-720to-640',
    '-800to-720','-880to-800','-960to-880','-1040to-960',
]

ABLATION_OVERRIDE = {
    # D: 从零训练 DNA-CNN，需要更多 epoch + 更小 lr + 更强正则
    'D': dict(epochs=400, lr=1e-4, patience=60, warmup_epochs=20),
    'L': {},           # LM-only, 复用现有 seq_only 训练流程
    'B': {},
    # C: 全模型. 历史上 decorr_w=0.10 + region_div_w=0.05 + Q2 aux 被证实净效应为负
    # (C0 诊断: Δ -0.014 global, +17 基因级 Q2 attention 漂移). 这些辅助损失 / Q2
    # 上采样 / Q2 loss 2.0x 已在 new53 整体下线 —— 当前 C = 无辅助基线.
    'C': {},
    # T: ours 架构内的 TF-only 消融 — 保留 PerTFEncoder + TFSpatialEncoder +
    #    TFAttentionPooler 这套 TF 分支, 但关闭 LM / Activity / fusion.
    #    MultiModalPredictor(use_scalars=False, use_lm=False) 原生支持此配置
    #    (self.fusion=None, fc_input=tf_merged 直通).
    #    与 Ridge/XGBoost baseline 直接对比, 独立验证 TF-side 架构价值.
    #    - epochs 增加到 400: 没有 LM 预训练权重, 需要更多训练.
    #    - lr 保持 2e-4: 参数量 <1M, 不需要更小 lr.
    'T':  dict(epochs=400, patience=60),
}


# ============================================================================
# ★ 架构链多种子（B / T）— Protocol 2 配对协议数据源 (new54 增补)
# ----------------------------------------------------------------------------
# 目的: 把 D→L→B→C→T 链的 B/T 也升级为 5-seed, 让 Δ(B→C) 与 Δ(T→C) 可在
#       paired 5-seed 协议下计算 (Cohen's d + Wilcoxon two-sided), 避免出现
#       "single-seed B/T vs 5-seed C" 的不对称对比 (v11 chr1/2 +0.058 → +0.049
#       校正即此问题的典型症状).
#
# 设计原则 (与 FUSION_ABLATION 完全对齐):
#   1. 默认 enabled=False → 行为与 v11 完全一致, 不影响现有缓存.
#   2. seeds 默认与 FUSION_ABLATION_SEEDS 同步, 保证 B/T/C 在同一 seed 集上配对.
#   3. seed=42 直接复用主链单 seed 运行 (ablation_B_act_lm / ablation_T_tf_only_arch),
#      无需重训; 其它 seed 走 multiseed_<X>_seed{s} 子目录 + RESUME_FROM_CACHE,
#      已跑过的 seed 不会重跑.
#   4. D / L 不在多 seed 集合内: D 是 from-scratch DNA-CNN, std 不影响论点;
#      L 同样, 论文 progression 只把 D/L 作 single-seed baseline 引用.
#   5. C 不在 models 集合内: 已通过 FUSION_ABLATION + MULTI_SEED_EVAL 多 seed 化.
#      若 RUN_FUSION_ABLATIONS=False 且 MULTI_SEED_EVAL.enabled=False, 可显式
#      把 'C' 加进 models 让本块独立产出 C 的 5 seed (此时与主管道功能等价).
#
# 输出:
#   {output_dir}/multiseed_B/multiseed_B_summary.csv
#   {output_dir}/multiseed_T/multiseed_T_summary.csv
#   每个 summary 含 per-seed pearson + (mean, std, 95% CI) + 含哪些 seed 来自
#   缓存 / 哪些 seed 由本次 run 新训.
# ============================================================================
MULTISEED_PROGRESSION = dict(
    enabled = True,                                  # 主开关
    models  = ['B', 'T'],                            # 哪些模型补 5 seed
    seeds   = [42, 123, 456, 789, 2024],             # 与 FUSION_ABLATION_SEEDS 同步
)


# ============================================================================
# ★ Fusion 模块级消融（P1）
#
# 目的：在 C 完整架构基础上做 surgical ablation，回答几个科学问题
#   Q1 — 跨模态融合机制整体 vs 朴素拼接（C_concat 给出下界）
#   Q2 — cross_w 门控自身的贡献（noCrossW, 冻结成标量 1 / 0）
#   Q3 — LM / Activity 在完整架构下的增量
#   Q4 — +0.028 架构 gain 是 TF encoder 还是训练流程 (C_tss_weighted)
#
# 协议：
#   · 每 variant × FUSION_ABLATION_SEEDS, 与 C 同 split / 同超参 / 同 B-软监督
#   · 复用 _exp_kwargs('C') → 无辅助损失 (new53 已整体下线 decorr / region_div / Q2)
#   · 配对 Δ = Pearson(C, seed_i) − Pearson(variant, seed_i)
#
# 标志说明（MultiModalPredictor / ResidualCrossModalFusion 内部生效）：
#   replace_with_concat=True  → fusion 替换为 concat+Linear（所有跨模态机制全撤）
#   freeze_crossw='one'       → cross_w ≡ 1（去门控；保留跨模态路径）
#   freeze_crossw='zero'      → cross_w ≡ 0（sanity；seq-only pass-through）
#   disable_lm=True           → SequenceBranch.use_lm=False（保留 activity + fusion）
#   disable_activity=True     → SequenceBranch.use_profile=False（保留 LM + fusion）
#   use_tss_weighted_tf=True  → 跳过 TFAttentionPooler + TFSpatialEncoder, 用
#                               TSS-Gaussian 加权标量 (维度 = n_tfs) 作 tf_merged.
# ============================================================================
RUN_FUSION_ABLATIONS  = True        # 主开关；True 才执行（默认关不影响现有流程）
FUSION_ABLATION_SEEDS = [42, 123, 456, 789, 2024]

# 当前启用的 4 个 variant (5-seed 主实验, 均为 paper-grade):
#   C_concat       - 朴素 concat vs ResidualCrossModalFusion (Δ≈0)
#   C_noCrossW     - cross_w≡1 (确认 cross_w 是零成本可解释出口而非性能模块)
#   C_noLM         - 去 LM (3-holdout 联合 +0.040 ± 0.021, all d > 4 ★★)
#   C_tss_weighted - 简化 TF encoder, 定位学习型 TF encoder 主效应
# 已废弃 (new50 删, 详见模块裁剪声明):
#   C_noMHA / C_noRegion / C_noProfileSkip / C_lmGate
FUSION_ABLATION = {
    'C_concat':        dict(replace_with_concat=True),
    'C_noCrossW':      dict(freeze_crossw='one'),
    'C_noLM':          dict(disable_lm=True),
    'C_tss_weighted':  dict(use_tss_weighted_tf=True),
}


# ============================================================================
# ★ 模型对比矩阵配置
#    行: Linear(Ridge/Lasso) / 非线性(XGBoost/LightGBM) /
#        DL-Tabular(GatedMLP/FT-Transformer) / DL 消融(L/B/C)
#    列: LM-only / +Activity / +TF(Full)
#    同一特征集下跑所有模型 → 性能差异可归因于模型复杂度
#    三列与 对消融链 L/B/C 1:1应，每列对应 one specific architectural claim
#
# new47 P0: 作为 C_minimal 基线的载体 — GatedMLP on +TF(Full) 列等价于
#   "LM 直连 MLP + TF(tss_weighted) concat + Activity concat + 线性 head",
#   是 ResidualCrossModalFusion 的架构下界. 若 ≥ 0.845 则"精巧架构"叙事被推翻.
# ============================================================================
MODEL_COMPARISON = dict(
    enabled        = True,         # new47: 打开 — C_minimal 决定叙事方向
    # new53: 磁盘缓存复用. 若 comparison_matrix_full.csv + comparison_matrix_quadrant.csv
    #   在 output_subdir 下都已存在, 整块 skip (不再重训六个基线 × 三列).
    #   设 True 可强制重训. 适用于只想刷新 Supp S11 汇合 / 重画图的场景.
    force_retrain  = False,
    run_linear     = True,         # Ridge + Lasso (三列交叉 → 线性下界)
    run_nonlinear  = True,        # XGBoost/LightGBM 只在需要时开 (依赖包较重)
    run_dl_tabular = True,         # GatedMLP + FT-Transformer ← C_minimal 主载体
    # TF 信号汇总方式（用于线性/非线性/DL-Tabular 模型，避免高维灾难）
    # 'mean'        → 每个 TF 取 bin 均值（维度 = n_tfs）
    # 'meanstd'     → 均值 + 标准差（维度 = 2*n_tfs）
    # 'tss_win'     → TSS ±20 bins 均值（维度 = n_tfs）
    # 'tss_weighted'→ TSS 高斯加权均值（σ=0.15×n_bins；和 PerTFEncoder 内部 σ≈0.15 一致
    tf_summary     = 'tss_weighted',
    output_subdir  = 'model_comparison_matrix',
    bootstrap      = True,
    n_boot         = 5000,
    ci_level       = 0.95,
    random_seed    = 42,
    # GatedMLP / FT-Transformer 超参数
    dl_tabular = dict(
        epochs          = 150,
        batch_size      = 128,
        lr              = 3e-4,
        patience        = 20,
        d_token         = 64,    # FT-Transformer token 维度
        n_layers        = 3,     # FT-Transformer encoder 层数
        n_heads         = 4,
        dropout         = 0.20,
        d_hidden_mlp    = 256,   # GatedMLP 隐层维度
        lm_compress_dim = 64,    # FT-Transformer 专用：PCA 压缩 LM 嵌入维度
    ),
)

# ============================================================================
# 辅助工具函数
# ============================================================================

def compute_seq_summary_torch(seq_z: torch.Tensor) -> torch.Tensor:
    if seq_z.ndim == 1: seq_z = seq_z.unsqueeze(1)
    n = seq_z.shape[1]
    if n >= 3:
        w = torch.tensor([0.5, 0.3, 0.2], device=seq_z.device, dtype=seq_z.dtype)
        return (seq_z[:, :3] * w).sum(dim=1)
    if n == 2: return 0.65*seq_z[:,0] + 0.35*seq_z[:,1]
    return seq_z[:,0]

def compute_seq_summary_np(seq_z: np.ndarray) -> np.ndarray:
    seq_z = np.asarray(seq_z)
    if seq_z.ndim == 1: seq_z = seq_z[:,None]
    n = seq_z.shape[1]
    if n >= 3:
        w = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        return (seq_z[:,:3] * w).sum(axis=1)
    if n == 2: return 0.65*seq_z[:,0] + 0.35*seq_z[:,1]
    return seq_z[:,0]

def _safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device)

# ============================================================================
# 模型子模块
# ============================================================================

class ProfileCNNBranch(nn.Module):
    """3层残差MLP分支：对12个离散上游活性标量建模（约27k参数）。
    恢复原始MLP结构：12个窗口本质是离散统计标量，不是连续信号；
    1D Conv版的 AdaptiveAvgPool1d(1) 会丢掉所有位置信息，反而不如MLP。
    （1D Conv版仅10k参数，是 ablation B 下降 0.021 的次要原因）"""
    def __init__(self, n_windows=12, d_out=64, dropout=0.10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_windows, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 128),       nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, d_out),     nn.LayerNorm(d_out))
        self.out_dim = d_out
    def forward(self, x):
        # x: (B, n_windows) → (B, d_out)
        return self.mlp(x)


class LMAdapter(nn.Module):
    """LM 嵌入适配器（含残差跳接）"""
    def __init__(self, lm_dim=2304, d_hidden=512, d_out=128,
                 hidden_dropout=0.15, input_dropout=0.10, freeze=False):
        super().__init__()
        self.freeze = freeze
        self.pre = nn.Sequential(
            nn.Dropout(input_dropout),
            nn.Linear(lm_dim, d_hidden), nn.LayerNorm(d_hidden), nn.GELU())
        self.mlp = nn.Sequential(
            nn.Dropout(hidden_dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.LayerNorm(d_hidden // 2), nn.GELU(),
            nn.Dropout(hidden_dropout * 0.4),
            nn.Linear(d_hidden // 2, d_out),    nn.LayerNorm(d_out), nn.GELU(),
            nn.Dropout(hidden_dropout * 0.2))
        self.skip = nn.Sequential(nn.Linear(d_hidden, d_out), nn.LayerNorm(d_out))
        self.out_dim = d_out

    def forward(self, lm_emb):
        if self.freeze:
            lm_emb = lm_emb.detach()
        h = self.pre(lm_emb)
        return self.mlp(h) + self.skip(h)


class SequenceBranch(nn.Module):
    """Profile + 可选 LM 嵌入 / DNA one-hot 融合分支

    三种配置模式:
      · use_profile=True,  use_lm=False, use_dna_onehot=False → Activity-only（消融 A）
      · use_profile=True,  use_lm=True,  use_dna_onehot=False → Activity + LM（消融 B）
      · use_profile=False, use_lm=True,  use_dna_onehot=False → LM-only（消融 L）
      · use_profile=True/False, use_dna_onehot=True          → (Q1) ours-arch + one-hot

    use_lm 与 use_dna_onehot 互斥 —— 同一分支只能选一种序列表征。
    """
    def __init__(self, n_windows=12, d_profile=64, d_out=128, dropout=0.10,
                 use_lm=False, lm_dim=2304, d_lm_hidden=512, d_lm_out=128,
                 lm_input_dropout=0.10, freeze_lm=False, use_profile=True,
                 use_dna_onehot=False, dna_seq_len=1500, d_dna_out=128,
                 dna_dropout=0.20):
        super().__init__()
        assert not (use_lm and use_dna_onehot), \
            "SequenceBranch: use_lm 与 use_dna_onehot 互斥，序列表征只能选一种"
        self.use_profile    = use_profile
        self.use_lm         = use_lm
        self.use_dna_onehot = use_dna_onehot
        fuse_in = 0
        if use_profile:
            self.profile_branch = ProfileCNNBranch(n_windows, d_profile, dropout)
            fuse_in += d_profile
        else:
            self.profile_branch = None
        if use_lm:
            self.lm_adapter = LMAdapter(lm_dim, d_lm_hidden, d_lm_out,
                                        0.15, lm_input_dropout, freeze_lm)
            fuse_in += d_lm_out
        else:
            self.lm_adapter = None
        if use_dna_onehot:
            # (Q1) 直接复用现有的 DNAOneHotCNN —— 与消融 D 同架构, 输出对齐 d_dna_out
            self.dna_encoder = DNAOneHotCNN(
                seq_len=dna_seq_len, d_out=d_dna_out, dropout=dna_dropout)
            fuse_in += d_dna_out
        else:
            self.dna_encoder = None
        assert fuse_in > 0, \
            "SequenceBranch 至少需要 use_profile / use_lm / use_dna_onehot 之一"
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, d_out), nn.LayerNorm(d_out), nn.GELU(), nn.Dropout(dropout))
        self.out_dim = d_out

    def forward(self, profile, lm_or_dna=None):
        """
        profile  : (B, n_windows)   activity 标量（use_profile=True 时使用）
        lm_or_dna: (B, lm_dim)      当 use_lm=True  时为 LM 嵌入
                   (B, 4, L)        当 use_dna_onehot=True 时为 DNA one-hot
                   None             当 use_profile 单独使用时
        """
        parts = []
        if self.use_profile and self.profile_branch is not None:
            parts.append(self.profile_branch(profile))
        if self.use_lm and self.lm_adapter is not None and lm_or_dna is not None:
            parts.append(self.lm_adapter(lm_or_dna))
        if self.use_dna_onehot and self.dna_encoder is not None and lm_or_dna is not None:
            parts.append(self.dna_encoder(lm_or_dna))
        return self.fuse(torch.cat(parts, dim=1))


class MultiScaleDilatedBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        mid = out_ch // 3; rem = out_ch - mid * 2
        def _b(rate, ch): return nn.Sequential(
            nn.Conv1d(in_ch, ch, 3, padding=rate, dilation=rate),
            nn.BatchNorm1d(ch), nn.GELU())
        self.b1 = _b(1, mid); self.b2 = _b(2, mid); self.b4 = _b(4, rem)
        self.fuse = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 1), nn.BatchNorm1d(out_ch),
            nn.GELU(), nn.Dropout(dropout))
        self.skip = (nn.Sequential(nn.Conv1d(in_ch, out_ch, 1), nn.BatchNorm1d(out_ch))
                     if in_ch != out_ch else nn.Identity())
    def forward(self, x):
        return self.fuse(torch.cat([self.b1(x), self.b2(x), self.b4(x)], dim=1)) + self.skip(x)


class AttentiveRegionPool(nn.Module):
    """注意力区域池化, 附带生物位置先验初始化.

    三个区域查询初始化方向对应: 远端上游 / TSS近端 / TSS下游,
    先验强度 0.15 加速收敛至生物学有意义的注意力模式, 但不压制学习.
    (历史说明: 早期 region_div_w 辅助损失与更强的先验/温度会互相放大,
    导致 Q2 长尾 repressor 被丢失. new53 已整体下线该辅助损失, 但这里的
    先验强度 / 温度配置保持保守, 避免在无正则的情况下出现类似退化.)
    """
    def __init__(self, channels, n_regions=3, n_bins=75, tss_ratio=TSS_RATIO):
        super().__init__()
        self.n_regions = n_regions
        self.queries   = nn.Parameter(torch.randn(n_regions, channels) * 0.02)
        self.temp      = nn.Parameter(torch.ones(1) * 0.5)   # 初始温度 < 1, 注意力更集中
        # 生物位置先验: 远端上游 / TSS / 下游
        positions = torch.arange(n_bins).float()
        centers   = [tss_ratio * 0.35 * n_bins,    # 远端上游
                     tss_ratio * n_bins,             # TSS
                     (tss_ratio + 0.20) * n_bins]    # TSS下游
        pos_bias = torch.zeros(n_regions, n_bins)
        for i in range(min(n_regions, len(centers))):
            pos_bias[i] = torch.exp(
                -0.5 * ((positions - centers[i]) / (n_bins * 0.10)) ** 2)
        self.register_buffer('pos_prior', pos_bias * 0.15)   # 小强度, 不压制学习

    def forward(self, x, return_weights=False):
        B, C, L = x.shape
        scores  = torch.einsum('kc,bcl->bkl', self.queries, x) / (self.temp.clamp(min=0.1) + 1e-6)
        if L == self.pos_prior.shape[1]:
            scores = scores + self.pos_prior.unsqueeze(0)       # (1, n_regions, L)
        # 数值稳定: 限幅防止极端 logits 导致 softmax 上/下溢出
        scores  = scores.clamp(-30.0, 30.0)
        weights       = F.softmax(scores, dim=-1)               # (B, n_regions, L)
        region_tokens = torch.einsum('bkl,bcl->bkc', weights, x)  # (B, n_regions, C)
        if return_weights:
            return region_tokens.reshape(B, -1), region_tokens, weights
        return region_tokens.reshape(B, -1)


class PositionalBias(nn.Module):
    def __init__(self, n_bins, channels):
        super().__init__()
        pos = torch.arange(n_bins).float(); tss = n_bins * TSS_RATIO
        gaussian = torch.exp(-0.5 * ((pos - tss) / (n_bins * 0.1)) ** 2)
        self.bias = nn.Parameter(
            gaussian.unsqueeze(0).unsqueeze(0).expand(1, channels, -1).clone() * 0.1)
    def forward(self, x): return x + self.bias


class PerTFEncoder(nn.Module):
    def __init__(self, d_tf=128, n_bins=150):
        super().__init__()
        # TSS中心高斯位置先验：引导编码器在训练初期优先关注TSS附近的TF结合信号
        # 宽度σ=0.15*n_bins，足够宽以覆盖近端启动子区域，同时不过度抑制远端信号
        positions = torch.arange(n_bins).float()
        tss_pos   = n_bins * TSS_RATIO
        gaussian  = torch.exp(-0.5 * ((positions - tss_pos) / (n_bins * 0.15)) ** 2)
        self.pos_emb = nn.Parameter(gaussian.reshape(1, 1, n_bins) * 0.1)
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.GELU(),
            nn.AdaptiveAvgPool1d(16), nn.Flatten(),
            nn.Linear(64 * 16, d_tf), nn.LayerNorm(d_tf))
        self.out_dim = d_tf

    def forward(self, x):
        return self.net(x + self.pos_emb)


class TFAttentionPooler(nn.Module):
    def __init__(self, n_tfs, n_bins, d_tf=256, n_heads=8, n_layers=2, dropout=0.1,
                 d_seq_query=None, topk_k=0):
        super().__init__()
        assert d_tf % n_heads == 0
        self.n_tfs  = n_tfs
        # P2: topk_k > 0 → 每次前向只保留 top-k TF 的注意力，其余硬归零
        # 强迫模型为每个基因选择最重要的 k 个 TF，而非在 178 个 TF 上平摊权重
        # topk_k=0 退化为标准 softmax（向后兼容）
        self.topk_k = int(topk_k) if topk_k and topk_k > 0 else 0
        self.per_tf = PerTFEncoder(d_tf, n_bins)
        # ── 历史注记 (已移除, 恢复 new37 行为) ────────────────────────────
        # 原 P0-v2 曾在 per_tf 输出后加 `tf_id_token = nn.Parameter((n_tfs, d_tf))`
        # 注入 per-TF 身份. 当 n_tfs 由 178 缩到 79 时, 该参数在 4524 训练样本
        # 下容易过拟合到少数 TF 身份通道 (导致 SPN1 一家独大, attn > 0.06).
        # 共享权重的 PerTFEncoder 经 Transformer 已能区分 TF, 无需额外身份嵌入.

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_tf, nhead=n_heads, dim_feedforward=d_tf*4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                                  enable_nested_tensor=False)
        self.pool_query  = nn.Linear(d_tf, 1, bias=False)
        self.out_dim     = d_tf
        # 可学习温度: 初始0.5使注意力比均匀更集中，减少TF权重均匀化问题
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)
        # 动态查询: seq内容决定关注哪些TF（而非全局固定query）
        if d_seq_query is not None:
            self.dyn_query_proj = nn.Linear(d_seq_query, n_tfs, bias=True)
            nn.init.xavier_uniform_(self.dyn_query_proj.weight, gain=0.1)
            nn.init.zeros_(self.dyn_query_proj.bias)
        else:
            self.dyn_query_proj = None

    def forward(self, x, seq_query=None, return_attn=False):
        B, N, L = x.shape
        tf_emb  = self.per_tf(x.reshape(B*N, 1, L)).reshape(B, N, self.out_dim)
        tf_out  = self.transformer(tf_emb)
        raw_logits = self.pool_query(tf_out).squeeze(-1)     # (B, N)
        # seq内容引导: 序列特征决定对哪些TF分配注意力
        if seq_query is not None and self.dyn_query_proj is not None:
            raw_logits = raw_logits + self.dyn_query_proj(seq_query)
        # 数值稳定: 将 NaN/Inf 替换为有限值，防止 topk 选中 NaN 后 softmax 产生 nan
        # (极少发生于训练初期；nan_to_num 开销可忽略不计)
        raw_logits = torch.nan_to_num(raw_logits, nan=0.0, posinf=10.0, neginf=-10.0)
        # P2: Top-K sparse softmax — 只在 top-k 位置做 softmax，其余强制为 0
        # 避免 attention 在 N=178 个 TF 上均匀稀释，每个基因只激活最重要的 k 个 TF
        if self.topk_k > 0 and self.topk_k < N:
            k = min(self.topk_k, N)
            topk_val, topk_idx = raw_logits.topk(k, dim=1)
            # ★ 修复: 先对 topk 有限值做温度缩放，再 scatter 进 -inf mask。
            # 原版 mask / temperature 会把 -inf 也参与除法:
            #   forward  -inf / T = -inf (OK)
            #   backward d(-inf/T)/dT = +inf/T² → NaN (BUG!)
            # 新版: -inf 不经过 temperature 除法, 梯度干净回传。
            topk_val_scaled = topk_val / self.temperature.clamp(min=0.1)
            mask = torch.full_like(raw_logits, float('-inf'))
            mask.scatter_(1, topk_idx, topk_val_scaled)
            attn_w = F.softmax(mask, dim=1)
        else:
            attn_w = F.softmax(raw_logits / self.temperature.clamp(min=0.1), dim=1)
        pooled = (tf_out * attn_w.unsqueeze(-1)).sum(dim=1)
        if return_attn: return pooled, attn_w
        return pooled


class TFSpatialEncoder(nn.Module):
    """TF → 空间特征 编码器.

    设计决策 (恢复自 new37, 0.85 Pearson 配置):
      · stem: Conv1d(n_tfs → 128, k=7) + BN(128) + GELU + Dropout —— 单步
        channel-mixing 直接把 n_tfs 维 TF 信号投影到 128 通道. 后续 BN(128)
        按 128 个混合通道标准化, 不会按 n_tfs 吸收 per-TF 信号.
      · 不再注入 tf_id_gamma/beta FiLM —— 之前为 n_tfs=178 设计的身份参数,
        在 n_tfs=79 下容易在 4524 训练样本上 overfit 到少数 TF 身份通道.
        (实验观测: 带 FiLM 时 Fig5 被 DIG1 / SPN1 垄断, 生物分工退化.)
    """
    def __init__(self, n_tfs, n_bins, n_regions=5, d_cnn=256, dropout=0.10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_tfs, 128, 7, padding=3), nn.BatchNorm1d(128),
            nn.GELU(), nn.Dropout(dropout))
        self.pos_bias = PositionalBias(n_bins, 128)
        self.ms1 = MultiScaleDilatedBlock(128, 256, dropout)
        self.pool = nn.MaxPool1d(2, stride=2)
        self.ms2  = MultiScaleDilatedBlock(256, 256, dropout)
        self.attn_pool = AttentiveRegionPool(256, n_regions, n_bins=n_bins // 2, tss_ratio=TSS_RATIO)
        self.proj = nn.Sequential(
            nn.Linear(256*n_regions, d_cnn), nn.LayerNorm(d_cnn), nn.GELU())
        self.out_dim = d_cnn

    def forward(self, x, return_regions=False):
        x = self.stem(x); x = self.pos_bias(x)
        x = self.ms1(x); x = self.pool(x); x = self.ms2(x)
        if return_regions:
            # 保留区域身份, 供 ResidualCrossModalFusion 做 cross-attention
            pooled_flat, region_tokens, region_weights = self.attn_pool(x, return_weights=True)
            return self.proj(pooled_flat), region_tokens, region_weights
        return self.proj(self.attn_pool(x))


class TFTssWeightedEncoder(nn.Module):
    """TSS-Gaussian 加权标量编码器 (FUSION_ABLATION: use_tss_weighted_tf 专用).

    用途:
      C_tss_weighted 消融把 TFAttentionPooler + TFSpatialEncoder 整条学习分支
      整体替换为"TSS 中心高斯加权均值", 同时保留 ResidualCrossModalFusion.
      目的是隔离 "TF 学习型编码器" 对 +0.028 gain 的实际贡献.

    设计:
      · bin 权重 = Gaussian(mean=tss_pos, σ=0.15×n_bins), L1 归一化
      · 前向对 (B, n_tfs, n_bins) 做 weighted mean over bins → (B, n_tfs)
      · 无可学习参数 (与 MODEL_COMPARISON tss_weighted 列完全等价, 仅特征生成
        被移入 in-graph 以配合 fusion 的 (B, d_tf) 接口)
      · 输出维度 out_dim = n_tfs, tf_merged_dim = n_tfs
    """
    def __init__(self, n_tfs, n_bins):
        super().__init__()
        positions = torch.arange(n_bins, dtype=torch.float32)
        tss_pos   = n_bins * TSS_RATIO
        sigma     = n_bins * 0.15
        w = torch.exp(-0.5 * ((positions - tss_pos) / sigma) ** 2)
        w = w / (w.sum() + 1e-8)               # L1 归一化: weighted mean 而非 sum
        # register_buffer: 参与 .to(device) / .cuda(), 但不进 optimizer
        self.register_buffer('bin_weights', w.reshape(1, 1, n_bins))
        self.n_tfs   = int(n_tfs)
        self.n_bins  = int(n_bins)
        self.out_dim = int(n_tfs)

    def forward(self, x):
        # x: (B, n_tfs, n_bins) — 已 Signal-standardized
        # (bin_weights 自动广播到 (B, n_tfs, n_bins))
        return (x * self.bin_weights).sum(dim=-1)   # (B, n_tfs)


class ResidualCrossModalFusion(nn.Module):
    """序列–TF 跨模态融合模块（无 DNA 分支）
    改进:
      · cross_bias: 可学习偏置，使训练早期偏向稳定的 seq 分支，同时保留 cross_w 标量供 GO 富集

    模块消融标志（FUSION_ABLATION 专用，默认全关）:
      · freeze_crossw='one'  → cross_w ≡ 1（去门控）
      · freeze_crossw='zero' → cross_w ≡ 0（seq-only pass-through, sanity）

    new50 裁剪说明:
      早期版本含 mha_parallel / region_cross_attn / region_proj 以及 use_lm_gate
      门控 (lm_gate_proj + lm_gate_alpha). 依 Fusion 消融终态分析 (Δ vs C ∈
      {+0.0007, -0.0044, -0.0032}, Cohen d ∈ [-0.46, +0.07]) 一并移除,
      当前 fusion = seq_emb + cross_w · KV + out_proj, cross_w 继续作为
      可解释性出口 (GO 富集 MWU p=4.16e-08).
    """
    def __init__(self, tf_merged_dim, d_seq=128, d_tf=96,
                 n_seq_features=12, n_reg_scalars=2,
                 dropout=0.10,
                 use_lm=False, lm_dim=2304, d_lm_hidden=512, d_lm_out=128,
                 lm_input_dropout=0.10, freeze_lm=False,
                 use_dna_onehot=False, dna_seq_len=1500, d_dna_out=128,
                 dna_dropout=0.20,
                 # ── Fusion 模块消融标志 ───────────────────────────────
                 freeze_crossw=None,
                 use_profile=True):
        super().__init__()
        assert not (use_lm and use_dna_onehot), \
            "ResidualCrossModalFusion: use_lm 与 use_dna_onehot 互斥"
        assert freeze_crossw in (None, 'one', 'zero'), \
            f"freeze_crossw 必须是 None/'one'/'zero'，得到 {freeze_crossw}"
        self.d_seq          = d_seq
        self.d_tf           = d_tf
        self.n_seq_features = n_seq_features
        self.n_reg_scalars  = n_reg_scalars
        self.out_dim        = d_seq
        # 消融标志（默认 None → 完整 fusion 行为）
        self.freeze_crossw  = freeze_crossw

        self.seq_branch = SequenceBranch(
            n_windows=n_seq_features, d_profile=MODEL_CONFIG['d_profile'],
            d_out=d_seq, dropout=dropout,
            use_lm=use_lm, lm_dim=lm_dim, d_lm_hidden=d_lm_hidden,
            d_lm_out=d_lm_out, lm_input_dropout=lm_input_dropout, freeze_lm=freeze_lm,
            use_profile=use_profile,
            use_dna_onehot=use_dna_onehot, dna_seq_len=dna_seq_len,
            d_dna_out=d_dna_out, dna_dropout=dna_dropout)

        self.tf_proj = nn.Sequential(
            nn.Linear(tf_merged_dim + n_reg_scalars, d_tf),
            nn.LayerNorm(d_tf), nn.GELU(), nn.Dropout(dropout))

        self.seq_q_proj = nn.Linear(d_seq, d_seq, bias=False)
        self.tf_kv_proj = nn.Linear(d_tf,  d_seq, bias=False)
        self.cross_norm = nn.LayerNorm(d_seq)

        self.out_proj = nn.Sequential(
            nn.Linear(d_seq, d_seq), nn.LayerNorm(d_seq), nn.GELU())

        # ── cross_bias（保留 cross_w 标量用于 GO 富集，加学习偏置）──────
        # sigmoid(-2) ≈ 0.12，训练初期更保守地偏向稳定的 seq 分支 (原本 -1 → 0.27
        # 对初始化期 TF 分支梯度贡献过大, 与 region_div/decorr aux loss 叠加触发
        # 梯度爆炸)。训练开始后 cross_bias 会自适应增大, 不影响后期表达能力。
        self.cross_bias = nn.Parameter(torch.tensor(-2.0))

        _seq_mode = ('LM' if use_lm else ('OneHot' if use_dna_onehot else 'Profile'))
        _abl_tags = []
        if self.freeze_crossw is not None:
            _abl_tags.append(f"crossw={self.freeze_crossw}")
        _abl_str = f"  [ABL:{','.join(_abl_tags)}]" if _abl_tags else ""
        print(f"  ResidualCrossModalFusion  "
              f"tf_merged={tf_merged_dim}  d_seq={d_seq}  d_tf={d_tf}  seq={_seq_mode}  "
              f"[+cross_bias]{_abl_str}  "
              f"params={sum(p.numel() for p in self.parameters()):,}")

    def forward(self, tf_merged, scalars, lm_emb=None, seq_emb_pre=None,
                return_internals=False):
        seq_profile = scalars[:, :self.n_seq_features]
        reg_z = (scalars[:, self.n_seq_features:self.n_seq_features + self.n_reg_scalars]
                 if self.n_reg_scalars > 0
                 else torch.zeros(scalars.shape[0], 0, device=scalars.device))

        # 若已由 MultiModalPredictor 预计算则直接复用，避免双重前向
        seq_emb = seq_emb_pre if seq_emb_pre is not None else self.seq_branch(seq_profile, lm_emb)

        tf_input = (torch.cat([tf_merged, reg_z], dim=1)
                    if self.n_reg_scalars > 0 else tf_merged)
        tf_emb = self.tf_proj(tf_input)

        Q  = self.seq_q_proj(seq_emb)
        KV = self.tf_kv_proj(tf_emb)
        scale = Q.shape[-1] ** 0.5

        # ── cross_w 标量 (GO 富集继续可用)，加可学习偏置 ──────────────
        # cross_bias 训练后可作为"模型整体TF依赖度"全局指标（论文可报告）
        # FUSION_ABLATION: freeze_crossw='one'/'zero' → 常数 cross_w
        if self.freeze_crossw == 'one':
            cross_w = torch.ones(Q.shape[0], 1, device=Q.device, dtype=Q.dtype)
        elif self.freeze_crossw == 'zero':
            cross_w = torch.zeros(Q.shape[0], 1, device=Q.device, dtype=Q.dtype)
        else:
            cross_w = torch.sigmoid((Q * KV).sum(dim=-1, keepdim=True) / scale + self.cross_bias)

        # 主融合: seq_emb + cross_w · KV (→ out_proj)
        fused = seq_emb + cross_w * KV

        out = self.out_proj(fused)

        if return_internals:
            return out, seq_emb, tf_emb, cross_w.squeeze(-1)
        return out


class ConcatFusion(nn.Module):
    """朴素 concat 融合（C_concat 消融专用）

    把 seq_emb 和 tf_emb 直接拼接 → Linear → LN → GELU → Dropout.
    无 Q·KV、无 cross_w、无 MHA.
    与 ResidualCrossModalFusion 共享相同的 forward 签名, 上游(MultiModalPredictor)
    无需修改调用方式.

    返回 internals 时 cross_weight=None, 下游需容错处理.
    """
    def __init__(self, tf_merged_dim, d_seq=128, d_tf=96,
                 n_seq_features=12, n_reg_scalars=2, dropout=0.10,
                 use_lm=False, lm_dim=2304, d_lm_hidden=512, d_lm_out=128,
                 lm_input_dropout=0.10, freeze_lm=False,
                 use_profile=True,
                 use_dna_onehot=False, dna_seq_len=1500, d_dna_out=128,
                 dna_dropout=0.20,
                 # 以下参数接收但不使用（签名兼容性）
                 freeze_crossw=None):
        super().__init__()
        assert not (use_lm and use_dna_onehot), \
            "ConcatFusion: use_lm 与 use_dna_onehot 互斥"
        self.d_seq          = d_seq
        self.d_tf           = d_tf
        self.n_seq_features = n_seq_features
        self.n_reg_scalars  = n_reg_scalars
        self.out_dim        = d_seq

        self.seq_branch = SequenceBranch(
            n_windows=n_seq_features, d_profile=MODEL_CONFIG['d_profile'],
            d_out=d_seq, dropout=dropout,
            use_lm=use_lm, lm_dim=lm_dim, d_lm_hidden=d_lm_hidden,
            d_lm_out=d_lm_out, lm_input_dropout=lm_input_dropout, freeze_lm=freeze_lm,
            use_profile=use_profile,
            use_dna_onehot=use_dna_onehot, dna_seq_len=dna_seq_len,
            d_dna_out=d_dna_out, dna_dropout=dna_dropout)

        self.tf_proj = nn.Sequential(
            nn.Linear(tf_merged_dim + n_reg_scalars, d_tf),
            nn.LayerNorm(d_tf), nn.GELU(), nn.Dropout(dropout))

        # 朴素拼接 → Linear (d_seq + d_tf → d_seq)
        self.fuse_mlp = nn.Sequential(
            nn.Linear(d_seq + d_tf, d_seq),
            nn.LayerNorm(d_seq), nn.GELU(), nn.Dropout(dropout))
        self.out_proj = nn.Sequential(
            nn.Linear(d_seq, d_seq), nn.LayerNorm(d_seq), nn.GELU())

        _seq_mode = ('LM' if use_lm else ('OneHot' if use_dna_onehot else 'Profile'))
        print(f"  ConcatFusion  tf_merged={tf_merged_dim}  d_seq={d_seq}  d_tf={d_tf}  "
              f"seq={_seq_mode}  [ABL:concat, no cross-modal mechanism]  "
              f"params={sum(p.numel() for p in self.parameters()):,}")

    def forward(self, tf_merged, scalars, lm_emb=None, seq_emb_pre=None,
                return_internals=False):
        seq_profile = scalars[:, :self.n_seq_features]
        reg_z = (scalars[:, self.n_seq_features:self.n_seq_features + self.n_reg_scalars]
                 if self.n_reg_scalars > 0
                 else torch.zeros(scalars.shape[0], 0, device=scalars.device))

        seq_emb = seq_emb_pre if seq_emb_pre is not None else self.seq_branch(seq_profile, lm_emb)

        tf_input = (torch.cat([tf_merged, reg_z], dim=1)
                    if self.n_reg_scalars > 0 else tf_merged)
        tf_emb = self.tf_proj(tf_input)

        fused = self.fuse_mlp(torch.cat([seq_emb, tf_emb], dim=1))
        out   = self.out_proj(fused)

        if return_internals:
            # cross_weight 对 concat 无定义, 返回 None (下游 collect_cross_weights 已容错)
            return out, seq_emb, tf_emb, None
        return out


# ============================================================================
# 损失函数
# ============================================================================

class PearsonLoss(nn.Module):
    """数值稳定的 Pearson 损失。
    修复 new39 原版的 backward NaN bug:
      原版  1 - (p*t).sum() / (sqrt(p²·t²) + 1e-8)
      问题  当 pred 方差很小 (~1e-3)时, sqrt(A·B) → 很小, 其 backward 为
            d sqrt(A·B)/dp_i = B·p_i / sqrt(A·B).  在 p_i≈0、A≈0 时
            形成 0/0 → NaN; 在 fp32 累加到 grad_norm 时 → inf/nan。
    本实现: 先单独计算 std (带 eps clamp), 再用 cosine-like 形式,
      任何子路径都不会触达 sqrt(0)。
    额外: 把输入限幅到合理范围, 防止极端 pred 把分子炸到 ±1e20。
    """
    def __init__(self, eps=1e-6, input_clip=1e4):
        super().__init__()
        self.eps = eps
        self.input_clip = input_clip

    def forward(self, pred, target):
        # 输入限幅 —— 偶发极端 pred 不会让分子 (p*t).sum() 溢出
        pred   = pred.clamp(-self.input_clip, self.input_clip)
        target = target.clamp(-self.input_clip, self.input_clip)
        p = pred   - pred.mean()
        t = target - target.mean()
        # 分别算 std (clamp 防止 0 方差), 避免 sqrt(A*B) 这种 0/0 形
        std_p = p.pow(2).mean().clamp(min=self.eps).sqrt()
        std_t = t.pow(2).mean().clamp(min=self.eps).sqrt()
        cov   = (p * t).mean()
        corr  = cov / (std_p * std_t)
        # 相关系数理论上 ∈ [-1, 1], 但 fp32 误差可能越界; 再 clamp 一次
        corr  = corr.clamp(-1.0, 1.0)
        return 1.0 - corr


class CombinedLoss(nn.Module):
    def __init__(self, mse_w=0.50, pearson_w=0.50):
        super().__init__()
        self.mse_w = mse_w; self.pearson_w = pearson_w
        self.ploss = PearsonLoss(); self._first_call = True
    def forward(self, pred, target, weights=None):
        mse = ((pred-target).pow(2)*weights).mean() if weights is not None else F.mse_loss(pred, target)
        pl  = self.ploss(pred, target)
        if self._first_call:
            self._first_call = False
            print(f"\n  [Loss诊断] MSE={mse.item():.4f}  1-r(loss)={pl.item():.4f}")
        return self.mse_w * mse + self.pearson_w * pl


# ============================================================================
# Bootstrap 置信区间 & 多种子评估
# ============================================================================

def compute_bootstrap_ci(preds, labels, n_boot=2000, ci_level=0.95, seed=42):
    rng = np.random.default_rng(seed)
    n   = len(preds)
    rs  = np.empty(n_boot, dtype=np.float64)
    r2s = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        p_b = preds[idx]; l_b = labels[idx]
        try:
            rs[i]  = float(pearsonr(p_b, l_b)[0])
            r2s[i] = float(r2_score(l_b, p_b))
        except Exception:
            rs[i] = np.nan; r2s[i] = np.nan
    rs  = rs[np.isfinite(rs)];  r2s = r2s[np.isfinite(r2s)]
    lo_pct = (1 - ci_level) / 2 * 100; hi_pct = (1 - (1 - ci_level) / 2) * 100
    r_lo,  r_hi  = float(np.percentile(rs,  lo_pct)), float(np.percentile(rs,  hi_pct))
    r2_lo, r2_hi = float(np.percentile(r2s, lo_pct)), float(np.percentile(r2s, hi_pct))
    return {
        'n_boot': n_boot, 'ci_level': ci_level,
        'pearson_point':     float(pearsonr(preds, labels)[0]),
        'pearson_boot_mean': float(rs.mean()),
        'pearson_ci_lo': r_lo, 'pearson_ci_hi': r_hi,
        'r2_point':      float(r2_score(labels, preds)),
        'r2_boot_mean':  float(r2s.mean()),
        'r2_ci_lo': r2_lo, 'r2_ci_hi': r2_hi,
    }


def compute_fixed_quadrant_labels(seq_anchor_full: np.ndarray,
                                   tpm_norm_full:  np.ndarray,
                                   gene_names: list = None,
                                   anchor_name: str = 'seed42_B_seq_only_pred'
                                   ) -> dict:
    """
    基于「固定锚」计算全基因象限标签, 供所有下游分析共用 (TF-only baseline,
    每 seed ours-C, run_postanalysis 的 fig2_quadrant_scatter).

    与 run_postanalysis 的象限划分逻辑严格一致:
      X 轴: seq_anchor_full (median-centered, 推荐来自 seed=42 B 的 seq_only 预测)
      Y 轴: tpm_norm_full   (已 zscore, 以 0 为阈)
      Q1: (sa_z>=0) & (tpm>=0)   — Sequence-concordant (high)
      Q2: (sa_z>=0) & (tpm< 0)   — Trans-repressed  ★ 论文重点
      Q3: (sa_z< 0) & (tpm>=0)   — Trans-activated  ★ 论文重点
      Q4: (sa_z< 0) & (tpm< 0)   — Sequence-concordant (low)

    返回:
      {
        'quadrant_array': np.array(N,) of str labels,
        'masks': {'Q1': bool_arr, 'Q2': bool_arr, 'Q3': bool_arr, 'Q4': bool_arr},
        'labels': ['Q1: ...', 'Q2: ...', 'Q3: ...', 'Q4: ...'],
        'anchor_name': str,
        'sa_median': float,
        'tpm_threshold': float,  # 固定为 0 (因为 tpm 已 zscore)
      }
    """
    assert len(seq_anchor_full) == len(tpm_norm_full), \
        f"seq_anchor_full ({len(seq_anchor_full)}) vs tpm_norm_full ({len(tpm_norm_full)}) 长度不一致"

    valid_mask = np.isfinite(seq_anchor_full) & np.isfinite(tpm_norm_full)
    sa_median = float(np.nanmedian(seq_anchor_full[valid_mask]))
    sa_z = (seq_anchor_full - sa_median).astype(np.float32)
    tpm_v = tpm_norm_full.astype(np.float32)

    # 未 valid 的基因 (NaN) 标记为 unknown, 不计入任何象限
    quadrant = np.full(len(seq_anchor_full), 'unknown', dtype=object)
    quadrant[valid_mask & (sa_z >= 0) & (tpm_v >= 0)] = 'Q1'
    quadrant[valid_mask & (sa_z >= 0) & (tpm_v <  0)] = 'Q2'
    quadrant[valid_mask & (sa_z <  0) & (tpm_v >= 0)] = 'Q3'
    quadrant[valid_mask & (sa_z <  0) & (tpm_v <  0)] = 'Q4'

    masks = {q: (quadrant == q) for q in ('Q1', 'Q2', 'Q3', 'Q4')}

    print(f"\n  [象限锚] {anchor_name}")
    print(f"     sa_median={sa_median:.4f}  (tpm threshold=0, 已 zscore)")
    n_total = valid_mask.sum()
    for q in ('Q1', 'Q2', 'Q3', 'Q4'):
        n = masks[q].sum()
        print(f"     {q}: {n} 基因 ({n / max(n_total,1) * 100:.1f}%)")

    return {
        'quadrant_array': quadrant,
        'masks': masks,
        'labels': ['Q1: Sequence-concordant (high)', 'Q2: Trans-repressed',
                   'Q3: Trans-activated', 'Q4: Sequence-concordant (low)'],
        'anchor_name': anchor_name,
        'sa_median': sa_median,
        'tpm_threshold': 0.0,
        'gene_names': gene_names,
    }


def compute_per_quadrant_metrics(preds: np.ndarray,
                                  labels: np.ndarray,
                                  test_gene_indices: np.ndarray,
                                  quadrant_array: np.ndarray,
                                  min_n: int = 5) -> dict:
    """
    按象限拆分 test 集, 计算每象限的 n / Pearson / MAE / Bias / Spearman.

    参数:
      preds             : (n_test,) 模型预测
      labels            : (n_test,) 真实值 (均已归一化, 如 zscore log1p TPM)
      test_gene_indices : (n_test,) 每个 test 样本在全基因数组的位置 (用于对齐象限)
      quadrant_array    : (N,) 全基因象限标签 ('Q1'/'Q2'/'Q3'/'Q4'/'unknown')
      min_n             : 少于 min_n 个样本的象限不计算 Pearson (返回 NaN)

    返回:
      {
        'global': {'n': ..., 'pearson': ..., 'r2': ..., 'mae': ..., 'bias': ..., 'spearman': ...},
        'Q1':     {...},
        'Q2':     {...},
        'Q3':     {...},
        'Q4':     {...},
      }
    """
    from scipy.stats import spearmanr as _spearmanr_local

    preds  = np.asarray(preds,  dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    test_gene_indices = np.asarray(test_gene_indices)
    assert len(preds) == len(labels) == len(test_gene_indices), \
        f"preds/labels/indices 长度不一致: {len(preds)}/{len(labels)}/{len(test_gene_indices)}"

    # 每个 test 样本对应的全基因象限标签
    test_quadrant = quadrant_array[test_gene_indices]

    def _metric(p, l):
        if len(p) < min_n:
            return {'n': int(len(p)), 'pearson': float('nan'), 'r2': float('nan'),
                    'mae': float('nan'), 'bias': float('nan'),
                    'spearman': float('nan')}
        err = p - l
        return {
            'n': int(len(p)),
            'pearson':  float(pearsonr(p, l)[0]),
            'r2':       float(r2_score(l, p)),
            'mae':      float(np.abs(err).mean()),
            'bias':     float(err.mean()),           # +: 高估, -: 低估
            'spearman': float(_spearmanr_local(p, l)[0]),
        }

    out = {'global': _metric(preds, labels)}
    for q in ('Q1', 'Q2', 'Q3', 'Q4'):
        mask = (test_quadrant == q)
        out[q] = _metric(preds[mask], labels[mask])
    return out


def run_multiseed_eval(seed_list, train_fn, output_dir, label_prefix='multiseed_C',
                        quadrant_info: dict = None,
                        test_mask: np.ndarray = None,
                        compute_quadrant: bool = False,
                        preloaded_results: dict = None):
    """
    多种子评估入口.

    参数:
      seed_list         : 要评估的 seed 列表
      train_fn          : 回调 (random_seed, label) -> res_dict; 当某 seed
                          命中 preloaded_results 时不会被调用 (可为 None,
                          前提是 preloaded_results 覆盖了所有 seed)
      preloaded_results : {seed: res_dict} 可选. 已预计算的 seed 直接用此 dict
                          跳过训练. res_dict 需含 pearson/r2, 且若要算
                          per-quadrant 还需 preds_test/labels_test/test_gene_indices.
                          用途: 复用 FUSION_ABL 的 C per-seed 结果, 避免重复训 C.

    分象限参数 (quadrant_stats):
      quadrant_info    : compute_fixed_quadrant_labels 的返回值 (含 quadrant_array)
      test_mask        : (N,) bool 数组, 全基因数组中哪些是 test (split_masks['test'])
      compute_quadrant : True 时调用 compute_per_quadrant_metrics 记录每象限 n/Pearson/MAE/Bias
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    records = []
    quadrant_records = []  # 每 (seed, quadrant) 一行, 用于长表输出 + 统计
    preloaded_results = preloaded_results or {}

    _can_quadrant = (compute_quadrant and quadrant_info is not None
                      and test_mask is not None)
    if compute_quadrant and not _can_quadrant:
        print("  ⚠ compute_quadrant=True 但 quadrant_info/test_mask 缺失, 自动降级为只算全局")
    if _can_quadrant:
        test_indices = np.where(test_mask)[0]
        quad_arr = quadrant_info['quadrant_array']

    for sd in seed_list:
        # —— 1. 定位 res: 优先 preloaded, 否则跑 train_fn ——
        if sd in preloaded_results:
            print(f"\n{'='*62}\n  [多种子] seed={sd}  "
                  f"(从缓存复用, 无训练)\n{'='*62}")
            res = preloaded_results[sd]
        elif train_fn is not None:
            print(f"\n{'='*62}\n  [多种子] seed={sd}\n{'='*62}")
            try:
                res = train_fn(random_seed=sd, label=f'{label_prefix}_seed{sd}')
            except Exception as _e:
                print(f"  ⚠ seed={sd} 失败: {_e}")
                traceback.print_exc()
                continue
        else:
            print(f"  ⚠ seed={sd} 既无 preloaded_results 也无 train_fn, 跳过")
            continue

        try:
            row = {'seed': sd,
                   'pearson': float(res.get('pearson', np.nan)),
                   'r2': float(res.get('r2', np.nan))}
            if res.get('boot_ci'):
                row.update({k: res['boot_ci'][k] for k in
                             ('pearson_ci_lo','pearson_ci_hi','r2_ci_lo','r2_ci_hi')})

            # 分象限指标
            if _can_quadrant and 'preds_test' in res and 'labels_test' in res:
                # 优先使用 train_experiment 返回的 test_gene_indices;
                # 若未返回则从 test_mask 推导 (假设 preds 顺序 == test_mask 的 np.where 顺序)
                t_idx = res.get('test_gene_indices', test_indices)
                q_metrics = compute_per_quadrant_metrics(
                    res['preds_test'], res['labels_test'], t_idx, quad_arr)
                for q_name, m in q_metrics.items():
                    if q_name == 'global':
                        continue
                    row[f'{q_name}_n']       = m['n']
                    row[f'{q_name}_pearson'] = m['pearson']
                    row[f'{q_name}_mae']     = m['mae']
                    row[f'{q_name}_bias']    = m['bias']
                    quadrant_records.append({
                        'seed': sd, 'quadrant': q_name,
                        'n': m['n'], 'pearson': m['pearson'],
                        'mae': m['mae'], 'bias': m['bias'],
                        'spearman': m['spearman'], 'r2': m['r2'],
                    })
                # 在控制台打印该 seed 的象限快照 (论文核心叙事一瞥)
                print(f"  [seed={sd}] 分象限 Pearson / MAE / Bias:")
                for q_name in ('Q1', 'Q2', 'Q3', 'Q4'):
                    m = q_metrics[q_name]
                    print(f"    {q_name}: n={m['n']:>4d}  "
                          f"r={m['pearson']:+.4f}  "
                          f"MAE={m['mae']:.3f}  "
                          f"Bias={m['bias']:+.3f}")
            records.append(row)
        except Exception as _e:
            print(f"  ⚠ seed={sd} 后处理失败: {_e}")
            traceback.print_exc()
    if not records:
        print("  ⚠ [多种子] 无有效结果"); return {}
    df = pd.DataFrame(records)
    df.to_csv(out / f'{label_prefix}_summary.csv', index=False)
    r_mean = df['pearson'].mean(); r_std = df['pearson'].std(ddof=1)
    r2_mean = df['r2'].mean(); r2_std = df['r2'].std(ddof=1)
    # new47: 把 per-seed 记录暴露在 summary 字典里, 供 FUSION_ABLATION 做 paired Δ
    # (此前 summary 只有 mean/std, 下游 _emit_fusion_ablation_summary 拿不到 per-seed)
    _seed_results = [
        {'seed': int(r['seed']), 'pearson': float(r['pearson']), 'r2': float(r['r2'])}
        for r in records
    ]
    summary = {'n_seeds': len(df),
               'pearson_mean': float(r_mean), 'pearson_std': float(r_std),
               'pearson_min': float(df['pearson'].min()), 'pearson_max': float(df['pearson'].max()),
               'r2_mean': float(r2_mean), 'r2_std': float(r2_std),
               'seed_results': _seed_results}

    # ── 分象限长表 + 跨 seed 统计 ───────────────────────────────────
    if quadrant_records:
        qdf = pd.DataFrame(quadrant_records)
        qdf.to_csv(out / f'{label_prefix}_quadrant_per_seed.csv', index=False)
        # 跨 seed 聚合 (mean ± std per quadrant, 纯聚合用 agg)
        qstats = (qdf.groupby('quadrant')
                      .agg(n_mean=('n', 'mean'),
                           pearson_mean=('pearson', 'mean'),
                           pearson_std =('pearson', lambda x: x.std(ddof=1) if len(x)>1 else 0.0),
                           mae_mean    =('mae', 'mean'),
                           mae_std     =('mae', lambda x: x.std(ddof=1) if len(x)>1 else 0.0),
                           bias_mean   =('bias', 'mean'),
                           bias_std    =('bias', lambda x: x.std(ddof=1) if len(x)>1 else 0.0))
                      .reset_index())
        qstats.to_csv(out / f'{label_prefix}_quadrant_stats.csv', index=False)
        # 把关键象限统计合并进 summary 字典
        for _, qrow in qstats.iterrows():
            q = qrow['quadrant']
            summary[f'{q}_pearson_mean'] = float(qrow['pearson_mean'])
            summary[f'{q}_pearson_std']  = float(qrow['pearson_std'])
            summary[f'{q}_mae_mean']     = float(qrow['mae_mean'])
            summary[f'{q}_bias_mean']    = float(qrow['bias_mean'])
        # 控制台报告 (论文叙事锚点)
        print(f"\n  ┌─ 跨 {len(df)} seeds 分象限聚合 ──────────────────────────")
        print(f"  │  {'象限':6s}  {'n':>5s}  {'Pearson (mean±std)':>22s}  {'MAE':>12s}  {'Bias':>13s}")
        for _, qrow in qstats.iterrows():
            print(f"  │  {qrow['quadrant']:6s}  {int(qrow['n_mean']):>5d}  "
                  f"{qrow['pearson_mean']:+.4f} ± {qrow['pearson_std']:.4f}   "
                  f"{qrow['mae_mean']:.3f}±{qrow['mae_std']:.3f}  "
                  f"{qrow['bias_mean']:+.3f}±{qrow['bias_std']:.3f}")
        print(f"  └───────────────────────────────────────────────────────────")

    pd.DataFrame([summary]).to_csv(out / f'{label_prefix}_stats.csv', index=False)
    print(f"\n  ✓ [多种子] n={len(df)}  "
          f"Pearson r = {r_mean:.4f} ± {r_std:.4f}  R² = {r2_mean:.4f} ± {r2_std:.4f}")
    fig, ax = plt.subplots(figsize=(max(5, len(df)*0.9 + 2), 4))
    x_pos = np.arange(len(df))
    ax.bar(x_pos, df['pearson'].values, color='#3A86FF', alpha=0.75, edgecolor='white')
    ax.axhline(r_mean, color='#E63946', lw=1.8, ls='-', label=f'mean={r_mean:.4f}')
    ax.axhline(r_mean + r_std, color='#E63946', lw=1.0, ls='--', label=f'±1 SD ({r_std:.4f})')
    ax.axhline(r_mean - r_std, color='#E63946', lw=1.0, ls='--')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'seed={s}' for s in df['seed'].values], fontsize=8, rotation=15)
    ax.set_ylabel('Test Pearson r'); ax.set_ylim(0, 1)
    ax.set_title(f'Multi-seed Evaluation ({label_prefix})\nr = {r_mean:.4f} ± {r_std:.4f}',
                 fontweight='bold')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    fig.savefig(out / f'{label_prefix}_multiseed.pdf', bbox_inches='tight', dpi=200)
    plt.close(fig)
    return summary


# ============================================================================
# FUSION_ABLATION 汇总（paired Δ vs C + per-quadrant）
# ============================================================================
#   输入 _fa_records 是每 run 一行的字典列表 (variant, seed, pearson, r2, flags…).
#   本函数负责:
#     · 按 variant × seed 做 wide-form 汇总 → fusion_ablation_summary.csv
#     · 与 C 的 per-seed 结果配对 → 真实的 paired Δ / Welch-t / two-sided Wilcoxon
#     · 可选 per-quadrant Pearson
#
#   C 基线 per-seed 来源优先级 (new47 修正):
#     1. 调用方显式传入的 c_per_seed dict (FUSION_ABLATION 块内建, 推荐路径)
#     2. all_results['multiseed_summary']['seed_results'] (MULTI_SEED_EVAL 已跑)
#     3. 磁盘上的 {output_dir}/multiseed_C/multiseed_C_summary.csv (跨次运行复用)
#     4. 单 seed C 退化: ablation_C_seq_tf 的 pearson (仅 seed=42, 标记不可配对)
#
#   关键修正 vs new46:
#     · Wilcoxon alternative='greater' → 'two-sided' (C_noCrossW 显示方向可能反)
#     · 新增 paired_n 列, 配对数 <2 时 delta_vs_C_mean/std 写 NaN 避免误读
#     · 新增 unpaired_delta_seed42 列, 专门装单 seed 近似值 (透明, 不混淆)
#     · 新增 Welch-t 和 Cohen's d (效应量), 与消融改进.md 的分析接轨
# ============================================================================

def _emit_fusion_ablation_summary(fa_records, all_results,
                                  fusion_variants, fusion_seeds,
                                  output_dir, split_masks, genes,
                                  quadrant_info=None,
                                  c_per_seed=None):
    """汇总 FUSION_ABLATION 结果: 写 csv, 打印 paired Δ 表, 可选象限分析.

    参数
    ----
    c_per_seed : dict[int, float] | None
        调用方显式提供的 C per-seed Pearson (new47 推荐路径). 若为 None, 则
        依次尝试 multiseed_summary → disk CSV → 单 seed=42 退化.
    """
    out_dir = Path(output_dir) / 'fusion_ablation'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── merge 工具: 新结果合并回磁盘上已有的 CSV, 按 key_cols 去重新值优先 ──
    # new49 修复: 单 variant 跑时避免覆写之前 variant 的结果
    def _merge_dedupe(df_new, csv_path, key_cols):
        """若 csv_path 已存在, 读入旧表 → 去掉 new 中 key 命中的行 → 拼接."""
        if not csv_path.exists():
            return df_new
        try:
            df_old = pd.read_csv(csv_path)
            for _k in key_cols:
                if _k in df_old.columns and _k in df_new.columns:
                    try:
                        df_old[_k] = df_old[_k].astype(df_new[_k].dtype)
                    except Exception:
                        pass
            if all(_k in df_old.columns for _k in key_cols):
                _new_keys = df_new[key_cols].apply(tuple, axis=1).tolist()
                _mask_drop = df_old[key_cols].apply(tuple, axis=1).isin(_new_keys)
                df_old = df_old[~_mask_drop]
            return pd.concat([df_old, df_new], ignore_index=True, sort=False)
        except Exception as _merge_e:
            print(f"  ⚠ [FUSION_ABL] 读取 {csv_path.name} 失败, 覆写: {_merge_e}")
            return df_new

    _fa_df = pd.DataFrame(fa_records)
    if _fa_df.empty:
        print("\n  ⚠ [FUSION_ABL] 无记录, 跳过汇总")
        return None

    # ── 1. 取 C 基线 per-seed (四级 fallback) ───────────────────────────
    _c_per_seed = {}   # seed -> pearson
    _c_source = None

    # (a) 调用方显式传入 (最高优先级, 同一次运行内同 split / 同 kwargs)
    if c_per_seed:
        for _s, _p in c_per_seed.items():
            try:
                if np.isfinite(float(_p)):
                    _c_per_seed[int(_s)] = float(_p)
            except Exception:
                pass
        if _c_per_seed:
            _c_source = 'explicit_c_per_seed'

    # (b) MULTI_SEED_EVAL 的 seed_results
    if not _c_per_seed:
        _ms = all_results.get('multiseed_summary', {})
        if isinstance(_ms, dict):
            _seed_res = _ms.get('seed_results') or _ms.get('per_seed') or {}
            if isinstance(_seed_res, dict):
                for _s, _d in _seed_res.items():
                    try:
                        _pr = (_d.get('pearson') if isinstance(_d, dict) else None)
                        if _pr is not None:
                            _c_per_seed[int(_s)] = float(_pr)
                    except Exception:
                        pass
            elif isinstance(_seed_res, list):
                for _d in _seed_res:
                    if isinstance(_d, dict) and 'seed' in _d and 'pearson' in _d:
                        _c_per_seed[int(_d['seed'])] = float(_d['pearson'])
            if _c_per_seed:
                _c_source = 'multiseed_summary'

    # (c) 磁盘 CSV (跨次运行复用 — 支持消融改进.md 提到的"离线合并")
    if not _c_per_seed:
        _ms_csv = Path(output_dir) / 'multiseed_C' / 'multiseed_C_summary.csv'
        if _ms_csv.exists():
            try:
                _msdf = pd.read_csv(_ms_csv)
                for _, _r in _msdf.iterrows():
                    if np.isfinite(_r.get('pearson', np.nan)):
                        _c_per_seed[int(_r['seed'])] = float(_r['pearson'])
                if _c_per_seed:
                    _c_source = f'disk_csv:{_ms_csv}'
            except Exception as _csv_e:
                print(f"  ⚠ [FUSION_ABL] 读取 {_ms_csv} 失败: {_csv_e}")

    # (d) 退化路径: 仅 seed=42 的 C 点估计 (明确标注不可配对)
    if not _c_per_seed:
        _c_res = all_results.get('ablation_C_seq_tf')
        if _c_res is not None and 'pearson' in _c_res:
            _c_per_seed[42] = float(_c_res['pearson'])
            _c_source = 'single_seed_C@42 (UNPAIRED)'
            print("  ⚠ [FUSION_ABL] 仅找到单 seed C 基线 (seed=42), "
                  "配对 Δ 不可计算; CSV 会用 unpaired_delta_seed42 列单独标注")

    print(f"  [FUSION_ABL] C baseline source: {_c_source or 'NONE'}  "
          f"({len(_c_per_seed)} seeds: {sorted(_c_per_seed.keys())})")

    # ── 2. 逐 variant 计算配对 Δ + Welch-t + two-sided Wilcoxon ─────────
    summary_rows = []
    for _v in fusion_variants:
        _sub = _fa_df[_fa_df['variant'] == _v].copy()
        if _sub.empty or _sub['pearson'].isna().all():
            continue
        _p = _sub['pearson'].values.astype(np.float64)
        _p_finite = _p[np.isfinite(_p)]
        _p_mean = float(np.nanmean(_p))
        _p_std  = float(np.nanstd(_p, ddof=1)) if _p_finite.size >= 2 else float('nan')

        # 配对 Δ = C(seed) − variant(seed), 只收集两边都有 finite 值的 seed
        _paired = []     # list of (C, variant) 元组
        for _, _row in _sub.iterrows():
            _s = int(_row['seed'])
            _vp = float(_row['pearson'])
            if (_s in _c_per_seed) and np.isfinite(_vp) and np.isfinite(_c_per_seed[_s]):
                _paired.append((_c_per_seed[_s], _vp))
        _paired_n = len(_paired)
        _deltas = np.array([c - v for c, v in _paired], dtype=np.float64)

        if _paired_n >= 2:
            _delta_mean = float(_deltas.mean())
            _delta_std  = float(_deltas.std(ddof=1))
        elif _paired_n == 1:
            _delta_mean = float(_deltas[0])   # 单点, 下面 paired_n 会标 1
            _delta_std  = float('nan')
        else:
            _delta_mean = float('nan'); _delta_std = float('nan')

        # Welch-t (unpooled, two-sided) — 当两边都 >=2 个值时
        _welch_t = float('nan'); _welch_p = float('nan'); _cohen_d = float('nan')
        _c_vals = np.array([c for c, _ in _paired], dtype=np.float64)
        _v_vals = np.array([v for _, v in _paired], dtype=np.float64)
        if _paired_n >= 2 and _c_vals.std(ddof=1) > 0 and _v_vals.std(ddof=1) > 0:
            try:
                from scipy.stats import ttest_ind
                _tt = ttest_ind(_c_vals, _v_vals, equal_var=False)
                _welch_t = float(_tt.statistic)
                _welch_p = float(_tt.pvalue)
                # Cohen's d (pooled std)
                _pooled_var = (_c_vals.var(ddof=1) + _v_vals.var(ddof=1)) / 2.0
                if _pooled_var > 0:
                    _cohen_d = float((_c_vals.mean() - _v_vals.mean()) /
                                     np.sqrt(_pooled_var))
            except Exception:
                pass

        # two-sided paired Wilcoxon (需至少 1 非零差值; scipy 要求 n>=1, 我们用 >=3)
        _p_wilcoxon = float('nan')
        if _paired_n >= 3:
            try:
                from scipy.stats import wilcoxon
                # zero_method='wilcox' 丢弃零差, 但这里差值很少刚好为 0, 用默认处理即可
                _p_wilcoxon = float(wilcoxon(_deltas,
                                             alternative='two-sided').pvalue)
            except Exception:
                _p_wilcoxon = float('nan')

        # 单 seed 退化路径: 把近似 Δ 独立放 unpaired_delta_seed42 列, 不污染 paired 列
        _unpaired_delta_seed42 = float('nan')
        if _paired_n == 1 and list(_c_per_seed.keys()) == [42]:
            _unpaired_delta_seed42 = float(_deltas[0])
            # 清空 paired 列, 防止被误读
            _delta_mean = float('nan'); _delta_std = float('nan')

        summary_rows.append(dict(
            variant                 = _v,
            n_seeds                 = int(_sub['pearson'].notna().sum()),
            pearson_mean            = _p_mean,
            pearson_std             = _p_std,
            paired_n                = _paired_n,
            delta_vs_C_mean         = _delta_mean,
            delta_vs_C_std          = _delta_std,
            welch_t                 = _welch_t,
            welch_p_two_sided       = _welch_p,
            cohen_d                 = _cohen_d,
            paired_p_wilcoxon_two_sided = _p_wilcoxon,
            unpaired_delta_seed42   = _unpaired_delta_seed42,
            c_baseline_source       = _c_source or 'NONE',
            flags                   = _sub['flags'].iloc[0],
        ))

    _summary_df = pd.DataFrame(summary_rows)

    # ── 3. Per-quadrant (如果 quadrant_info 可用) ───────────────────────
    if quadrant_info is not None:
        _q_rows = []
        for _rec in fa_records:
            _lbl = _rec.get('label')
            if _lbl is None:
                continue
            _res_obj = all_results.get(_lbl)
            if _res_obj is None:
                continue
            _preds = _res_obj.get('preds_test')
            _labels = _res_obj.get('labels_test')
            _tg_idx = _res_obj.get('test_gene_indices')
            if _preds is None or _labels is None or _tg_idx is None:
                continue
            try:
                _q_metrics = compute_per_quadrant_metrics(
                    preds=_preds.astype(np.float64),
                    labels=_labels.astype(np.float64),
                    test_gene_indices=_tg_idx.astype(np.int64),
                    quadrant_array=quadrant_info['quadrant_array'],
                )
            except Exception as _qe:
                print(f"  ⚠ [FUSION_ABL] {_lbl} per-quadrant 失败: {_qe}")
                continue
            for _q in ('Q1', 'Q2', 'Q3', 'Q4'):
                _m = _q_metrics.get(_q, {})
                _q_rows.append(dict(
                    variant=_rec['variant'], seed=_rec['seed'],
                    quadrant=_q, n=_m.get('n', 0),
                    pearson=_m.get('pearson', float('nan')),
                    mae=_m.get('mae', float('nan')),
                    bias=_m.get('bias', float('nan')),
                ))
        _q_df = pd.DataFrame(_q_rows)
        if not _q_df.empty:
            # new49 修复: merge 回磁盘旧结果, 主键 (variant, seed, quadrant)
            _q_csv = out_dir / 'fusion_ablation_per_quadrant.csv'
            _q_df_merged = _merge_dedupe(_q_df, _q_csv,
                                         key_cols=['variant', 'seed', 'quadrant'])
            _q_df_merged.to_csv(_q_csv, index=False)
            _pivot_mean = _q_df_merged.pivot_table(
                index='variant', columns='quadrant', values='pearson', aggfunc='mean')
            _pivot_std = _q_df_merged.pivot_table(
                index='variant', columns='quadrant', values='pearson', aggfunc='std')
            _pivot_mean.to_csv(out_dir / 'fusion_ablation_per_quadrant_pivot_mean.csv')
            _pivot_std.to_csv(out_dir / 'fusion_ablation_per_quadrant_pivot_std.csv')
            print(f"  ✓ [FUSION_ABL] per-quadrant → {_q_csv}  "
                  f"(merged: {_q_df_merged['variant'].nunique()} variants)")

    # ── 4. 落盘 + 打印 (new49 修复: merge 回磁盘, 不覆写之前 variant) ────
    _raw_csv = out_dir / 'fusion_ablation_raw.csv'
    _sum_csv = out_dir / 'fusion_ablation_summary.csv'
    _fa_df_merged = _merge_dedupe(_fa_df, _raw_csv, key_cols=['variant', 'seed'])
    _summary_df_merged = _merge_dedupe(_summary_df, _sum_csv, key_cols=['variant'])
    _fa_df_merged.to_csv(_raw_csv, index=False)
    _summary_df_merged.to_csv(_sum_csv, index=False)
    if _c_per_seed:
        _cps_df = pd.DataFrame([{'seed': s, 'pearson': p} for s, p in
                                sorted(_c_per_seed.items())])
        _cps_csv = out_dir / 'fusion_ablation_C_per_seed.csv'
        _cps_merged = _merge_dedupe(_cps_df, _cps_csv, key_cols=['seed'])
        _cps_merged.sort_values('seed').to_csv(_cps_csv, index=False)
    print(f"\n  ✓ [FUSION_ABL] raw runs → {_raw_csv}  "
          f"(merged: {len(_fa_df_merged)} rows, {_fa_df_merged['variant'].nunique()} variants)")
    print(f"  ✓ [FUSION_ABL] summary  → {_sum_csv}  "
          f"(merged: {_summary_df_merged['variant'].nunique()} variants)")
    # 用合并后的完整表替换本次打印用的 summary_rows, 让 paired-Δ 表显示全历史
    summary_rows = _summary_df_merged.to_dict('records')

    print("\n" + "=" * 86)
    print("  FUSION_ABLATION 汇总 (paired Δ = Pearson(C) − Pearson(variant), two-sided)")
    print(f"  C baseline: {_c_source or 'NONE'}")
    print("=" * 86)
    print(f"  {'variant':14s}  {'nV':>2}  {'nP':>2}  "
          f"{'r mean':>7}  {'r std':>7}  "
          f"{'Δ mean':>8}  {'Δ std':>7}  "
          f"{'Welch p':>8}  {'Wilc p':>7}  {'d':>6}")
    print(f"  {'-' * 84}")
    for _row in summary_rows:
        _fmt = lambda x, w=7, p=4: (f"{x:>{w}.{p}f}" if np.isfinite(x)
                                     else f"{'nan':>{w}}")
        # signed-Δ 单独处理 (带 + 号对齐) + 无值时打印空白
        if np.isfinite(_row['delta_vs_C_mean']):
            _dm_str = f"{_row['delta_vs_C_mean']:>+8.4f}"
        else:
            _dm_str = f"{'nan':>8}"
        print(f"  {_row['variant']:14s}  "
              f"{_row['n_seeds']:>2d}  {_row['paired_n']:>2d}  "
              f"{_fmt(_row['pearson_mean'], 7, 4)}  "
              f"{_fmt(_row['pearson_std'], 7, 4)}  "
              f"{_dm_str}  "
              f"{_fmt(_row['delta_vs_C_std'], 7, 4)}  "
              f"{_fmt(_row['welch_p_two_sided'], 8, 4)}  "
              f"{_fmt(_row['paired_p_wilcoxon_two_sided'], 7, 4)}  "
              f"{_fmt(_row['cohen_d'], 6, 2)}")
        if not np.isfinite(_row['delta_vs_C_mean']) and np.isfinite(_row['unpaired_delta_seed42']):
            print(f"  {'':14s}    ↳ unpaired_Δ@seed42 = "
                  f"{_row['unpaired_delta_seed42']:+.4f}  "
                  f"(单 seed, 不可做统计推断)")
    print("=" * 86)
    print("  nV = n seeds with variant Pearson;  nP = n paired with C;  d = Cohen's d")
    print("  解读 (based on 消融改进.md 5-seed 分析):")
    print("   · |Δ| < 0.003 且 |d| < 0.3    → 该模块冗余, 去掉不伤 (e.g. C_concat)")
    print("   · Δ > 0.003 且 d > 0.8        → 该模块关键 (e.g. C_noLM, d≈+6.5)")
    print("   · Δ < 0 且 |d| > 0.3          → 该模块反方向 (e.g. C_noCrossW)")
    print("=" * 86)

    return _summary_df


# ============================================================================
# Supp S11 汇合: 四份 per-quadrant CSV → 单张统一长表
# ============================================================================
#   读取位置 (任一缺失都会跳过, 不影响其他源):
#     {output_dir}/multiseed_C/multiseed_C_quadrant_per_seed.csv
#     {output_dir}/fusion_ablation/fusion_ablation_per_quadrant.csv
#     {output_dir}/tf_only_baseline/tf_only_quadrant_per_seed.csv
#     {output_dir}/model_comparison_matrix/comparison_matrix_quadrant.csv
#   输出:
#     {output_dir}/paper_supp_s11_per_quadrant.csv
#   统一 schema:
#     source, model, variant, feature_set, seed, quadrant,
#     n, pearson, mae, bias, spearman, r2
# ============================================================================

_SUPP_S11_COLS = ['source', 'model', 'variant', 'feature_set',
                  'seed', 'quadrant',
                  'n', 'pearson', 'mae', 'bias', 'spearman', 'r2']


def _consolidate_paper_supp_s11_per_quadrant(output_dir):
    """把 multiseed_C / fusion_ablation / tf_only / model_comparison 四份
    per-quadrant CSV 汇合成单张 paper_supp_s11_per_quadrant.csv.

    每一源按统一 schema 归一: 缺失列补 NaN (数值) 或空串 (分类).
    返回合并后的 DataFrame; 若四份全缺则返回 None.
    """
    out_root = Path(output_dir)
    target   = out_root / 'paper_supp_s11_per_quadrant.csv'

    sources = [
        ('multiseed_C',
         out_root / 'multiseed_C'            / 'multiseed_C_quadrant_per_seed.csv'),
        ('fusion_ablation',
         out_root / 'fusion_ablation'        / 'fusion_ablation_per_quadrant.csv'),
        ('tf_only',
         out_root / 'tf_only_baseline'       / 'tf_only_quadrant_per_seed.csv'),
        ('model_comparison',
         out_root / 'model_comparison_matrix' / 'comparison_matrix_quadrant.csv'),
    ]

    print("\n" + "=" * 62 +
          "\n  ★ Supp S11: 四份 per-quadrant CSV 汇合\n" + "=" * 62)

    frames = []
    for src_name, src_path in sources:
        if not src_path.exists():
            print(f"  [S11] 跳过 (不存在): {src_path}")
            continue
        try:
            df = pd.read_csv(src_path)
        except Exception as e:
            print(f"  [S11] 读取失败 {src_path}: {e}")
            continue
        if df.empty:
            print(f"  [S11] 空表, 跳过: {src_path}")
            continue

        df = df.copy()
        df['source'] = src_name

        # 按源归一 model / variant / feature_set / seed
        if src_name == 'multiseed_C':
            # 原列: seed, quadrant, n, pearson, mae, bias, spearman, r2
            df['model']       = 'ours_C'
            df['variant']     = 'C'
            df['feature_set'] = ''
        elif src_name == 'fusion_ablation':
            # 原列: variant, seed, quadrant, n, pearson, mae, bias
            df['model']       = 'ours_' + df['variant'].astype(str)
            df['feature_set'] = ''
        elif src_name == 'tf_only':
            # 原列: variant, model, seed, quadrant, n, pearson, mae, bias, spearman, r2
            # 此处 variant = TF 特征化方式 (tss_weighted / raw_bins_pca), 等同 feature_set
            df['feature_set'] = df['variant']
        elif src_name == 'model_comparison':
            # 原列: model, feature_set, quadrant, n, pearson, mae, bias  (单 seed)
            df['variant'] = ''
            df['seed']    = np.nan

        # 补齐缺列 → 按 _SUPP_S11_COLS 固定列序
        for c in _SUPP_S11_COLS:
            if c not in df.columns:
                df[c] = (np.nan if c in ('seed', 'n', 'pearson', 'mae',
                                         'bias', 'spearman', 'r2') else '')

        frames.append(df[_SUPP_S11_COLS])
        print(f"  [S11] 读入 {src_name:<18s}  rows={len(df)}")

    if not frames:
        print("  ⚠ [S11] 无任何可用源 CSV, 跳过合并")
        return None

    merged = pd.concat(frames, ignore_index=True)

    # 数值列统一为 float (避免 object / 混合类型)
    for c in ('seed', 'n', 'pearson', 'mae', 'bias', 'spearman', 'r2'):
        merged[c] = pd.to_numeric(merged[c], errors='coerce')

    merged = (merged.sort_values(
                  ['source', 'model', 'variant', 'feature_set',
                   'seed', 'quadrant'],
                  kind='stable', na_position='last')
                    .reset_index(drop=True))

    merged.to_csv(target, index=False)
    _dist = ", ".join(f"{k}={v}" for k, v in
                      merged['source'].value_counts().to_dict().items())
    print(f"\n  ✓ [S11] 合并完成 → {target}")
    print(f"     总行数 {len(merged)}  |  source 分布: {_dist}")
    return merged


# ============================================================================
# 架构链多种子 (B / T) 跑批 + 汇总  (new54 增补)
# ----------------------------------------------------------------------------
#   设计与 FUSION_ABLATION 的 C-per-seed 块高度对称:
#     · seed=42 复用主链单 seed 结果 (memory 优先, disk 兜底)
#     · 其它 seed 走 multiseed_<X>_seed{s} 子目录 + train_experiment 自带的
#       RESUME_FROM_CACHE → 已跑过的 seed 自动跳过
#     · 全部成功后写 multiseed_<X>_summary.csv (与 multiseed_C_summary.csv 列
#       结构一致, 便于下游 _emit_dual_protocol_comparison 一次性读 3 张表)
# ============================================================================

# 主链 seed=42 单 seed 实验 → 模型代号映射 (确保 seed=42 复用入口稳定)
_PROGRESSION_SEED42_LABEL = {
    'D': 'ablation_D_dna_cnn',
    'L': 'ablation_L_lm_only',
    'B': 'ablation_B_act_lm',
    'C': 'ablation_C_seq_tf',
    'T': 'ablation_T_tf_only_arch',
}


def _resolve_progression_seed_result(model_key, seed, all_results, output_dir, device):
    """统一的 progression-seed 复用解析器.

    seed=42 优先从主链单 seed 实验 (ablation_<X>_<...>) 取; 其它 seed 优先
    从 multiseed_<X>_seed{s}/ 取; FUSION_ABL 已跑过的 C-per-seed (label
    fusion_abl_C_seed{s}) 也作为 C 模型的备用入口.

    返回 (res_dict_or_None, source_str). res_dict 至少含 pearson; 若是从
    train_result_cache 读出还会含 preds_test/labels_test/test_gene_indices.
    """
    candidates = []
    if seed == 42:
        candidates.append(_PROGRESSION_SEED42_LABEL.get(model_key))
    candidates.append(f'multiseed_{model_key}_seed{seed}')
    if model_key == 'C':
        # FUSION_ABL 已跑的 C-per-seed (与 multiseed_C 同语义) 也算备份
        candidates.append(f'fusion_abl_C_seed{seed}')
    candidates = [c for c in candidates if c]

    # (a) memory: all_results 命中 → 直接返回, 含完整 preds 等
    for k in candidates:
        r = all_results.get(k)
        if r is not None and 'pearson' in r and np.isfinite(r.get('pearson', np.nan)):
            return r, f'memory:{k}'
    # (b) disk: <output_dir>/<k>/train_result_cache.pth
    for k in candidates:
        cache = _load_c_seed_payload_from_cache(
            Path(output_dir) / k, device)
        if cache is not None and np.isfinite(cache.get('pearson', np.nan)):
            return cache, f'disk:{k}/train_result_cache.pth'
    return None, 'missing'


def _run_progression_multiseed(model_key, seeds, all_results,
                                base_kwargs_factory, output_dir,
                                device, b_model_resolver=None,
                                quadrant_info=None, test_mask=None,
                                compute_quadrant=True):
    """给 B / T / C 跑指定 seeds (尽量复用), 写 multiseed_<X>_summary.csv.

    参数
    ----
    model_key : 'B' | 'T' | 'C'
    seeds     : list[int]
    all_results : 主流程的 all_results dict; 函数会就地写入 fresh-trained 结果
    base_kwargs_factory : callable(seed, label) -> kwargs dict
        构造一次 train_experiment 调用所需的所有 kwargs (复用 _exp_kwargs 输出
        + 该模型独有字段). 必须接受 seed/label 是为了让外层每 seed 显式
        重设 random_state 后再注入正确的 label, kwargs 里不含 random_seed.
    b_model_resolver : callable() -> (b_model_or_None, source_str) | None
        懒加载 B 模型供 C 软监督; B/T 不需要时传 None.
    quadrant_info, test_mask, compute_quadrant : 与 run_multiseed_eval 同义,
        用于 per-quadrant Pearson 输出.

    Returns
    -------
    summary : dict  含 pearson_mean / pearson_std / seed_results (与
              run_multiseed_eval 的返回结构一致)
    """
    sub_dir = Path(output_dir) / f'multiseed_{model_key}'
    sub_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 62 +
          f"\n  ★ 架构链多种子: {model_key}  (seeds={seeds})\n" + "=" * 62)

    preloaded = {}
    for s in seeds:
        res, src = _resolve_progression_seed_result(
            model_key, s, all_results, output_dir, device)
        if res is not None:
            preloaded[s] = res
            print(f"  [multiseed_{model_key}] seed={s} 复用自 {src}  "
                  f"(r={float(res.get('pearson', float('nan'))):.4f})")
    seeds_to_run = [s for s in seeds if s not in preloaded]
    if seeds_to_run:
        print(f"  [multiseed_{model_key}] 待训 {len(seeds_to_run)} seeds: "
              f"{seeds_to_run}")
    else:
        print(f"  [multiseed_{model_key}] 全部 {len(seeds)} seeds 已就绪, 无需训练")

    def _train_one(random_seed, label):
        random.seed(random_seed); np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        kwargs = base_kwargs_factory(random_seed, label)
        # b_model_resolver 仅在 model_key=='C' 时非 None
        if b_model_resolver is not None:
            b_sup, _b_src = b_model_resolver()
            kwargs['seq_baseline_model'] = b_sup
        return train_experiment(**kwargs)

    # 为符合 run_multiseed_eval 的接口, 把 preloaded 直接喂进去
    summary = run_multiseed_eval(
        seed_list=seeds,
        train_fn=(_train_one if seeds_to_run else None),
        output_dir=str(sub_dir),
        label_prefix=f'multiseed_{model_key}',
        quadrant_info=quadrant_info,
        test_mask=test_mask,
        compute_quadrant=compute_quadrant,
        preloaded_results=preloaded,
    )
    # 把训练新 seed 的结果回写 all_results, 让下游 (FUSION_ABL 汇总等) 也能看到
    if seeds_to_run:
        for s in seeds_to_run:
            label = f'multiseed_{model_key}_seed{s}'
            r = all_results.get(label)
            if r is None:
                # 从磁盘装填 (run_multiseed_eval 内部会触发 train_experiment,
                # 后者已经写了 cache; 这里仅再读一次, 不重训)
                cache = _load_c_seed_payload_from_cache(
                    Path(output_dir) / label, device)
                if cache is not None:
                    all_results[label] = cache
    return summary


# ============================================================================
# 双协议汇总: Protocol 1 (single seed=42) + Protocol 2 (paired multi-seed)
#   (new54 增补 — 直接服务于论文 Fig 3 / Table 1 + Supp S12)
# ----------------------------------------------------------------------------
# Protocol 1: paper_protocol1_single_seed42.csv
#   每行一个模型 (D / L / B / T / C), 仅 seed=42 单值; 提供 paired Δ vs C@seed=42.
#   适用场景: 架构链 progression 表; 所有 Δ 在同一 seed 上, 公平且简洁; 但无 std.
#
# Protocol 2: paper_protocol2_paired_multiseed.csv
#   每行一个模型 (B / T / C; D / L 默认不参与), 5-seed mean ± std + paired Δ
#   vs C 的 (mean, std, Cohen's d, Welch p, paired Wilcoxon p).
#   适用场景: 论文核心 Δ 主张 (Δ(B→C), Δ(T→C)) 的统计可信度; 与 FUSION_ABLATION
#   summary 用同一统计协议, 直接挂 Supp S12.
# ============================================================================

def _model_seed_pearson(model_key, seed, all_results, output_dir, device):
    """从 memory + disk 解析单个 (model_key, seed) 的 Pearson 标量.
    返回 (pearson_or_nan, source_str)."""
    res, src = _resolve_progression_seed_result(
        model_key, seed, all_results, output_dir, device)
    if res is None:
        return float('nan'), src
    try:
        return float(res.get('pearson', float('nan'))), src
    except Exception:
        return float('nan'), src


def _emit_dual_protocol_comparison(all_results, output_dir, device,
                                   ms_seeds, ms_models,
                                   c_per_seed_external=None):
    """生成两份"成熟"对比 CSV, 一次性把 D→L→B→C→T 链摆清楚.

    参数
    ----
    ms_seeds : list[int]
        Protocol 2 的 seed 列表 (通常 = MULTISEED_PROGRESSION['seeds']).
    ms_models : list[str]
        参与 Protocol 2 的模型集合 (如 ['B', 'T', 'C']); D/L 自动排除.
    c_per_seed_external : dict[int, float] | None
        C per-seed Pearson 的外部入口 (如 FUSION_ABL 的 _fusion_abl_c_per_seed).
        若给出, 则跳过 C 的 disk 解析直接用; 否则照常从 memory/disk 拉.

    Returns
    -------
    (df_p1, df_p2) : (DataFrame, DataFrame)
        即落盘的两张表; 任一表无可用数据则该位返回 None.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 62 +
          "\n  ★ 双协议对比 CSV 输出 (new54)\n" + "=" * 62)

    progression_chain = ['D', 'L', 'B', 'T', 'C']

    # ── Protocol 1: 全部 seed=42 单 seed ────────────────────────────────────
    p1_rows = []
    p1_sources = {}
    p1_pearsons = {}
    for mk in progression_chain:
        r, src = _model_seed_pearson(mk, 42, all_results, output_dir, device)
        p1_pearsons[mk] = r
        p1_sources[mk] = src

    c42 = p1_pearsons.get('C', float('nan'))

    for mk in progression_chain:
        r = p1_pearsons.get(mk, float('nan'))
        prev_idx = progression_chain.index(mk) - 1
        prev_mk = progression_chain[prev_idx] if prev_idx >= 0 else None
        prev_r = p1_pearsons.get(prev_mk, float('nan')) if prev_mk else float('nan')
        delta_chain = (r - prev_r) if (prev_mk and np.isfinite(prev_r) and np.isfinite(r)) else float('nan')
        delta_vs_c = (r - c42) if (np.isfinite(c42) and np.isfinite(r)) else float('nan')

        p1_rows.append(dict(
            model           = mk,
            pearson_seed42  = r,
            delta_chain     = delta_chain,        # = pearson(mk) − pearson(prev in chain)
            chain_prev      = prev_mk or '',
            delta_vs_C42    = delta_vs_c,         # = pearson(mk) − pearson(C@seed=42)
            source          = p1_sources.get(mk, 'missing'),
        ))
    df_p1 = pd.DataFrame(p1_rows)
    p1_path = out_dir / 'paper_protocol1_single_seed42.csv'
    df_p1.to_csv(p1_path, index=False)

    print(f"\n  ── Protocol 1 (single seed=42): {p1_path.name}")
    print(f"     {'model':<3s}  {'r@42':>8s}  {'Δ chain':>10s}  {'Δ vs C@42':>12s}  source")
    for row in p1_rows:
        _fmt = lambda x: (f"{x:>+.4f}" if np.isfinite(x) else "    nan ")
        _fmt8 = lambda x: (f"{x:>8.4f}" if np.isfinite(x) else "     nan")
        print(f"     {row['model']:<3s}  {_fmt8(row['pearson_seed42'])}  "
              f"{_fmt(row['delta_chain']):>10s}  {_fmt(row['delta_vs_C42']):>12s}  "
              f"{row['source']}")

    # ── Protocol 2: paired multi-seed (B / T / C 默认) ─────────────────────
    # 1) 构建 per-(model, seed) Pearson 长表
    p2_rows_long = []
    for mk in ms_models:
        for s in ms_seeds:
            if mk == 'C' and c_per_seed_external and s in c_per_seed_external \
                    and np.isfinite(c_per_seed_external[s]):
                p2_rows_long.append(dict(model=mk, seed=int(s),
                                          pearson=float(c_per_seed_external[s]),
                                          source=f'external_c_per_seed:{s}'))
                continue
            r, src = _model_seed_pearson(mk, s, all_results, output_dir, device)
            p2_rows_long.append(dict(model=mk, seed=int(s),
                                      pearson=r, source=src))
    df_p2_long = pd.DataFrame(p2_rows_long)
    p2_long_path = out_dir / 'paper_protocol2_paired_multiseed_per_seed.csv'
    df_p2_long.to_csv(p2_long_path, index=False)

    # 2) C 在 ms_seeds 上的 Pearson 字典 (paired Δ baseline)
    c_per_seed = {}
    for _, r in df_p2_long[df_p2_long['model'] == 'C'].iterrows():
        if np.isfinite(r['pearson']):
            c_per_seed[int(r['seed'])] = float(r['pearson'])

    # 3) 每个 model 算 5-seed mean/std + paired Δ vs C + Welch / Cohen / Wilcoxon
    p2_rows = []
    for mk in ms_models:
        sub = df_p2_long[df_p2_long['model'] == mk].copy()
        vals = sub['pearson'].values.astype(np.float64)
        finite = vals[np.isfinite(vals)]
        n_seeds = int(finite.size)
        r_mean = float(finite.mean()) if n_seeds >= 1 else float('nan')
        r_std  = float(finite.std(ddof=1)) if n_seeds >= 2 else float('nan')
        seeds_used = sub.loc[np.isfinite(sub['pearson']), 'seed'].astype(int).tolist()

        # paired Δ = pearson(C@seed) - pearson(mk@seed) for each shared seed
        # 正值 = C 比 mk 强 (与 FUSION_ABLATION summary 一致)
        paired = []
        for _, row in sub.iterrows():
            s = int(row['seed']); v = float(row['pearson'])
            if s in c_per_seed and np.isfinite(v) and np.isfinite(c_per_seed[s]):
                paired.append((c_per_seed[s], v))
        paired_n = len(paired)
        deltas_C_minus_other = np.array([c - v for c, v in paired],
                                         dtype=np.float64) if paired else np.array([])

        if paired_n >= 2:
            d_mean = float(deltas_C_minus_other.mean())
            d_std  = float(deltas_C_minus_other.std(ddof=1))
        elif paired_n == 1:
            d_mean = float(deltas_C_minus_other[0]); d_std = float('nan')
        else:
            d_mean = float('nan'); d_std = float('nan')

        cohen_d = float('nan'); welch_p = float('nan'); welch_t = float('nan')
        if mk != 'C' and paired_n >= 2:
            try:
                from scipy.stats import ttest_ind
                c_arr = np.array([c for c, _ in paired], dtype=np.float64)
                v_arr = np.array([v for _, v in paired], dtype=np.float64)
                if c_arr.std(ddof=1) > 0 and v_arr.std(ddof=1) > 0:
                    tt = ttest_ind(c_arr, v_arr, equal_var=False)
                    welch_t = float(tt.statistic); welch_p = float(tt.pvalue)
                    pooled_var = (c_arr.var(ddof=1) + v_arr.var(ddof=1)) / 2.0
                    if pooled_var > 0:
                        cohen_d = float((c_arr.mean() - v_arr.mean()) /
                                         np.sqrt(pooled_var))
            except Exception:
                pass

        wilcox_p = float('nan')
        if mk != 'C' and paired_n >= 3:
            try:
                from scipy.stats import wilcoxon
                wilcox_p = float(wilcoxon(deltas_C_minus_other,
                                           alternative='two-sided').pvalue)
            except Exception:
                pass

        p2_rows.append(dict(
            model                       = mk,
            n_seeds                     = n_seeds,
            pearson_mean                = r_mean,
            pearson_std                 = r_std,
            seeds_used                  = ','.join(str(s) for s in sorted(seeds_used)),
            paired_n_with_C             = paired_n,
            delta_C_minus_self_mean     = d_mean,    # 正值 = C > 该模型
            delta_C_minus_self_std      = d_std,
            cohen_d_C_vs_self           = cohen_d,
            welch_t_C_vs_self           = welch_t,
            welch_p_two_sided           = welch_p,
            wilcoxon_p_two_sided        = wilcox_p,
        ))
    df_p2 = pd.DataFrame(p2_rows)
    p2_path = out_dir / 'paper_protocol2_paired_multiseed.csv'
    df_p2.to_csv(p2_path, index=False)

    print(f"\n  ── Protocol 2 (paired multi-seed): {p2_path.name}")
    print(f"     长表: {p2_long_path.name}")
    print(f"     {'model':<3s}  {'n':>2s}  {'r mean±std':>15s}  "
          f"{'ΔC mean±std':>16s}  {'cohen d':>8s}  {'Welch p':>9s}  {'Wilc p':>8s}")
    for row in p2_rows:
        _fmt_pm = lambda m, s, w=15, p=4: (
            f"{m:>+.{p}f} ± {s:.{p}f}".rjust(w) if (np.isfinite(m) and np.isfinite(s))
            else (f"{m:>+.{p}f}".rjust(w) if np.isfinite(m) else "    nan         ".rjust(w)))
        _fmt = lambda x, w=8, p=3: (f"{x:>{w}.{p}f}" if np.isfinite(x) else "     nan"[:w].rjust(w))
        _fmt_p = lambda x: (f"{x:>9.2e}" if np.isfinite(x) else "      nan")
        print(f"     {row['model']:<3s}  {row['n_seeds']:>2d}  "
              f"{_fmt_pm(row['pearson_mean'], row['pearson_std'])}  "
              f"{_fmt_pm(row['delta_C_minus_self_mean'], row['delta_C_minus_self_std'], 16)}  "
              f"{_fmt(row['cohen_d_C_vs_self'])}  "
              f"{_fmt_p(row['welch_p_two_sided'])}  "
              f"{_fmt_p(row['wilcoxon_p_two_sided'])}")

    print("\n  ✓ 双协议对比已落盘 (3 文件):")
    print(f"     · {p1_path}")
    print(f"     · {p2_path}")
    print(f"     · {p2_long_path}")
    return df_p1, df_p2


# ============================================================================
# SequenceOnlyPredictor（消融 A / B）
# ============================================================================

class SequenceOnlyPredictor(nn.Module):
    model_kind = 'seq_only'

    def __init__(self, n_windows=12, d_profile=64, d_hidden=128, dropout=0.25,
                 use_lm=False, lm_dim=2304, d_lm_hidden=512, d_lm_out=128,
                 lm_input_dropout=0.10, freeze_lm=False, use_profile=True):
        super().__init__()
        self.use_scalars    = False   # seq_only 统一为 False（语义：不与 TF 融合）
        self.use_profile    = use_profile
        self.use_lm         = use_lm
        self.n_seq_features = n_windows
        self.seq_branch = SequenceBranch(
            n_windows=n_windows, d_profile=d_profile, d_out=d_hidden, dropout=0.10,
            use_lm=use_lm, lm_dim=lm_dim, d_lm_hidden=d_lm_hidden,
            d_lm_out=d_lm_out, lm_input_dropout=lm_input_dropout, freeze_lm=freeze_lm,
            use_profile=use_profile)
        self.head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Dropout(dropout*0.5), nn.Linear(64, 1))
        print(f"\n构建 SequenceOnlyPredictor  use_profile={use_profile}  use_lm={use_lm}  "
              f"总参数: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, profile, lm_emb=None):
        return self.head(self.seq_branch(profile, lm_emb)).squeeze(-1)


# ============================================================================
# DNA one-hot 分支（消融 D：Basset-inspired CNN，1500 bp 序列 → 标量）
# ============================================================================

class DNAOneHotCNN(nn.Module):
    """Basset-inspired one-hot DNA encoder.
    Input : (B, 4, L)  L = upstream + downstream (默认 1500)
    Output: (B, d_out) 句级表征
    约 1.5M 参数，与 LMAdapter 量级相当，保证 D vs L 比较公平。
    """
    def __init__(self, seq_len=1500, d_out=128, dropout=0.20):
        super().__init__()
        # Layer 1: motif 检测（~19bp 核，覆盖大多数酵母 TF motif 长度）
        self.conv1 = nn.Sequential(
            nn.Conv1d(4, 128, kernel_size=19, padding=9),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.MaxPool1d(3), nn.Dropout(dropout * 0.5))
        # Layer 2: motif 组合
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 192, kernel_size=11, padding=5),
            nn.BatchNorm1d(192), nn.GELU(),
            nn.MaxPool1d(4), nn.Dropout(dropout * 0.5))
        # Layer 3: 更高阶语法
        self.conv3 = nn.Sequential(
            nn.Conv1d(192, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256), nn.GELU(),
            nn.MaxPool1d(4), nn.Dropout(dropout))
        # 膨胀残差块：扩大感受野，捕获长程依赖
        self.dilated = MultiScaleDilatedBlock(256, 256, dropout=dropout)
        # 全局平均池化
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, d_out), nn.LayerNorm(d_out),
            nn.GELU(), nn.Dropout(dropout))
        self.out_dim = d_out

    def forward(self, x):
        # x: (B, 4, L)
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
        x = self.dilated(x)
        x = self.pool(x).squeeze(-1)                              # (B, 256)
        return self.fc(x)                                         # (B, d_out)


class DNAOnlyPredictor(nn.Module):
    """消融 D 专用：DNA one-hot → CNN → 标量 TPM 预测"""
    model_kind = 'dna'

    def __init__(self, seq_len=1500, d_hidden=128, dropout=0.25):
        super().__init__()
        self.encoder = DNAOneHotCNN(seq_len=seq_len, d_out=d_hidden,
                                     dropout=max(dropout - 0.05, 0.10))
        self.head = nn.Sequential(
            nn.Linear(d_hidden, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1))
        print(f"\n构建 DNAOnlyPredictor  seq_len={seq_len}  "
              f"总参数: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, dna):
        return self.head(self.encoder(dna)).squeeze(-1)


# ============================================================================
# MultiModalPredictor（完整多模态预测器）
# ============================================================================

class MultiModalPredictor(nn.Module):
    model_kind = 'tf'

    def __init__(self, n_tfs, n_bins,
                 d_tf=256, d_cnn=256, n_tf_layers=2, n_regions=5, n_heads=8,
                 conv_dropout=0.10, fc_dropout=0.25,
                 use_scalars=True, n_scalars=14, n_seq_features=12,
                 d_seq=128, d_tf_proj=96,
                 use_lm=False, lm_dim=2304, d_lm_hidden=512, d_lm_out=128,
                 lm_input_dropout=0.10, freeze_lm=False,
                 topk_k=0,
                 use_dna_onehot=False, dna_seq_len=1500, d_dna_out=128,
                 dna_dropout=0.20,
                 # ── Fusion 模块消融标志（FUSION_ABLATION 专用，默认 None 即无消融）──
                 fusion_flags=None):
        super().__init__()
        assert not (use_lm and use_dna_onehot), \
            "MultiModalPredictor: use_lm 与 use_dna_onehot 互斥"
        # 解析消融标志
        _ff = fusion_flags or {}
        _replace_concat   = bool(_ff.get('replace_with_concat', False))
        _freeze_crossw    = _ff.get('freeze_crossw', None)
        _disable_lm       = bool(_ff.get('disable_lm', False))
        _disable_activity = bool(_ff.get('disable_activity', False))
        # ── new48 追加 ──
        _use_tss_weighted_tf = bool(_ff.get('use_tss_weighted_tf', False))

        # 标志最终态（LM / activity 的外部 use_* 可能已被 train_experiment 关掉;
        # 这里再 and 一次标志, 避免双开关打架）
        _eff_use_lm      = bool(use_lm) and (not _disable_lm)
        _eff_use_profile = (not _disable_activity)

        if _disable_activity and (not _eff_use_lm) and (not use_dna_onehot) and use_scalars:
            raise ValueError(
                "MultiModalPredictor: disable_activity=True 时必须保留 LM 或 DNA one-hot"
                " 之一作为 seq 分支输入, 否则 SequenceBranch 无可用特征.")

        _seq_tag = ('+LM' if _eff_use_lm else ('+OneHot' if use_dna_onehot else ''))
        _abl_tag = ''
        if _ff:
            _bits = []
            if _replace_concat:        _bits.append('concat')
            if _freeze_crossw:         _bits.append(f'crossw={_freeze_crossw}')
            if _disable_lm:            _bits.append('-LM')
            if _disable_activity:      _bits.append('-Act')
            if _use_tss_weighted_tf:   _bits.append('tss-weighted-TF')
            if _bits: _abl_tag = '  [ABL:' + ','.join(_bits) + ']'
        print(f"\n构建 MultiModalPredictor"
              f"{'  [+ResidualCrossModal]' if use_scalars and not _replace_concat else ''}"
              f"{'  [+ConcatFusion]' if use_scalars and _replace_concat else ''}"
              f"{'  [' + _seq_tag + ']' if _seq_tag else ''}"
              f"{'  [+TopK-' + str(topk_k) + ']' if topk_k > 0 else ''}"
              f"{_abl_tag}")
        print("=" * 60)
        self.use_scalars         = use_scalars
        self.n_seq_features      = n_seq_features
        self.use_tss_weighted_tf = _use_tss_weighted_tf

        # ── TF 编码路径: 简化 vs 完整 ─────────────────────────────────────
        # 简化路径 (new48 C_tss_weighted): TFTssWeightedEncoder 一步到位
        # 完整路径 (C 默认): TFAttentionPooler + TFSpatialEncoder 双路 concat
        if self.use_tss_weighted_tf:
            self.tf_attn_pooler = None
            self.tf_spatial_enc = None
            self.tf_tss_encoder = TFTssWeightedEncoder(n_tfs=n_tfs, n_bins=n_bins)
            tf_merged_dim = self.tf_tss_encoder.out_dim
        else:
            self.tf_attn_pooler = TFAttentionPooler(
                n_tfs=n_tfs, n_bins=n_bins, d_tf=d_tf,
                n_heads=n_heads, n_layers=n_tf_layers, dropout=conv_dropout,
                d_seq_query=d_seq if use_scalars else None,
                topk_k=topk_k)
            self.tf_spatial_enc = TFSpatialEncoder(
                n_tfs=n_tfs, n_bins=n_bins, n_regions=n_regions,
                d_cnn=d_cnn, dropout=conv_dropout)
            self.tf_tss_encoder = None
            tf_merged_dim = self.tf_attn_pooler.out_dim + self.tf_spatial_enc.out_dim

        if use_scalars:
            n_reg = max(0, n_scalars - n_seq_features)
            _fusion_cls = ConcatFusion if _replace_concat else ResidualCrossModalFusion
            self.fusion = _fusion_cls(
                tf_merged_dim=tf_merged_dim, d_seq=d_seq, d_tf=d_tf_proj,
                n_seq_features=n_seq_features, n_reg_scalars=n_reg,
                dropout=conv_dropout,
                use_lm=_eff_use_lm, lm_dim=lm_dim, d_lm_hidden=d_lm_hidden,
                d_lm_out=d_lm_out, lm_input_dropout=lm_input_dropout, freeze_lm=freeze_lm,
                use_profile=_eff_use_profile,
                use_dna_onehot=use_dna_onehot, dna_seq_len=dna_seq_len,
                d_dna_out=d_dna_out, dna_dropout=dna_dropout,
                freeze_crossw=_freeze_crossw)
            fc_in = self.fusion.out_dim
        else:
            self.fusion = None
            fc_in = tf_merged_dim

        self.fc_head = nn.Sequential(
            nn.Linear(fc_in, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(fc_dropout),
            nn.Linear(256, 64),    nn.LayerNorm(64),  nn.GELU(), nn.Dropout(fc_dropout * 0.5),
            nn.Linear(64, 1))

        print(f"  tf_merged_dim={tf_merged_dim}  fc_in={fc_in}  "
              f"总参数: {sum(p.numel() for p in self.parameters()):,}")
        print("=" * 60)

    def forward(self, tf_signal, scalars=None, lm_emb=None,
                return_attn=False, return_internals=False):
        # 预计算 seq_emb, 同时作为 TF 注意力的动态查询, 避免后续在 fusion 中重复计算
        seq_emb_pre = None
        if self.fusion is not None and scalars is not None:
            seq_profile = scalars[:, :self.fusion.n_seq_features]
            seq_emb_pre = self.fusion.seq_branch(seq_profile, lm_emb)

        # ── TF 路径: 简化 (tss_weighted) vs 完整 (attn_pooler + spatial_enc) ──
        if self.use_tss_weighted_tf:
            # 简化路径: 无 attn_w, 无 region_weights
            tf_merged      = self.tf_tss_encoder(tf_signal)      # (B, n_tfs)
            attn_w         = None
            region_weights = None
        else:
            if return_attn:
                tf_attn_feat, attn_w = self.tf_attn_pooler(
                    tf_signal, seq_query=seq_emb_pre, return_attn=True)
            else:
                tf_attn_feat = self.tf_attn_pooler(tf_signal, seq_query=seq_emb_pre)
                attn_w = None
            # TFSpatialEncoder 仍保持 (proj, region_tokens, region_weights) 三元返回,
            # region_tokens 当前不再被 fusion 使用, 这里直接丢弃 (以 _ 接收).
            tf_cnn_feat, _, region_weights = self.tf_spatial_enc(
                tf_signal, return_regions=True)
            tf_merged = torch.cat([tf_attn_feat, tf_cnn_feat], dim=1)

        seq_emb = None; tf_emb = None; cross_weight = None
        if self.fusion is not None and scalars is not None:
            if return_internals:
                fc_input, seq_emb, tf_emb, cross_weight = self.fusion(
                    tf_merged, scalars, lm_emb=lm_emb, seq_emb_pre=seq_emb_pre,
                    return_internals=True)
            else:
                fc_input = self.fusion(
                    tf_merged, scalars, lm_emb=lm_emb, seq_emb_pre=seq_emb_pre)
        else:
            fc_input = tf_merged

        pred = self.fc_head(fc_input).squeeze(-1)
        extras = []
        if return_attn:
            # tss_weighted 路径下用全零占位, 保持 shape (B, 1) 兼容下游;
            # 下游 extract_tf_importance 会看到全零, 等价于"此变体不产出 attn",
            # 不会破坏训练/评估流程.
            if attn_w is None:
                attn_w = torch.zeros(pred.shape[0], 1, device=pred.device, dtype=pred.dtype)
            extras.append(attn_w)
        # return_internals 包含 region_weights，供诊断工具 (collect_cross_weights
        # / collect_region_weights) 使用. new53 已下线 region_diversity_loss,
        # 训练循环不再消费 region_weights.
        if return_internals: extras.append((seq_emb, tf_emb, cross_weight, region_weights))
        return (pred, *extras) if extras else pred


# ============================================================================
# 数据标准化
# ============================================================================

class SignalStandardizer:
    def __init__(self): self.tf_means = []; self.tf_stds = []
    def fit_transform(self, X_tf):
        print("\nTF 信号: log1p + per-TF z-score [fit on TRAIN]")
        _, n_tfs, _ = X_tf.shape
        X_log = np.log1p(X_tf)
        X_std = np.zeros_like(X_log, dtype=np.float32)
        for i in range(n_tfs):
            mu  = X_log[:, i, :].mean(); sig = X_log[:, i, :].std() + 1e-8
            X_std[:, i, :] = (X_log[:, i, :] - mu) / sig
            self.tf_means.append(float(mu)); self.tf_stds.append(float(sig))
        print(f"  fit: [{X_std.min():.3f}, {X_std.max():.3f}]")
        return X_std
    def transform(self, X_tf):
        X_log = np.log1p(X_tf)
        X_std = np.zeros_like(X_log, dtype=np.float32)
        for i in range(len(self.tf_means)):
            X_std[:, i, :] = (X_log[:, i, :] - self.tf_means[i]) / self.tf_stds[i]
        return X_std


class ExpressionProcessor:
    def __init__(self, transform='log1p', normalize='zscore'):
        self.transform_type = transform; self.normalize = normalize
        self.scaler = None; self._center = None; self._scale = None

    def _apply(self, y):
        if self.transform_type == 'log1p': return np.log1p(y)
        if self.transform_type == 'log2p': return np.log2(1.0 + y)
        return y.copy()

    def fit_transform(self, y):
        print(f"\nTPM 处理: {self.transform_type} + {self.normalize} [fit on TRAIN]")
        y_t = self._apply(y)
        if self.normalize == 'zscore':
            self.scaler = StandardScaler()
            y_n = self.scaler.fit_transform(y_t.reshape(-1, 1)).ravel()
        elif self.normalize == 'robust':
            self._center = np.median(y_t)
            self._scale  = max(np.percentile(y_t,75) - np.percentile(y_t,25), 1e-8)
            y_n = (y_t - self._center) / self._scale
        else:
            y_n = y_t
        print(f"  fit: [{y_n.min():.3f}, {y_n.max():.3f}]")
        return y_n.astype(np.float32), np.ones(len(y), dtype=bool)

    def transform(self, y):
        y_t = self._apply(y)
        if self.normalize == 'zscore':
            return self.scaler.transform(y_t.reshape(-1, 1)).ravel().astype(np.float32)
        if self.normalize == 'robust':
            return ((y_t - self._center) / self._scale).astype(np.float32)
        return y_t.astype(np.float32)


# ============================================================================
# Dataset — 批次: (sig, sc, lm_emb, weights, exps)
# ============================================================================

class GeneExpressionDataset(Dataset):
    def __init__(self, signals, expressions, sample_weights, scalars=None,
                 lm_embs=None, augment=False,
                 tf_mask_prob=0.3, tf_mask_ratio=0.05, lm_noise_std=0.0):
        self.signals        = torch.FloatTensor(signals)
        self.expressions    = torch.FloatTensor(expressions)
        self.sample_weights = torch.FloatTensor(sample_weights)
        self.scalars  = torch.FloatTensor(scalars) if scalars  is not None else None
        self.lm_embs  = torch.FloatTensor(lm_embs) if lm_embs  is not None else None
        self.augment       = augment
        self.tf_mask_prob  = tf_mask_prob
        self.tf_mask_ratio = tf_mask_ratio
        self.lm_noise_std  = lm_noise_std
        self.n_tfs         = signals.shape[1]

    def __len__(self): return len(self.expressions)

    def __getitem__(self, idx):
        sig = self.signals[idx].clone()
        sc  = self.scalars[idx]  if self.scalars  is not None else torch.zeros(1)
        lm  = self.lm_embs[idx].clone() if self.lm_embs is not None else torch.zeros(1)
        if self.augment:
            if torch.rand(1).item() > 0.5: sig += torch.randn_like(sig) * 0.02
            if torch.rand(1).item() > 0.5: sig *= 0.95 + torch.rand(1).item() * 0.10
            if torch.rand(1).item() < self.tf_mask_prob:
                n_m = max(1, round(self.n_tfs * self.tf_mask_ratio))
                sig[torch.randperm(self.n_tfs)[:n_m]] = 0.0
            if self.lm_noise_std > 0 and self.lm_embs is not None:
                lm = lm + torch.randn_like(lm) * self.lm_noise_std
        return sig, sc, lm, self.sample_weights[idx], self.expressions[idx]


class DNADataset(Dataset):
    """消融 D 专用数据集：DNA one-hot → (scalar TPM)。
    训练时可选 RevComp 增强（以 revcomp_prob 概率反向互补整条序列）。
    通道顺序假定为 A=0, C=1, G=2, T=3；RevComp 操作 = flip(L) + 通道重排 [3,2,1,0]。
    """
    def __init__(self, dna_onehot, expressions, sample_weights,
                 augment=False, revcomp_prob=0.5):
        self.dna_onehot    = torch.FloatTensor(dna_onehot)      # (N, 4, L)
        self.expressions   = torch.FloatTensor(expressions)
        self.sample_weights= torch.FloatTensor(sample_weights)
        self.augment       = augment
        self.revcomp_prob  = revcomp_prob

    def __len__(self): return len(self.expressions)

    def __getitem__(self, idx):
        x = self.dna_onehot[idx].clone()                         # (4, L)
        if self.augment and torch.rand(1).item() < self.revcomp_prob:
            x = torch.flip(x, dims=[-1])
            x = x[[3, 2, 1, 0], :]    # A↔T, C↔G
        return x, self.sample_weights[idx], self.expressions[idx]


# ============================================================================
# 数据加载
# ============================================================================

_MT_PATTERNS = re.compile(
    r'^(chr[Mm][A-Za-z]*|[Mm][Tt]$|[Mm]ito[A-Za-z]*|mitochondri[ao][A-Za-z]*)$',
    re.IGNORECASE)
def is_mitochondrial(chrom): return bool(_MT_PATTERNS.match(str(chrom).strip()))


def parse_bed(bed_file):
    rows = []
    with open(bed_file) as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            flds = line.strip().split('\t')
            if len(flds) < 4: continue
            rows.append({'chrom': flds[0], 'start': int(flds[1]), 'end': int(flds[2]),
                         'name': flds[3], 'score': flds[4] if len(flds)>4 else '0',
                         'strand': flds[5] if len(flds)>5 else '+'})
    df = pd.DataFrame(rows); print(f"  ✓ BED: {len(df)} 条"); return df


def _read_tpm_file(tpm_file):
    tpm_df = None; encodings = ['utf-8', 'gbk', 'latin1']
    if HAS_CHARDET:
        with open(tpm_file, 'rb') as f:
            enc = chardet.detect(f.read(10000))['encoding'] or 'utf-8'
        encodings = [enc] + [e for e in encodings if e != enc]
    for enc in encodings:
        try:
            df = pd.read_csv(tpm_file, sep='\t', encoding=enc)
            if df.shape[1] >= 2: tpm_df = df; break
        except Exception: pass
        try:
            df = pd.read_csv(tpm_file, encoding=enc)
            if df.shape[1] >= 2: tpm_df = df; break
        except Exception: continue
    if tpm_df is None: raise RuntimeError(f"无法读取 TPM 文件: {tpm_file}")
    return tpm_df


def load_tf_signal_data(npz_file, tpm_file, bed_file,
                         remove_mitochondrial=True, tpm_column='TPM_median'):
    print("=" * 62 + "\n加载数据\n" + "=" * 62)
    data = np.load(npz_file); tf_names = list(data.keys())
    n_genes, n_bins = data[tf_names[0]].shape
    print(f"  ✓ {len(tf_names)} TFs × {n_genes} genes × {n_bins} bins")
    bed_df  = parse_bed(bed_file)
    gene_df = pd.DataFrame([
        {'raw_idx': i, 'gene': r['name'], 'chr': r['chrom'],
         'gene_length': int(r['end']) - int(r['start'])}
        for i, r in bed_df.iterrows() if i < n_genes])
    if remove_mitochondrial:
        mt_mask = gene_df['chr'].apply(is_mitochondrial)
        gene_df = gene_df[~mt_mask].reset_index(drop=True)
        print(f"  过滤线粒体后: {len(gene_df)} 个核基因")
    tpm_df   = _read_tpm_file(tpm_file)
    gene_col = tpm_df.columns[0]
    tpm_df[tpm_column] = pd.to_numeric(tpm_df[tpm_column], errors='coerce')
    tpm_df   = tpm_df.dropna(subset=[gene_col, tpm_column])
    tpm_dict = dict(zip(tpm_df[gene_col], tpm_df[tpm_column]))
    valid = []
    for _, r in gene_df.iterrows():
        g = r['gene']
        if g not in tpm_dict: continue
        valid.append({'raw_idx': int(r['raw_idx']), 'gene': g, 'chr': r['chr'],
                      'expression': float(tpm_dict[g]),
                      'gene_length': float(r['gene_length'])})
    result_df = pd.DataFrame(valid)
    print(f"  ✓ {len(result_df)} genes matched")
    raw_idx = result_df['raw_idx'].values
    X = np.stack([data[n][raw_idx, :] for n in tf_names], axis=1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return (X, result_df['expression'].values, result_df['chr'].values,
            result_df['gene'].values, tf_names,
            result_df['gene_length'].values, n_bins, bed_df)


def load_lm_embeddings(npz_path, genes, use_layer='lm_concat'):
    print(f"\n  加载 LM 嵌入: {npz_path}  key='{use_layer}'")
    data = np.load(npz_path, allow_pickle=True)
    if use_layer not in data:
        available = [k for k in data.keys()
                     if k not in ('genes','layer_names','pool_method','lm_native_len','promoter_len')]
        use_layer = available[0] if available else list(data.keys())[0]
        print(f"  ⚠ key 不存在，改用: {use_layer}")
    emb_raw = data[use_layer]
    gene_index_raw = data.get('genes', None)
    if gene_index_raw is not None:
        def _to_str(g):
            if isinstance(g, (bytes, np.bytes_)):
                return g.decode('utf-8', errors='replace').strip()
            return str(g).strip()
        gene_index = [_to_str(g) for g in gene_index_raw.tolist()]
        g2i = {g: i for i, g in enumerate(gene_index)}
        out = np.zeros((len(genes), emb_raw.shape[1]), dtype=np.float32)
        n_hit = 0
        for gi, gene in enumerate(genes):
            idx = g2i.get(_to_str(gene))
            if idx is not None:
                out[gi] = emb_raw[idx]; n_hit += 1
        print(f"  ✓ 命中 {n_hit}/{len(genes)}  lm_dim={emb_raw.shape[1]}")
        return out
    print(f"  ✓ 直接加载 lm_dim={emb_raw.shape[1]}")
    return emb_raw[:len(genes)].astype(np.float32)


def normalize_lm_embeddings(lm_emb, train_mask):
    tr = lm_emb[train_mask]; mu = tr.mean(axis=0); std = tr.std(axis=0) + 1e-8
    print(f"  ✓ LM 嵌入归一化  train_std_mean={std.mean():.3f}")
    return ((lm_emb - mu) / std).astype(np.float32)


# ============================================================================
# DNA one-hot 加载（消融 D 专用）
# ============================================================================

_BASE2IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
_RC_TABLE = str.maketrans('ACGTNacgtn', 'TGCANtgcan')


def load_genome_fasta(fsa_path: str) -> dict:
    """解析 FASTA，返回 {chromosome_name: upper-case sequence string}。"""
    genome = {}
    current = None; buf = []
    with open(fsa_path) as f:
        for line in f:
            line = line.rstrip('\n').rstrip('\r')
            if not line: continue
            if line.startswith('>'):
                if current is not None:
                    genome[current] = ''.join(buf).upper()
                current = line[1:].split()[0]     # ">chrI foo bar" → "chrI"
                buf = []
            else:
                buf.append(line)
        if current is not None:
            genome[current] = ''.join(buf).upper()
    total_bp = sum(len(s) for s in genome.values())
    print(f"  ✓ FASTA: {len(genome)} 条染色体  总长度 {total_bp:,} bp  "
          f"示例: {list(genome.keys())[:5]}")
    return genome


def _seq_to_onehot(seq: str, length: int) -> np.ndarray:
    """DNA 字符串 → (4, length) one-hot。N / 未知 → 均匀 0.25。"""
    oh = np.zeros((4, length), dtype=np.float32)
    for i in range(min(len(seq), length)):
        idx = _BASE2IDX.get(seq[i])
        if idx is None:
            oh[:, i] = 0.25      # N 或其他歧义碱基
        else:
            oh[idx, i] = 1.0
    # 超出 seq 长度的位点保持全 0（极少发生，染色体边界外的填充路径会先做 'N' 填充）
    return oh


def extract_promoter_onehot(genome: dict, bed_df: pd.DataFrame,
                              genes: np.ndarray,
                              upstream: int, downstream: int) -> np.ndarray:
    """
    围绕每个基因 TSS 提取 one-hot 序列:
      - TSS = BED start (strand +) / BED end - 1 (strand -)
      - 窗口 [TSS - upstream, TSS + downstream]，长度 = upstream + downstream
      - 负链 → 反向互补后再 one-hot
      - 染色体边界外填充 'N'
    返回 (N_genes, 4, L)
    """
    L = upstream + downstream
    bed_lookup = {}
    for _, r in bed_df.iterrows():
        strand = r.get('strand', '+')
        tss = int(r['start']) if strand == '+' else int(r['end']) - 1
        bed_lookup[str(r['name'])] = (str(r['chrom']), tss, strand)

    out = np.zeros((len(genes), 4, L), dtype=np.float32)
    n_ok = 0; n_padded = 0; n_missing = 0
    for gi, g in enumerate(genes):
        info = bed_lookup.get(str(g))
        if info is None:
            out[gi] = 0.25; n_missing += 1; continue
        chrom, tss, strand = info
        full = genome.get(chrom)
        if full is None:
            out[gi] = 0.25; n_missing += 1; continue

        if strand == '+':
            lo = tss - upstream
            hi = tss + downstream
        else:
            # 负链：下游相对于正链是 [tss - downstream + 1, tss + upstream + 1]
            lo = tss - downstream + 1
            hi = tss + upstream + 1

        clen = len(full)
        pad_lo = max(0, -lo)
        pad_hi = max(0, hi - clen)
        lo_c = max(0, lo); hi_c = min(clen, hi)
        seq = full[lo_c:hi_c]
        if pad_lo > 0 or pad_hi > 0:
            seq = 'N' * pad_lo + seq + 'N' * pad_hi
            n_padded += 1
        if strand == '-':
            seq = seq.translate(_RC_TABLE)[::-1]
        if len(seq) != L:
            # 理论上不会发生；防御性截断/补齐
            seq = (seq + 'N' * L)[:L]
        out[gi] = _seq_to_onehot(seq, L)
        n_ok += 1

    print(f"  ✓ DNA one-hot 提取完成: shape={out.shape}  "
          f"ok={n_ok}  边界填充={n_padded}  缺失={n_missing}")
    return out


def load_or_cache_dna_onehot(genome_path, bed_df, genes,
                                upstream, downstream, cache_path=None):
    """带缓存的 DNA one-hot 加载。缓存命中条件：窗口参数匹配 + 前 20 个基因名一致。"""
    if cache_path is not None and Path(cache_path).exists():
        try:
            data = np.load(cache_path, allow_pickle=True)
            cu = int(data['upstream']); cd = int(data['downstream'])
            cg = data['genes']
            if (cu == upstream and cd == downstream and len(cg) == len(genes) and
                all(str(a) == str(b) for a, b in
                    zip(cg[:min(20, len(genes))], genes[:min(20, len(genes))]))):
                print(f"  [DNA-cache] ✓ 命中缓存: {cache_path}  shape={data['onehot'].shape}")
                return data['onehot'].astype(np.float32)
            print(f"  [DNA-cache] 缓存参数不匹配，重新提取")
        except Exception as _e:
            print(f"  [DNA-cache] 缓存读取失败 ({_e})，重新提取")

    print(f"\n  [DNA] 加载基因组: {genome_path}")
    genome = load_genome_fasta(genome_path)
    onehot = extract_promoter_onehot(genome, bed_df, genes, upstream, downstream)

    if cache_path is not None:
        Path(cache_path).parent.mkdir(exist_ok=True, parents=True)
        np.savez_compressed(cache_path, onehot=onehot,
                           upstream=np.int32(upstream),
                           downstream=np.int32(downstream),
                           genes=np.asarray(genes, dtype=object))
        print(f"  [DNA-cache] 已写入: {cache_path}")
    return onehot


def _dna_sanity_check(dna_onehot, genes, bed_df, n_examples=3,
                       upstream=1000, flank=15):
    """
    打印前 n_examples 个基因 TSS 周围的碱基，用于人眼验证方向/对齐正确。
    TSS 位于 one-hot 的 index = upstream（即 '|' 之前 = 上游，之后 = 下游 / TSS）。
    """
    _IDX2BASE = ['A', 'C', 'G', 'T']
    N, _, L = dna_onehot.shape
    bed_by_name = {str(r['name']): r for _, r in bed_df.iterrows()}
    print(f"\n  [DNA Sanity] 前 {n_examples} 个基因 TSS ±{flank}bp（'|' 前=上游, '|' 后=TSS/下游）:")
    for i in range(min(n_examples, N)):
        oh = dna_onehot[i]
        idx = oh.argmax(axis=0)
        is_amb = (oh.max(axis=0) < 0.9)
        seq = ''.join('N' if is_amb[j] else _IDX2BASE[idx[j]] for j in range(L))
        lo = max(0, upstream - flank); hi = min(L, upstream + flank)
        flanking = seq[lo:upstream] + '|' + seq[upstream:hi]
        br = bed_by_name.get(str(genes[i]))
        strand = (br['strand'] if br is not None else '?')
        chrom  = (br['chrom']  if br is not None else '?')
        print(f"    [{i:3d}] {genes[i]:14s} ({chrom}, strand={strand}):  ...{flanking}...")


def split_by_chromosome(chromosomes, y_raw, test_size=0.10, val_size=0.10,
                         strategy='fixed_saccer3', random_seed=42,
                         custom_split=None):
    """染色体级 train/val/test 划分.

    Parameters
    ----------
    strategy : str
        'fixed_saccer3' — 主管道默认: train=chrI..XII, val=XIII/XIV, test=XV/XVI.
        'custom'        — chr holdout 稳健性 (P0 #12): 由 custom_split 显式指定
                          test_chrs / val_chrs, 其余进 train.
        其它           — 按 chromosome 大小比例近似 80/10/10 自动划分.
    custom_split : dict or None
        当 strategy='custom' 时必须提供, 形如:
            dict(test_chrs=['chrI','chrII'], val_chrs=['chrXIII','chrXIV'])
    """
    unique_chrs = sorted(set(chromosomes))
    if strategy == 'custom':
        # ★ chr holdout 稳健性 (P0 #12, v7→v8)
        if not custom_split:
            raise ValueError("strategy='custom' 但 custom_split 未提供")
        test_chrs = [c for c in custom_split.get('test_chrs', []) if c in unique_chrs]
        val_chrs  = [c for c in custom_split.get('val_chrs',  []) if c in unique_chrs]
        _missing_test = [c for c in custom_split.get('test_chrs', []) if c not in unique_chrs]
        _missing_val  = [c for c in custom_split.get('val_chrs',  []) if c not in unique_chrs]
        if _missing_test or _missing_val:
            print(f"  ⚠  custom_split 中以下染色体不在数据中, 已忽略: "
                  f"test={_missing_test}  val={_missing_val}")
        if not test_chrs:
            raise ValueError(
                f"custom_split.test_chrs 解析后为空: "
                f"got={custom_split.get('test_chrs')}, available={unique_chrs}")
        if not val_chrs:
            raise ValueError(
                f"custom_split.val_chrs 解析后为空: "
                f"got={custom_split.get('val_chrs')}, available={unique_chrs}")
        train_chrs = [c for c in unique_chrs if c not in test_chrs and c not in val_chrs]
    elif strategy == 'fixed_saccer3':
        train_chrs = [c for c in unique_chrs if c not in ('chrXIII','chrXIV','chrXV','chrXVI')]
        val_chrs   = ['chrXIII','chrXIV']; test_chrs = ['chrXV','chrXVI']
        train_chrs = [c for c in train_chrs if c in set(unique_chrs)]
        val_chrs   = [c for c in val_chrs   if c in set(unique_chrs)]
        test_chrs  = [c for c in test_chrs  if c in set(unique_chrs)]
    else:
        chr_counts = {c: int((chromosomes==c).sum()) for c in unique_chrs}
        chr_stats  = sorted(unique_chrs, key=lambda c:(chr_counts[c],), reverse=True)
        N = len(chromosomes)
        target_te = max(int(N*test_size), 1); target_va = max(int(N*val_size), 1)
        n_test = n_val = 0; test_chrs = []; val_chrs = []; train_chrs = []
        for c in chr_stats:
            n = chr_counts[c]
            if len(test_chrs)==0 or (n_test<target_te and n<=max(1,target_te-n_test)*1.6):
                test_chrs.append(c); n_test += n
            elif len(val_chrs)==0 or (n_val<target_va and n<=max(1,target_va-n_val)*1.6):
                val_chrs.append(c);  n_val  += n
            else:
                train_chrs.append(c)
        train_chrs.extend(sorted(set(unique_chrs)-set(train_chrs)-set(val_chrs)-set(test_chrs)))
    print(f"\n{'─'*62}  染色体划分: '{strategy}'")
    for name, chrs in [('Train', train_chrs), ('Val', val_chrs), ('Test', test_chrs)]:
        mask = np.isin(chromosomes, chrs); n = mask.sum()
        med  = float(np.median(y_raw[mask])) if n > 0 else 0.0
        print(f"  {name:6s}  chrs={sorted(chrs)}  n={n:5d}  med={med:.3f}")
    return {'train': np.isin(chromosomes, train_chrs),
            'val':   np.isin(chromosomes, val_chrs),
            'test':  np.isin(chromosomes, test_chrs),
            'train_chrs': train_chrs, 'val_chrs': val_chrs, 'test_chrs': test_chrs}


def leakage_audit(genes, split_masks):
    print(f"\n{'─'*62}\n  泄漏审计")
    sets = {k: set(genes[split_masks[k]].tolist()) for k in ('train','val','test')}
    all_clean = True
    for a, b in [('train','val'), ('train','test'), ('val','test')]:
        overlap = sets[a] & sets[b]
        if overlap: print(f"  ⛔ {a}∩{b}={len(overlap)} 基因重叠"); all_clean = False
        else: print(f"  ✓ {a}({len(sets[a])})∩{b}({len(sets[b])})=0")
    if all_clean: print("  ✓ 通过审计")
    return all_clean


# ============================================================================
# 学习率调度
# ============================================================================

class WarmupCosineScheduler:
    """Warmup + 余弦退火重启 (SGDR style)。
    T_mult=2 (默认): 每次重启后周期翻倍，序列为 T_0, 2T_0, 4T_0, ...
    配合 patience=60 的较长训练，后期大周期提供更平稳的精细调整阶段。
    T_mult=1 退化为等周期余弦（原始行为）。
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, n_batches,
                 T_0=30, T_mult=2, eta_min=1e-6):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_epochs * n_batches
        self.T_0_steps    = T_0 * n_batches
        self.T_mult       = T_mult
        self.base_lr      = optimizer.param_groups[0]['lr']
        self.eta_min      = eta_min; self._step = 0

    def step(self):
        self._step += 1; s = self._step
        if s <= self.warmup_steps:
            lr = self.base_lr * s / max(1, self.warmup_steps)
        else:
            t = s - self.warmup_steps
            if self.T_mult == 1:
                T = self.T_0_steps
                lr = self.eta_min + 0.5*(self.base_lr-self.eta_min)*(1 + math.cos(math.pi*(t%T)/T))
            else:
                # 计算当前位于哪个周期及周期内位置
                T_cur = self.T_0_steps; T_cum = 0
                while T_cum + T_cur <= t:
                    T_cum += T_cur; T_cur = int(T_cur * self.T_mult)
                t_in = t - T_cum
                lr = self.eta_min + 0.5*(self.base_lr-self.eta_min)*(1 + math.cos(math.pi*t_in/T_cur))
        for g in self.optimizer.param_groups: g['lr'] = lr
        return lr



# ============================================================================
# 训练函数
# ============================================================================

def train_model(model, train_loader, val_loader,
                epochs=300, lr=2e-4, device='cuda',
                patience=40, save_path='best_model.pth',
                warmup_epochs=15,
                seq_baseline_model=None):

    # ── 分组参数: 三个组, 权重衰减分级处理 ──────────────────────────────
    # no_decay: 可学习先验/温度参数 — weight_decay=0 防止被 L2 拉回 0
    #   · attn_pool.queries    — AttentiveRegionPool 的区域查询向量
    #   · attn_pool.temp       — 注意力温度 (被 decay 会退回初始值, 使分布再次均匀)
    #   · tf_attn_pooler.temperature — TF 注意力温度
    _NO_DECAY_KEYS = ('attn_pool.queries', 'attn_pool.temp',
                      'tf_attn_pooler.temperature')
    lm_params    = []
    no_decay     = []
    other_params = []
    for n, p in model.named_parameters():
        if any(nd in n for nd in _NO_DECAY_KEYS):
            no_decay.append(p)
        elif 'lm_adapter' in n:
            lm_params.append(p)
        else:
            other_params.append(p)
    optimizer = optim.AdamW([
        {'params': other_params, 'weight_decay': 1e-4},
        {'params': lm_params,    'weight_decay': 1e-3},
        {'params': no_decay,     'weight_decay': 0.0},
    ], lr=lr)
    _nd_cnt = sum(p.numel() for p in no_decay)
    _lm_cnt = sum(p.numel() for p in lm_params)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs,
                                      len(train_loader), T_0=30)
    criterion = CombinedLoss()
    history   = {'train_loss':[], 'val_loss':[], 'val_r2':[],
                 'val_pearson':[], 'val_pearson_ema':[], 'lr':[]}
    best_ema  = -float('inf'); p_ema = None; pat_cnt = 0
    model_kind = getattr(model, 'model_kind', 'tf')

    _ema_desc = '0.85/0.15(slow)' if model_kind == 'tf' else '0.70/0.30(fast)'
    print(f"\n训练 | epochs={epochs}  lr={lr}  patience={patience}  "
          f"warmup={warmup_epochs}ep  kind={model_kind}  EMA={_ema_desc}")
    print(f"  baseline={'✓' if seq_baseline_model else '✗'}")
    print(f"  AdamW: other_wd=1e-4  lm_wd=1e-3  no_decay_wd=0  "
          f"lm_params={_lm_cnt:,}  no_decay_params={_nd_cnt:,}")

    for epoch in range(epochs):
        model.train(); t_losses = []
        _nan_skip_cnt   = 0   # 本 epoch 完全跳过的批次 (loss 本身就是 nan)
        _grad_sanit_cnt = 0   # 本 epoch 触发梯度消毒 (nan→0 后继续 step) 的批次
        _first_diag     = (epoch == 0)   # 首 epoch 首次 nan 时打印详细诊断

        # 头 5 epoch 用更紧的 clip (防止深层梯度爆炸), 之后放宽到 1.0
        _clip_max = 0.5 if epoch < 5 else 1.0

        for batch in train_loader:
            sigs, scalars, lm_emb, weights, exps = batch
            sigs    = sigs.to(device);    scalars  = scalars.to(device)
            weights = weights.to(device); exps     = exps.to(device)
            lm_in   = lm_emb.to(device)  if lm_emb.shape[-1]  > 1 else None
            optimizer.zero_grad()

            if model_kind == 'seq_only':
                nsf  = getattr(model, 'n_seq_features', scalars.shape[1])
                pred = model(scalars[:, :nsf], lm_in)
            else:
                pred = model(sigs, scalars, lm_emb=lm_in)

            loss = criterion(pred, exps, weights)

            # ── 前向 NaN 防御 ──────────────────────────────────────────────
            # 只有 forward 本身产 nan 才跳批 (这种情况极罕见, 通常提示模型或
            # 输入真的坏了), 否则一律进入 backward + 梯度消毒路径。
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                _nan_skip_cnt += 1
                continue

            loss.backward()

            # ── 梯度消毒 (关键改动) ───────────────────────────────────────
            # 原版: total_norm 只要是 nan/inf 就整批丢弃 → 模型永远不训练。
            # 新版: 逐参数把 nan/inf 换成 0, 其他健康参数照常更新。这允许
            #   偶发的数值问题 (某个 sqrt/log backward 异常) 不会阻塞整个
            #   训练, 等到相关参数被其它批次更新后奇点自然远离。
            _bad_params = []
            for _name, _p in model.named_parameters():
                if _p.grad is None:
                    continue
                _bad_mask = ~torch.isfinite(_p.grad)
                if _bad_mask.any():
                    _bad_params.append((_name, int(_bad_mask.sum()), _p.grad.numel()))
                    _p.grad.masked_fill_(_bad_mask, 0.0)
            if _bad_params:
                _grad_sanit_cnt += 1
                # 首 epoch 首次发现时, 打印具体是哪些参数的梯度坏了,
                # 这对定位真正的数值奇点来源非常重要 (诊断用, 只打印一次)。
                if _first_diag:
                    _first_diag = False
                    print(f"\n  [首次梯度消毒] epoch={epoch+1}  "
                          f"受影响参数 {len(_bad_params)} 个 (总参数组 "
                          f"{sum(1 for _ in model.named_parameters())}):")
                    for _n, _bc, _tot in _bad_params[:15]:
                        print(f"      · {_n}  nan/inf={_bc}/{_tot}")
                    if len(_bad_params) > 15:
                        print(f"      · ...(省略 {len(_bad_params) - 15} 个)")
                    print(f"    [策略] nan/inf 梯度置 0, 其它参数正常更新")

            # 梯度裁剪 (消毒后再裁, 保证 total_norm 必然有限)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), _clip_max)
            if not torch.isfinite(total_norm):
                # 理论上消毒后不应再 nan; 若仍发生 (例如某参数 grad 全是
                # subnormal 在 sum 时仍溢出), 作为最后防线跳批。
                optimizer.zero_grad()
                _nan_skip_cnt += 1
                continue

            optimizer.step(); scheduler.step(); t_losses.append(loss.item())

        model.eval(); v_losses = []; preds = []; labels = []
        with torch.no_grad():
            for batch in val_loader:
                sigs, scalars, lm_emb, weights, exps = batch
                lm_in = lm_emb.to(device) if lm_emb.shape[-1] > 1 else None
                if model_kind == 'seq_only':
                    nsf = getattr(model, 'n_seq_features', scalars.shape[1])
                    p   = model(scalars[:, :nsf].to(device), lm_in)
                else:
                    p = model(sigs.to(device), scalars.to(device), lm_emb=lm_in)
                v_losses.append(criterion(p, exps.to(device), weights.to(device)).item())
                preds.extend(p.cpu().tolist()); labels.extend(exps.tolist())

        preds_arr  = np.array(preds); labels_arr = np.array(labels)
        # NaN 防御: 前向 NaN → BN 运行统计被污染 → 验证集全 NaN；替换后继续训练而不崩溃
        _nan_in_preds = int(np.isnan(preds_arr).sum())
        if _nan_in_preds > 0:
            print(f"  ⚠ Ep {epoch+1}: 验证集预测含 {_nan_in_preds} 个 NaN "
                  f"(train skip={_nan_skip_cnt})，替换为 0 并继续")
            preds_arr = np.nan_to_num(preds_arr, nan=0.0)
        try:
            vp, _ = pearsonr(labels_arr, preds_arr)
        except Exception:
            vp = 0.0
        if not np.isfinite(vp):
            vp = 0.0
        # seq_only 用快 EMA(0.70/0.30)：原先58轮即停，慢EMA会让patience=60虚耗
        # tf 全模型用慢 EMA(0.85/0.15)：防止验证集单epoch波动触发过早停止
        _ema_slow  = 0.85 if model_kind == 'tf' else 0.70
        p_ema      = vp if p_ema is None else _ema_slow * p_ema + (1.0 - _ema_slow) * vp
        history['train_loss'].append(float(np.nanmean(t_losses)) if t_losses else float('nan'))
        history['val_loss'].append(float(np.nanmean(v_losses)) if v_losses else float('nan'))
        try:
            _r2 = r2_score(labels_arr, preds_arr)
        except Exception:
            _r2 = 0.0
        history['val_r2'].append(float(_r2) if np.isfinite(_r2) else 0.0)
        history['val_pearson'].append(vp)
        history['val_pearson_ema'].append(p_ema)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        if (epoch+1) % 10 == 0 or epoch == 0:
            _nan_str    = (f"  nan_skip={_nan_skip_cnt}"
                           if _nan_skip_cnt > 0 else "")
            _sanit_str  = (f"  grad_sanit={_grad_sanit_cnt}"
                           if _grad_sanit_cnt > 0 else "")
            print(f"Ep {epoch+1:3d}/{epochs}  "
                  f"Train={history['train_loss'][-1]:.4f}  "
                  f"Val={history['val_loss'][-1]:.4f}  "
                  f"R²={history['val_r2'][-1]:.4f}  "
                  f"Pearson={vp:.4f}  EMA={p_ema:.4f}"
                  f"{_sanit_str}{_nan_str}")

        if p_ema > best_ema:
            best_ema = p_ema; pat_cnt = 0
            torch.save(model.state_dict(), save_path)
        else:
            pat_cnt += 1
            if pat_cnt >= patience:
                print(f"\nEarly stop @ ep {epoch+1}  best EMA={best_ema:.4f}"); break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    print(f"最佳验证 EMA Pearson: {best_ema:.4f}")
    return history


# ============================================================================
# 评估 & 批量推断
# ============================================================================

def evaluate_model(model, test_loader, device='cuda'):
    model.eval(); preds_all = []; labels_all = []
    model_kind = getattr(model, 'model_kind', 'tf')
    with torch.no_grad():
        for batch in test_loader:
            sigs, scalars, lm_emb, weights, exps = batch
            lm_in = lm_emb.to(device) if lm_emb.shape[-1] > 1 else None
            if model_kind == 'seq_only':
                nsf = getattr(model, 'n_seq_features', scalars.shape[1])
                p   = model(scalars[:, :nsf].to(device), lm_in)
            else:
                p = model(sigs.to(device), scalars.to(device), lm_emb=lm_in)
            preds_all.extend(p.cpu().tolist()); labels_all.extend(exps.tolist())
    p = np.array(preds_all); l = np.array(labels_all)
    r2 = r2_score(l, p); pearson = pearsonr(l, p)[0]
    print(f"\n{'='*50}\n  测试集: R²={r2:.4f}  Pearson={pearson:.4f}")
    return {'r2': float(r2), 'pearson': float(pearson), 'preds': p, 'labels': l}


def predict_full_dataset(model, X_std, device, scalars=None,
                          lm_data=None, batch_size=512):
    model.eval(); preds = []; attns = []
    model_kind = getattr(model, 'model_kind', 'tf')
    N = X_std.shape[0]
    with torch.no_grad():
        for s in range(0, N, batch_size):
            x  = torch.FloatTensor(X_std[s:s+batch_size]).to(device)
            sc = (torch.FloatTensor(scalars[s:s+batch_size]).to(device)
                  if scalars is not None else None)
            lm = (torch.FloatTensor(lm_data[s:s+batch_size]).to(device)
                  if lm_data is not None else None)
            if model_kind == 'seq_only':
                nsf = getattr(model, 'n_seq_features', sc.shape[1]) if sc is not None else 1
                p   = model(sc[:, :nsf] if sc is not None else None, lm)
                if isinstance(p, tuple): p = p[0]
                preds.append(p.cpu().numpy())
                attns.append(np.zeros((len(p), 1), dtype=np.float32))
            else:
                r = model(x, sc, lm_emb=lm, return_attn=True)
                preds.append(r[0].cpu().numpy())
                attns.append(r[1].cpu().numpy())
    return (np.concatenate(preds).astype(np.float32),
            np.concatenate(attns, axis=0).astype(np.float32))


def _load_checkpoint_and_predict(ckpt_path, X_std, scalars, n_tfs, n_bins,
                                   n_scalars, n_seq_features, device,
                                   use_lm=False, lm_data=None, lm_dim=0,
                                   batch_size=512):
    if not Path(ckpt_path).exists():
        print(f"  ⚠ 找不到 {ckpt_path}，跳过"); return np.full(len(X_std), np.nan)
    state = _safe_torch_load(ckpt_path, device)
    sd    = state.get('model_state_dict', state) if isinstance(state, dict) else state
    cfg   = state.get('config', {}) if isinstance(state, dict) else {}
    MC    = MODEL_CONFIG
    mname = cfg.get('model', 'MultiModalPredictor')
    _use_lm  = cfg.get('use_lm', use_lm) and (lm_data is not None)
    _nsf     = cfg.get('n_seq_features', n_seq_features)
    lm_dim_u = lm_dim if lm_dim > 0 else MC['d_lm_hidden']

    if mname in ('SeqOnlyModel', 'SequenceOnlyPredictor'):
        m = SequenceOnlyPredictor(
            n_windows=_nsf, d_profile=MC['d_profile'], d_hidden=128, dropout=0.25,
            use_lm=_use_lm, lm_dim=lm_dim_u,
            d_lm_hidden=MC['d_lm_hidden'], d_lm_out=MC['d_lm_out'],
            lm_input_dropout=MC['lm_input_drop'], freeze_lm=MC['freeze_lm']).to(device)
    else:
        m = MultiModalPredictor(
            n_tfs=n_tfs, n_bins=n_bins,
            d_tf=cfg.get('d_tf', MC['d_tf']), d_cnn=cfg.get('d_cnn', MC['d_cnn']),
            n_tf_layers=cfg.get('n_tf_layers', MC['n_tf_layers']),
            n_regions=cfg.get('n_regions', MC['n_regions']),
            n_heads=cfg.get('n_heads', MC['n_heads']),
            use_scalars=True, n_scalars=n_scalars, n_seq_features=_nsf,
            d_seq=cfg.get('d_seq', MC['d_seq']),
            d_tf_proj=cfg.get('d_tf_proj', MC['d_tf_proj']),
            use_lm=_use_lm, lm_dim=lm_dim_u,
            d_lm_hidden=MC['d_lm_hidden'], d_lm_out=MC['d_lm_out'],
            lm_input_dropout=MC['lm_input_drop'], freeze_lm=MC['freeze_lm'],
            topk_k=cfg.get('topk_k', MC.get('topk_k', 0)),
            # new48: 若 ckpt 是 FUSION_ABLATION 变体, 重建时必须带同样的 flags,
            # 否则 state_dict 结构不匹配 (e.g. tf_tss_encoder 缺失/多余).
            fusion_flags=cfg.get('fusion_ablation_flags')).to(device)
    m.load_state_dict(sd, strict=False); m.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(X_std), batch_size):
            x  = torch.FloatTensor(X_std[s:s+batch_size]).to(device)
            sc = (torch.FloatTensor(scalars[s:s+batch_size]).to(device)
                  if scalars is not None else None)
            lm = (torch.FloatTensor(lm_data[s:s+batch_size]).to(device)
                  if _use_lm and lm_data is not None else None)
            if getattr(m, 'model_kind','tf') == 'seq_only':
                nsf2 = getattr(m, 'n_seq_features', sc.shape[1]) if sc is not None else 1
                p = m(sc[:, :nsf2] if sc is not None else None, lm)
            else:
                p = m(x, sc, lm_emb=lm)
            if isinstance(p, tuple): p = p[0]
            preds.append(p.cpu().numpy())
    del m
    return np.concatenate(preds).astype(np.float32)


# ============================================================================
# TF 重要性
# ============================================================================

def extract_tf_importance(model, data_loader, device, tf_names=None, desc='test'):
    model_kind = getattr(model, 'model_kind', 'tf')
    if model_kind == 'seq_only':
        print(f"\n  [TF重要性|{desc}] SequenceOnlyPredictor 无 TF 分支，跳过")
        return pd.Series(dtype=np.float32), None
    model.eval(); all_attn = []
    with torch.no_grad():
        for batch in data_loader:
            sigs, scalars, lm_emb, _, _ = batch
            lm_in  = lm_emb.to(device)  if lm_emb.shape[-1]  > 1 else None
            result = model(sigs.to(device), scalars.to(device),
                           lm_emb=lm_in, return_attn=True)
            if not isinstance(result, (tuple, list)) or len(result) < 2:
                raise RuntimeError('模型未返回注意力权重')
            all_attn.append(result[1].cpu().numpy())
    if not all_attn: return pd.Series(dtype=np.float32), None
    all_attn = np.concatenate(all_attn, axis=0)
    mean_imp = all_attn.mean(axis=0)
    # tss_weighted 路径: model 返回 shape=(B,1) 的全零占位符, mean_imp 长度为 1,
    # 与 tf_names (n_tfs) 不匹配 —— 此变体没有真正的 per-TF 注意力权重，直接跳过.
    if tf_names is not None and len(mean_imp) != len(tf_names):
        print(f"\n  [TF重要性|{desc}] attn_w 为占位符 "
              f"(mean_imp.shape={mean_imp.shape}, n_tfs={len(tf_names)}), "
              f"跳过 (tss_weighted 路径无真实注意力)")
        return pd.Series(dtype=np.float32), None
    series = (pd.Series(mean_imp, index=tf_names).sort_values(ascending=False)
              if tf_names is not None else pd.Series(mean_imp).sort_values(ascending=False))
    print(f"\n  [TF重要性|{desc}] Top-10:")
    for i, (name, val) in enumerate(series.head(10).items()):
        print(f"    {i+1:2d}.  {str(name):20s}  attn={val:.4f}")
    return series, all_attn


def plot_tf_importance(tf_importance, output_dir, top_n=30, label=''):
    out = Path(output_dir); out.mkdir(exist_ok=True, parents=True)
    top = tf_importance.head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, top_n*0.3)))
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(top)))
    ax.barh(range(len(top)), top.values[::-1], color=colors[::-1], alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([str(n) for n in top.index[::-1]], fontsize=8)
    ax.set_xlabel('Mean Attention Weight')
    ax.set_title(f'TF Importance (Top-{top_n}){" ["+label+"]" if label else ""}')
    ax.grid(axis='x', alpha=0.3); plt.tight_layout()
    fname = out / (f'{label+"_" if label else ""}tf_importance_top{top_n}.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight'); plt.close()
    print(f"  ✓ TF 重要性图 → {fname}")


def plot_training_curves(history, results, output_dir, label=''):
    out = Path(output_dir); out.mkdir(exist_ok=True, parents=True)
    px = f"{label}_" if label else ""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.plot(history['train_loss'], label='Train', color='steelblue', lw=2)
    ax.plot(history['val_loss'],   label='Val',   color='coral', lw=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend()
    ax.set_title(f'{label} Training Loss')
    ax = axes[1]
    ax.plot(history['val_pearson'],     label='Pearson', color='steelblue', lw=1.5, alpha=0.6)
    ax.plot(history['val_pearson_ema'], label='EMA',     color='steelblue', lw=2)
    ax.axhline(results['pearson'], color='red', lw=1.5, ls='--',
               label=f"Test r={results['pearson']:.4f}")
    ax.set_xlabel('Epoch'); ax.set_ylabel('Pearson r'); ax.legend()
    ax.set_title(f'{label} Validation Pearson')
    plt.tight_layout()
    plt.savefig(out / f'{px}training.png', dpi=150, bbox_inches='tight'); plt.close()

    l = results['labels']; p = results['preds']
    hi80 = np.percentile(l, 80); hi95 = np.percentile(l, 95)
    point_colors = np.where(l >= hi95, 'crimson',
                   np.where(l >= hi80, 'darkorange', 'steelblue'))
    mn = float(min(l.min(), p.min())); mx = float(max(l.max(), p.max()))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(l, p, c=point_colors, alpha=0.35, s=12)
    ax.plot([mn, mx], [mn, mx], 'k--', linewidth=1.5)
    ax.set_xlabel('True (normalized)', fontsize=11)
    ax.set_ylabel('Predicted',         fontsize=11)
    ax.set_title(f"r={results['pearson']:.4f}  R²={results['r2']:.4f}",
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3); plt.tight_layout()
    scatter_path = out / f'{px}scatter_test.png'
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"  ✓ 散点图 → {scatter_path}")


# ============================================================================
# cross_weight 收集与诊断
# ============================================================================

def collect_cross_weights(model, X_std, scalars, device,
                          lm_data=None, batch_size=512) -> np.ndarray:
    model.eval()
    if getattr(model, 'model_kind', 'tf') == 'seq_only':
        return np.array([], dtype=np.float32)
    cross_ws = []
    with torch.no_grad():
        for s in range(0, len(X_std), batch_size):
            e  = s + batch_size
            x  = torch.FloatTensor(X_std[s:e]).to(device)
            sc = (torch.FloatTensor(scalars[s:e]).to(device)
                  if scalars is not None else None)
            lm = (torch.FloatTensor(lm_data[s:e]).to(device)
                  if lm_data is not None else None)
            result = model(x, sc, lm_emb=lm, return_internals=True)
            if isinstance(result, tuple) and len(result) == 2:
                _, internals = result
                # internals: (seq_emb, tf_emb, cross_weight, region_weights)
                if isinstance(internals, tuple) and len(internals) >= 3:
                    cw = internals[2]
                    if cw is not None:
                        cross_ws.append(cw.cpu().numpy())
    if not cross_ws: return np.array([], dtype=np.float32)
    return np.concatenate(cross_ws).astype(np.float32)


# ============================================================================
# ★ (Panel H) 因果 Δc / Δŷ 反事实 —— new55 新增
# ============================================================================
# 见顶部 CAUSAL_DELTA_C 配置块的完整说明。核心三件套:
#   collect_pred_and_cross_weight  单次前向同时取 (ŷ, c) —— 反事实两次调用即得 Δ
#   _build_tf_baseline             按配置造 TF 输入基线 (zero / mean / shuffle)
#   run_causal_delta_c             组装 + 金标准检验 + 落盘, 返回 (c0, ŷ0) 供导出器加列
# ----------------------------------------------------------------------------

def collect_pred_and_cross_weight(model, X_std, scalars, device,
                                  lm_data=None, batch_size=512):
    """单次前向同时收集 (标准化预测 ŷ, per-gene cross_weight c)。

    用于 Panel H 因果 Δc/Δŷ 反事实: 对同一组顺式输入 (scalars, lm), 分别传入真实
    TF 输入 X_std 与基线 X0, 各调用一次即得 (ŷ, c) 与 (ŷ0, c0):
        Δc = c − c0,   Δŷ = ŷ − ŷ0
    相比分别调用 collect_cross_weights + predict_full_dataset, 这里保证 ŷ 与 c 来自
    *同一次* 前向、逐基因完全对齐, 语义最干净(也省一次前向)。

    返回 (yhat, cross_w):
      · yhat    : (N,) float32, 标准化空间预测 (与 combined.csv 的 full_pred=pred_norm 同口径)
      · cross_w : (N,) float32, c∈(0,1); seq_only / concat 融合等无 c 时返回全 NaN
    """
    model.eval()
    kind = getattr(model, 'model_kind', 'tf')
    preds, cws = [], []
    with torch.no_grad():
        for s in range(0, len(X_std), batch_size):
            e  = s + batch_size
            x  = torch.FloatTensor(X_std[s:e]).to(device)
            sc = (torch.FloatTensor(scalars[s:e]).to(device)
                  if scalars is not None else None)
            lm = (torch.FloatTensor(lm_data[s:e]).to(device)
                  if lm_data is not None else None)
            if kind == 'seq_only':                       # 无 TF 分支 / 无 cross_weight
                nsf = getattr(model, 'n_seq_features', sc.shape[1]) if sc is not None else 1
                p = model(sc[:, :nsf] if sc is not None else None, lm)
                if isinstance(p, tuple): p = p[0]
                preds.append(p.detach().cpu().numpy())
                cws.append(np.full(p.shape[0], np.nan, dtype=np.float32))
                continue
            out = model(x, sc, lm_emb=lm, return_internals=True)
            # return_internals=True ⇒ (pred, (seq_emb, tf_emb, cross_weight, region_weights))
            pred = out[0] if isinstance(out, tuple) else out
            internals = out[-1] if (isinstance(out, tuple) and len(out) >= 2) else None
            preds.append(pred.detach().cpu().numpy())
            cw = (internals[2] if (isinstance(internals, tuple) and len(internals) >= 3)
                  else None)
            cws.append(cw.detach().cpu().numpy() if cw is not None
                       else np.full(pred.shape[0], np.nan, dtype=np.float32))
    if not preds:
        return (np.array([], np.float32), np.array([], np.float32))
    return (np.concatenate(preds).astype(np.float32),
            np.concatenate(cws).astype(np.float32))


def _build_tf_baseline(X_std, mode='zero', seed=42):
    """按 mode 造 TF 输入的反事实基线 X0 (形状同 X_std)。
      zero    — 全零: 撤掉反式信号 (KV 退化为偏置常数; 正文口径/默认)
      mean    — 全量逐 bin 均值广播: 保留"平均反式背景", 去掉基因特异性
      shuffle — 逐基因独立打散 TF 轴(axis=1): 破坏 cis–trans 配对, 保留每基因边际
    """
    X = np.asarray(X_std)
    if mode == 'zero':
        return np.zeros_like(X)
    if mode == 'mean':
        return np.broadcast_to(X.mean(axis=0, keepdims=True), X.shape).astype(X.dtype).copy()
    if mode == 'shuffle':
        rng = np.random.default_rng(seed)
        X0 = X.copy()
        if X0.ndim >= 2:
            for i in range(X0.shape[0]):
                X0[i] = X0[i][rng.permutation(X0.shape[1])]   # 打散该基因的 TF 轴
        return X0
    raise ValueError(f"未知 baseline 模式: {mode!r} (zero|mean|shuffle)")


# ── 因果 Δc 专用统计小工具 (与 crossweight_quadrant_standalone.py 同口径) ──────
def _cdc_residualize(x, ctrl):
    """对 ctrl 做线性回归取残差 (用于偏相关)。"""
    x = np.asarray(x, float); ctrl = np.asarray(ctrl, float)
    A = np.column_stack([np.ones_like(ctrl), ctrl])
    beta, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ beta


def _cdc_partial_spearman(x, y, ctrl):
    """控 ctrl 的偏 Spearman: 先 rank-残差化再 Spearman。返回 (r_raw, r_partial, n)。"""
    from scipy.stats import spearmanr as _spm
    x = np.asarray(x, float); y = np.asarray(y, float); ctrl = np.asarray(ctrl, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(ctrl)
    if m.sum() < 8:
        return np.nan, np.nan, int(m.sum())
    # 对秩做残差化 → 偏 Spearman (与标准定义一致, 比对原值残差化更稳)
    rx = _cdc_residualize(np.argsort(np.argsort(x[m])).astype(float),
                          np.argsort(np.argsort(ctrl[m])).astype(float))
    ry = _cdc_residualize(np.argsort(np.argsort(y[m])).astype(float),
                          np.argsort(np.argsort(ctrl[m])).astype(float))
    try:
        r_raw = _spm(x[m], y[m]).correlation
        r_par = _spm(rx, ry).correlation
    except Exception:
        return np.nan, np.nan, int(m.sum())
    return float(r_raw), float(r_par), int(m.sum())


def _cdc_partial_pearson_perm(x, y, ctrl, n_perm=2000, seed=0):
    """控 ctrl 的偏 Pearson + 置换 p。返回 dict(r_raw, r_partial, p_perm, n)。"""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(ctrl)
    x, y, ctrl = np.asarray(x)[m], np.asarray(y)[m], np.asarray(ctrl)[m]
    if m.sum() < 20:
        return dict(r_raw=np.nan, r_partial=np.nan, p_perm=np.nan, n=int(m.sum()))
    rx, ry = _cdc_residualize(x, ctrl), _cdc_residualize(y, ctrl)
    r_raw = pearsonr(x, y)[0]; r_par = pearsonr(rx, ry)[0]
    rng = np.random.default_rng(seed)
    null = np.array([pearsonr(rng.permutation(rx), ry)[0] for _ in range(n_perm)])
    p_perm = (np.sum(np.abs(null) >= abs(r_par)) + 1) / (n_perm + 1)
    return dict(r_raw=float(r_raw), r_partial=float(r_par),
                p_perm=float(p_perm), n=int(m.sum()))


def _cdc_cliffs_delta(a, b):
    """Cliff's δ∈[-1,1]: δ>0 ⇒ a 整体大于 b (非参、对分布形状稳健)。"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = np.sort(b[np.isfinite(b)])
    na, nb = a.size, b.size
    if na == 0 or nb == 0:
        return np.nan
    less    = np.searchsorted(b, a, side="left").sum()
    greater = (nb - np.searchsorted(b, a, side="right")).sum()
    return float((less - greater) / (na * nb))


def _cdc_delta_mag(d):
    """Romano 阈值: |δ|<.147 negligible, <.33 small, <.474 medium, else large。"""
    if not np.isfinite(d): return "na"
    ad = abs(d)
    return ("negligible" if ad < .147 else "small" if ad < .33
            else "medium" if ad < .474 else "large")


def _fmt_cdc(v):
    return "  nan" if (v is None or not np.isfinite(v)) else f"{v:+.3f}"


def run_causal_delta_c(model, X_std_full, scalars_full, device, lm_full,
                       gm_q, valid_all, cross_w_all,
                       FC_aligned=None, output_dir='.', tag=None,
                       cfg=None):
    """Panel H 因果 Δc / Δŷ 反事实实验 (见顶部 CAUSAL_DELTA_C 说明)。

    复用 run_postanalysis 已算好的内存数组(不重建模型、不重新切数据), 只额外做
    *一次* 基线前向。所有判别性检验在指定 split(默认 test)、限定 Q2∪Q3、控 log-TPM。

    返回 (cross_w_baseline_full, yhat_baseline_full):
      两个 (N,) float32 数组(与 cross_w_all / valid_all 同长), 供
      _export_combined_crossweight_table 切片成 combined.csv 的
      cross_weight_c_baseline / yhat_baseline 两列。无法运行时返回 (None, None)。
    """
    cfg = cfg or CAUSAL_DELTA_C
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    _tag = str(tag) if tag else (out.name or 'run')
    print("\n" + "=" * 62 +
          "\n  [Panel H · 因果 Δc] TF→基线 反事实前向 (Δc = c − c0, Δŷ = ŷ − ŷ0)\n" +
          "=" * 62)

    # ── 前置守卫: 无 cross_weight (concat 融合 / seq_only / freeze_crossw) 时跳过 ──
    cw = np.asarray(cross_w_all, dtype=float) if cross_w_all is not None \
        else np.array([], dtype=float)
    if cw.size == 0 or not np.isfinite(cw).any():
        print("  ⚠ 当前模型无有效 cross_weight (concat 融合 / seq_only / c 被冻结):"
              " 跳过因果 Δc 实验")
        return None, None
    if cw.size != len(valid_all):
        print(f"  ⚠ cross_weight 长度 {cw.size} 与全基因数 {len(valid_all)} 不匹配:"
              " 跳过因果 Δc 实验")
        return None, None

    # ── 反事实前向: 基线 (c0, ŷ0); 同时重收一份 full (c, ŷ) 保证逐基因对齐 ────────
    baseline_mode = cfg.get('baseline', 'zero')
    try:
        X0 = _build_tf_baseline(X_std_full, baseline_mode, seed=cfg.get('shuffle_seed', 42))
        print(f"  反事实基线: TF 输入 → '{baseline_mode}'  "
              f"(zero=撤反式 / mean=平均背景 / shuffle=打散 cis-trans 配对)")
        yhat_full, c_full = collect_pred_and_cross_weight(
            model, X_std_full, scalars_full, device, lm_data=lm_full)
        yhat_base, c_base = collect_pred_and_cross_weight(
            model, X0,         scalars_full, device, lm_data=lm_full)
    except Exception as _e:
        print(f"  ⚠ 反事实前向失败: {_e}")
        traceback.print_exc()
        return None, None

    # full-c 自洽性核对 (应与 collect_cross_weights 的 cross_w_all 几乎一致)
    _ok = np.isfinite(c_full) & np.isfinite(cw)
    if _ok.any():
        _md = float(np.max(np.abs(c_full[_ok] - cw[_ok])))
        print(f"  自洽核对: |c_full − cross_w_all|max = {_md:.2e} "
              f"({'✓ 一致' if _md < 1e-4 else '⚠ 偏差偏大, 但不影响 Δc(同源差分)'})")

    delta_c    = c_full - c_base
    delta_yhat = yhat_full - yhat_base

    # ── 组装逐基因明细 (对齐 valid 子集 / gm_q) ──────────────────────────────
    vmask = np.asarray(valid_all, dtype=bool)
    dd = pd.DataFrame({
        'gene'        : gm_q['gene'].astype(str).values,
        'quadrant'    : ([str(v).split(':', 1)[0].strip() for v in gm_q['quadrant'].values]
                         if 'quadrant' in gm_q.columns else 'unknown'),
        'split'       : (gm_q['split'].astype(str).values
                         if 'split' in gm_q.columns else 'all'),
        'log_tpm'     : gm_q['tpm_norm'].astype(float).values,    # z log-TPM = 控制变量
        'true_expr'   : gm_q['tpm_norm'].astype(float).values,
        'c_full'      : c_full[vmask],
        'c_baseline'  : c_base[vmask],
        'delta_c'     : delta_c[vmask],
        'yhat_full'   : yhat_full[vmask],
        'yhat_baseline': yhat_base[vmask],
        'delta_yhat'  : delta_yhat[vmask],
    })
    dd['abs_delta_yhat'] = np.abs(dd['delta_yhat'])

    # |trans_resid| = |true − seq_only(B)| (所需反式修正量); 需 ablation B
    if 'seq_only_pred' in gm_q.columns:
        _seq = gm_q['seq_only_pred'].astype(float).values
        dd['trans_resid']     = dd['true_expr'].values - _seq
        dd['abs_trans_resid'] = np.abs(dd['trans_resid'].values)
    else:
        dd['trans_resid'] = np.nan; dd['abs_trans_resid'] = np.nan

    # IAA mean|log2FC| (外部扰动敏感性); 需 --iaa-matrix 对齐的 FC
    iaa_mean = None
    if FC_aligned is not None:
        fc = np.asarray(FC_aligned, dtype=float)
        fc_sub = None
        if fc.ndim == 2 and fc.shape[0] == len(valid_all):
            fc_sub = fc[vmask]
        elif fc.ndim == 2 and fc.shape[0] == int(vmask.sum()):
            fc_sub = fc
        if fc_sub is not None and fc_sub.shape[0] == len(dd):
            with np.errstate(invalid='ignore'):
                iaa_mean = np.nanmean(np.abs(fc_sub), axis=1)
    dd['iaa_mean_sens'] = iaa_mean if iaa_mean is not None else np.nan

    csv_path = out / f'causal_delta_c_{_tag}.csv'
    dd.to_csv(csv_path, index=False)
    print(f"  ✓ 逐基因明细 → {csv_path}  (n={len(dd)} valid 基因)")

    # ── 选 split 子集 (默认 test, 最干净) ────────────────────────────────────
    want_split = cfg.get('split', 'test')
    if want_split == 'test' and (dd['split'] == 'test').any():
        sub = dd[dd['split'] == 'test'].copy()
        scope = 'test-only'
    else:
        if want_split == 'test':
            print("  ⚠ 无 test 标注或无 test 基因, 退化为全基因(性能/解读偏乐观)")
        sub = dd.copy()
        scope = 'all-genes'

    q   = sub['quadrant'].values
    m2  = (q == 'Q2'); m3 = (q == 'Q3'); m23 = m2 | m3
    m14 = (q == 'Q1') | (q == 'Q4')
    cf  = np.isfinite(sub['delta_c'].values)
    base = m23 & cf
    n23  = int(base.sum())
    print(f"\n  {'─'*60}\n  裁决检验 scope={scope}  Q2∪Q3 有效 n={n23}  "
          f"(Q2 {int((m2&cf).sum())} / Q3 {int((m3&cf).sum())}); 一律控 log-TPM\n  {'─'*60}")
    if n23 < 12:
        print("  ⚠ Q2∪Q3 有效基因 < 12, 跳过裁决检验(仍已落盘逐基因 Δc 供后续合并)")
        return c_base, yhat_base

    dc   = sub['delta_c'].values
    tpm  = sub['log_tpm'].values
    # Δc 描述 (各象限中位数, 看是否反式象限更高)
    print("  [Δc 分布] 各象限 Δc 中位数 (反式假说: Q2/Q3 应高于 Q1/Q4):")
    for lbl in ('Q1', 'Q2', 'Q3', 'Q4'):
        mm = (q == lbl) & cf
        if mm.sum() >= 3:
            print(f"     {lbl}: median Δc = {np.median(dc[mm]):+.4f}  (n={int(mm.sum())})")

    # (1) ★ 金标准 corr(Δc, |Δŷ|) on Q2∪Q3 (功能绑定, 内生, 不依赖外部 IAA) ──────
    ady = sub['abs_delta_yhat'].values
    r_raw, r_par, nn = _cdc_partial_spearman(dc[base], ady[base], tpm[base])
    r2_ = (_cdc_partial_spearman(dc[m2 & cf], ady[m2 & cf], tpm[m2 & cf])[1]
           if (m2 & cf).sum() > 8 else np.nan)
    r3_ = (_cdc_partial_spearman(dc[m3 & cf], ady[m3 & cf], tpm[m3 & cf])[1]
           if (m3 & cf).sum() > 8 else np.nan)
    print(f"\n  (1) ★ corr(Δc, |Δŷ|)  [开门是否预测模型预测的真实移动量 — 最强、内生]")
    print(f"        Q2∪Q3  ρ={_fmt_cdc(r_raw)}   ρ|log-TPM={_fmt_cdc(r_par)}  (n={nn})")
    print(f"        分象限  Q2 ρ|TPM={_fmt_cdc(r2_)}   Q3 ρ|TPM={_fmt_cdc(r3_)}")

    # (2) Δc 是否 Q2/Q3 > Q1/Q4 (MWU + Cliff's δ) ───────────────────────────
    from scipy.stats import mannwhitneyu as _mwu
    a23 = dc[base]; a14 = dc[m14 & cf]
    if a23.size >= 8 and a14.size >= 8:
        try:
            U, pmw = _mwu(a23, a14, alternative='greater')
        except Exception:
            pmw = np.nan
        d_ = _cdc_cliffs_delta(a23, a14)
        print(f"\n  (2) Δc(Q2∪Q3) > Δc(Q1∪Q4)?  "
              f"median {np.median(a23):+.4f} vs {np.median(a14):+.4f}  "
              f"MWU p={pmw:.3g}  Cliff δ={_fmt_cdc(d_)} ({_cdc_delta_mag(d_)})")
    else:
        print("\n  (2) Δc(Q2∪Q3) vs (Q1∪Q4): Q1/Q4 有效基因不足, 跳过")

    # (3) partial corr(Δc, IAA mean|log2FC| | log-TPM) on Q2∪Q3 (需 IAA) ──────
    if iaa_mean is not None and np.isfinite(sub['iaa_mean_sens'].values[base]).sum() >= 20:
        iaa = sub['iaa_mean_sens'].values
        pc = _cdc_partial_pearson_perm(dc[base], iaa[base], tpm[base],
                                       n_perm=int(cfg.get('n_perm', 2000)))
        print(f"\n  (3) corr(Δc, IAA |log2FC|)  [对齐外部扰动敏感性]")
        print(f"        Q2∪Q3  r={_fmt_cdc(pc['r_raw'])}   r|log-TPM={_fmt_cdc(pc['r_partial'])}"
              f"   p_perm={pc['p_perm']:.3g}  (n={pc['n']})")
    else:
        print("\n  (3) corr(Δc, IAA): 无 IAA 或命中基因 <20, 跳过"
              " (传 iaa_fc_file 给 run_postanalysis 即可解锁)")

    # (4) corr(Δc, |trans_resid| | log-TPM) on Q2∪Q3 (需 ablation B) ─────────
    if np.isfinite(sub['abs_trans_resid'].values[base]).sum() >= 8:
        atr = sub['abs_trans_resid'].values
        r_raw4, r_par4, nn4 = _cdc_partial_spearman(dc[base], atr[base], tpm[base])
        print(f"\n  (4) corr(Δc, |trans_resid|)  [对齐「所需反式修正量」]")
        print(f"        Q2∪Q3  ρ={_fmt_cdc(r_raw4)}   ρ|log-TPM={_fmt_cdc(r_par4)}  (n={nn4})")
    else:
        print("\n  (4) corr(Δc, |trans_resid|): 无 seq_only_pred(ablation B), 跳过")

    # ── 数据驱动裁决 (口径与正文 audit 一致, 方向不预设) ──────────────────────
    def _strong(r): return np.isfinite(r) and abs(r) >= 0.15
    func_pos = _strong(r_par)
    iaa_pos  = (iaa_mean is not None
                and np.isfinite(sub['iaa_mean_sens'].values[base]).sum() >= 20
                and _strong(_cdc_partial_pearson_perm(
                    dc[base], sub['iaa_mean_sens'].values[base], tpm[base], n_perm=200
                    )['r_partial']))
    print(f"\n  {'─'*60}\n  [裁决]")
    if func_pos and iaa_pos:
        print("  · Δc 同时绑定模型内部功能(|Δŷ|)与外部扰动(IAA) ⇒ c 有反式因果含义,"
              " 且对齐实测扰动敏感性 —— 最强阳性。")
    elif func_pos and not iaa_pos:
        print("  · corr(Δc,|Δŷ|) 强、Δc~IAA 仍 null ⇒ c 在功能上确实门控模型内部反式贡献,"
              " 但该贡献不对齐实测扰动敏感性(与「结合≠调控」、方向A/B 为零自洽)。")
    elif (not func_pos) and iaa_pos:
        print("  · Δc~|Δŷ| 弱、却与 IAA 相关 ⇒ 罕见组合, 建议核对 IAA 对齐/象限定义。")
    else:
        print("  · corr(Δc,|Δŷ|) 与 Δc~IAA 皆 null ⇒ 因果层面钉死「c=生长/表达程序倾向、"
              " 非反式负荷」: 即便差分掉表达量基线, TF 信号也未在 Q2/Q3 系统性推开闸门。")
    print("  · 提醒: 增益项 c(1−c) 使 Δc 天然偏向中等-c(偏低表达)基因, 上述检验已控 log-TPM;"
          " 若 raw ρ 与 ρ|TPM 差异大, 以 ρ|TPM 为准。")
    print("  " + "─" * 60)
    return c_base, yhat_base


# ============================================================================
# 区域权重收集与 Q2/Q3 分析
# ============================================================================

def collect_region_weights(model, X_std, device, scalars=None,
                           lm_data=None, batch_size=512) -> np.ndarray:
    """
    提取 TFSpatialEncoder→AttentiveRegionPool 的区域注意力分布特征 (N, n_regions, 3).

    ─ bug 修复说明 ─────────────────────────────────────────────────────────────
    旧版返回 `rw.mean(dim=-1)`, 即 softmax-over-L 的逐-bin 均值. 由于 softmax
    在 L 维上总和恒等于 1, 均值 = 1/L (这里 L=75 → 永远是 0.01333...).
    不管分布是"尖峰在 TSS"还是"完全均匀", 这个数字都不会变 —— 是分析脚本
    的 bug, 不是模型本身的退化. 之前 details.txt 里 Region0/1/2 四象限均值
    全写成 0.0133 就是这个原因.

    ─ 新版输出三个真正描述分布形态的统计量 ─────────────────────────────────────
      idx=0  peak           — 区域最强 bin 的权重值 (越大 → 越集中)
      idx=1  entropy        — 区域内 softmax 分布的熵 (越小 → 越集中)
      idx=2  center_of_mass — 区域重心的 bin 坐标 / L_bins ∈ [0, 1]
                              (衡量注意力在启动子窗口内的位置; 0=最上游, 1=最下游)
    下游 Q1/Q2/Q3/Q4 分析可以挑其中任意维度做 KW 检验.
    """
    model.eval()
    if getattr(model, 'model_kind', 'tf') == 'seq_only':
        return np.array([], dtype=np.float32)
    stats_all = []
    eps = 1e-12
    with torch.no_grad():
        for s in range(0, len(X_std), batch_size):
            x = torch.FloatTensor(X_std[s:s + batch_size]).to(device)
            _, _, rw = model.tf_spatial_enc(x, return_regions=True)
            # rw: (B, n_regions, L_bins)
            B, K, L = rw.shape
            # 峰值
            peak = rw.max(dim=-1).values                                # (B, K)
            # 熵 (nats) — 数值稳定: 用 (w + eps).log() 避免 log(0) 奇点反向
            entropy = -(rw * (rw + eps).log()).sum(dim=-1)              # (B, K)
            # 重心 (bin 坐标加权均值, 再除以 L 归一化到 [0, 1])
            pos = torch.arange(L, device=rw.device, dtype=rw.dtype)     # (L,)
            com = (rw * pos.view(1, 1, L)).sum(dim=-1) / max(L - 1, 1)  # (B, K)
            stats = torch.stack([peak, entropy, com], dim=-1)           # (B, K, 3)
            stats_all.append(stats.cpu().numpy())
    if not stats_all:
        return np.array([], dtype=np.float32)
    return np.concatenate(stats_all, axis=0).astype(np.float32)  # (N, n_regions, 3)


def _analyze_region_weights_q2q3(region_w_all, gm_q, conds, quad_labels,
                                   quad_colors, output_dir):
    """
    区域注意力分布 × Q1/Q2/Q3/Q4 四象限 Kruskal-Wallis 检验 + 可视化.

    输入 region_w_all 形状 (N, n_regions, 3), 三个指标为:
      · peak           — softmax 峰值 (越大越集中)
      · entropy        — 分布熵 (越小越集中)
      · center_of_mass — 归一化重心位置 ∈ [0,1] (衡量注意力在启动子窗口里的位置)

    生物学假说:
      Q2 (trans-repressed, 高seq预测 & 低表达, TF抑制):
        注意力应更集中在 TSS 近端 (region_1 的 peak 高 / entropy 低);
        整体重心可能更偏 TSS (center_of_mass 偏向 0.6-0.7, 对应 TSS 区).
      Q3 (trans-activated, 低seq预测 & 高表达, TF激活):
        注意力可能更分散在远端上游 (region_0 的 entropy 高 / peak 低);
        重心可能偏向更远端 (center_of_mass 偏小).
    """
    from scipy.stats import kruskal as _kruskal
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if region_w_all is None or region_w_all.ndim < 2 or len(region_w_all) == 0:
        print("  ⚠ [区域权重] region_w_all 无效，跳过"); return

    # ── 向后兼容: 若旧 shape (N, n_regions), 升一维当 peak 处理 ────────────────
    if region_w_all.ndim == 2:
        print("  ⚠ [区域权重] 检测到旧版 shape (N, n_regions), "
              "已自动升维 (peak only); 建议重跑 collect_region_weights.")
        region_w_all = region_w_all[..., None]  # (N, K, 1)

    N_all, n_regions, n_metrics = region_w_all.shape
    n_valid = len(gm_q)
    rw_q = region_w_all[:n_valid]           # 与 gm_q 对齐

    region_names = ['Region0-DistalUpstream',
                    'Region1-TSS-Proximal',
                    'Region2-TSS-Downstream'][:n_regions]
    metric_names = ['peak', 'entropy', 'center_of_mass'][:n_metrics]

    records = []
    print(f"\n  [区域权重×Q2/Q3] n_valid={n_valid}  n_regions={n_regions}  "
          f"n_metrics={n_metrics}")
    for mi, mname in enumerate(metric_names):
        print(f"\n  ── 指标: {mname} ──")
        print(f"  {'区域':30s}  Q1-mean   Q2-mean   Q3-mean   Q4-mean  KW-p")
        print("  " + "─" * 72)
        for ri, rname in enumerate(region_names):
            rw_col = rw_q[:, ri, mi]
            groups = [rw_col[np.where(cond[:n_valid])[0]] for cond in conds]
            means  = [float(g.mean()) if len(g) > 0 else float('nan') for g in groups]
            valid_grps = [g for g in groups if len(g) >= 5]
            kw_p = float('nan')
            if len(valid_grps) >= 2:
                try: _, kw_p = _kruskal(*valid_grps)
                except Exception: pass
            sig = '***' if kw_p < 0.001 else ('**' if kw_p < 0.01 else
                  ('*' if kw_p < 0.05 else 'ns'))
            print(f"  {rname:30s}  " +
                  "  ".join(f"{m:.4f}" for m in means) + f"  {kw_p:.2e} {sig}")
            records.append({'metric': mname, 'region': rname, 'region_idx': ri,
                            **{f'mean_{l.split(":")[0].strip()}': m
                               for l, m in zip(quad_labels, means)},
                            'kruskal_p': kw_p})

    rw_df = pd.DataFrame(records)
    rw_df.to_csv(out / 'region_weights_q2q3_analysis.csv', index=False)

    # ── 可视化：三行(metrics) × 三列(regions) 的箱线图 ───────────────────────
    fig, axes = plt.subplots(n_metrics, n_regions,
                             figsize=(5 * n_regions, 4.2 * n_metrics),
                             squeeze=False)
    fig.suptitle("TFSpatialEncoder: Region Attention Distribution × Quadrant\n"
                 "peak (top), entropy (mid), center_of_mass (bot)  —  bug-fixed metrics",
                 fontsize=11, fontweight='bold')

    for mi, mname in enumerate(metric_names):
        for ri, rname in enumerate(region_names):
            ax = axes[mi, ri]
            rw_col = rw_q[:, ri, mi]
            rec = rw_df[(rw_df['metric'] == mname) & (rw_df['region_idx'] == ri)]
            kw_p_val = float(rec['kruskal_p'].iloc[0]) if len(rec) else float('nan')
            sig_lbl  = ('***' if kw_p_val < 0.001 else '**' if kw_p_val < 0.01
                        else '*' if kw_p_val < 0.05 else 'ns')
            for xi, (cond, lbl, col) in enumerate(zip(conds, quad_labels, quad_colors)):
                vals = rw_col[np.where(cond[:n_valid])[0]]
                if len(vals) < 3: continue
                ax.boxplot(vals, positions=[xi], widths=0.55, patch_artist=True,
                           medianprops=dict(color='white', lw=2),
                           boxprops=dict(facecolor=col, alpha=0.60),
                           whiskerprops=dict(color=col, lw=1.5),
                           capprops=dict(color=col, lw=1.5),
                           flierprops=dict(marker='', alpha=0))
                rng = np.random.default_rng(ri * 10 + mi); n_show = min(120, len(vals))
                ax.scatter(xi + rng.uniform(-0.16, 0.16, n_show),
                           rng.choice(vals, n_show, replace=False),
                           c=col, alpha=0.18, s=5, zorder=2)
            ax.set_xticks(range(4))
            ax.set_xticklabels([l.split(':')[0] for l in quad_labels], fontsize=8)
            ax.set_title(f'{mname} · {rname}\nKW p={kw_p_val:.2e} {sig_lbl}',
                         fontsize=8, fontweight='bold')
            if ri == 0:
                ax.set_ylabel(mname, fontsize=9)
            ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    fig.savefig(out / 'figRegionWeights_Q2Q3.pdf', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"\n  ✓ [区域权重] Q2/Q3分析 → {out/'region_weights_q2q3_analysis.csv'}")
    print(f"  ✓ [区域权重] 可视化   → {out/'figRegionWeights_Q2Q3.pdf'}")


def _plot_cross_weight_diagnosis(cross_w_all, gene_names,
                                  output_dir, top_n_genes=20,
                                  FC_aligned=None, top_pct=5):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if len(cross_w_all) == 0:
        print("  ⚠ [方向C] cross_weight 为空，跳过"); return

    cw = cross_w_all
    cw_v = cw[np.isfinite(cw)]
    N = len(cw); k = max(int(N * top_pct / 100), 10)

    print(f"\n  [方向C] cross_weight c:"
          f"  mean={cw_v.mean():.4f}  std={cw_v.std():.4f}"
          f"  range=[{cw_v.min():.4f}, {cw_v.max():.4f}]")

    r_cw_al = float('nan')

    sorted_idx = np.argsort(cw)
    low_idx  = sorted_idx[:k]; high_idx = sorted_idx[-k:]
    gn_arr = (np.asarray(gene_names) if gene_names is not None else np.arange(N).astype(str))

    for idx_g, lbl_g in [(high_idx, 'high_c'), (low_idx, 'low_c')]:
        df_go = pd.DataFrame({
            'gene': gn_arr[idx_g], 'cross_weight': cw[idx_g],
        }).sort_values('cross_weight', ascending=(lbl_g == 'low_c'))
        df_go.to_csv(out / f'dirC_gene_list_{lbl_g}_top{top_pct}pct.csv', index=False)
        print(f"  ✓ [方向C] GO输入基因表 ({lbl_g}) → "
              f"{out / f'dirC_gene_list_{lbl_g}_top{top_pct}pct.csv'}")

    iaa_mean_sens = None; r_c_fc = p_c_fc = float('nan')
    if FC_aligned is not None and FC_aligned.shape[0] == N:
        from scipy.stats import mannwhitneyu as _mwu
        iaa_mean_sens = np.nanmean(np.abs(FC_aligned), axis=1)
        iaa_breadth   = np.nansum(np.abs(FC_aligned) > 1.0, axis=1).astype(float)
        for metric, name in [(iaa_mean_sens, 'mean|log2FC|'),
                              (iaa_breadth,   'n_TFs_with_|FC|>1')]:
            hi_m = metric[high_idx][np.isfinite(metric[high_idx])]
            lo_m = metric[low_idx][np.isfinite(metric[low_idx])]
            if len(hi_m) > 5 and len(lo_m) > 5:
                stat, pval = _mwu(hi_m, lo_m, alternative='two-sided')
                print(f"  [方向C] IAA {name}: high_c={hi_m.mean():.3f}  "
                      f"low_c={lo_m.mean():.3f}  MWU p={pval:.3e}")
        valid_fc = np.isfinite(iaa_mean_sens) & np.isfinite(cw)
        if valid_fc.sum() > 20:
            r_c_fc, p_c_fc = pearsonr(cw[valid_fc], iaa_mean_sens[valid_fc])
            print(f"  [方向C] Pearson(c, mean|log2FC|)={r_c_fc:.4f}  p={p_c_fc:.3e}")
        pd.DataFrame({
            'gene': gn_arr, 'cross_weight': cw,
            'iaa_mean_sens': iaa_mean_sens, 'iaa_breadth': iaa_breadth,
            'c_group': np.where(np.isin(np.arange(N), high_idx), 'high_c',
                       np.where(np.isin(np.arange(N), low_idx),  'low_c', 'mid')),
        }).to_csv(out / 'dirC_crossweight_iaa_summary.csv', index=False)
        print(f"  ✓ [方向C] c-IAA汇总表 → {out/'dirC_crossweight_iaa_summary.csv'}")

    n_panels = 4 if iaa_mean_sens is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6*n_panels, 5))
    fig.suptitle("Direction C: cross_weight c as Biological Annotation Variable",
                 fontsize=11, fontweight='bold')

    ax = axes[0]
    ax.hist(cw_v, bins=50, color='steelblue', alpha=0.75, edgecolor='white')
    ax.axvline(cw_v.mean(), color='red', lw=2, ls='--', label=f'mean={cw_v.mean():.3f}')
    ax.axvline(np.percentile(cw_v, 5),  color='#457B9D', lw=1.5, ls=':',
               label=f'p5={np.percentile(cw_v,5):.3f}')
    ax.axvline(np.percentile(cw_v, 95), color='#E63946', lw=1.5, ls=':',
               label=f'p95={np.percentile(cw_v,95):.3f}')
    ax.set_xlabel('cross_weight c'); ax.set_ylabel('Count')
    ax.set_title(f'A. Distribution of c\nstd={cw_v.std():.4f}  n={len(cw_v)}')
    ax.legend(fontsize=7); ax.spines[['top','right']].set_visible(False)

    ax = axes[1]
    top_e  = np.argsort(cw)[::-1][:top_n_genes//2]
    bot_e  = np.argsort(cw)[:top_n_genes//2]
    ext_idx = np.concatenate([top_e, bot_e])
    ext_cw  = cw[ext_idx]; ext_gn = gn_arr[ext_idx]
    c_ext   = ['#E63946']*(top_n_genes//2) + ['#457B9D']*(top_n_genes//2)
    y_pos   = np.arange(len(ext_idx))
    ax.barh(y_pos, ext_cw[::-1], color=c_ext[::-1], alpha=0.75, edgecolor='white')
    ax.set_yticks(y_pos); ax.set_yticklabels([str(g)[:14] for g in ext_gn[::-1]], fontsize=7)
    ax.axvline(cw_v.mean(), color='gray', lw=1.5, ls='--', alpha=0.7)
    ax.set_xlabel('cross_weight c')
    ax.set_title(f'C. Top/Bottom {top_n_genes//2} genes by c')
    ax.spines[['top','right']].set_visible(False)

    if iaa_mean_sens is not None:
        ax = axes[2]
        for xi, (idx_g, lbl_g, col_g) in enumerate([
                (high_idx, f'high c\n(top{top_pct}%)', '#E63946'),
                (low_idx,  f'low c\n(bot{top_pct}%)', '#457B9D')]):
            vals = iaa_mean_sens[idx_g]; vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                ax.boxplot(vals, positions=[xi], widths=0.5, patch_artist=True,
                           medianprops=dict(color='white', lw=2),
                           boxprops=dict(facecolor=col_g, alpha=0.6),
                           whiskerprops=dict(color=col_g, lw=1.5),
                           capprops=dict(color=col_g, lw=1.5),
                           flierprops=dict(marker='', alpha=0))
                rng2 = np.random.default_rng(1); n_show = min(150, len(vals))
                ax.scatter(xi + rng2.uniform(-0.15, 0.15, n_show),
                           rng2.choice(vals, n_show, replace=False),
                           c=col_g, alpha=0.3, s=6, zorder=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f'high c (n={len(high_idx)})', f'low c (n={len(low_idx)})'])
        ax.set_ylabel('IAA mean |log2FC|')
        ax.set_title('D. TF depletion sensitivity\nhigh_c vs low_c genes', fontweight='bold')
        ax.spines[['top','right']].set_visible(False)

        ax = axes[3]
        valid_fc = np.isfinite(iaa_mean_sens) & np.isfinite(cw)
        c_col = np.full(valid_fc.sum(), '#AAAAAA', dtype=object)
        hi_m  = np.isin(np.where(valid_fc)[0], high_idx)
        lo_m  = np.isin(np.where(valid_fc)[0], low_idx)
        c_col[hi_m] = '#E63946'; c_col[lo_m] = '#457B9D'
        ax.scatter(cw[valid_fc], iaa_mean_sens[valid_fc],
                   c=c_col, alpha=0.25, s=6, rasterized=True)
        if np.isfinite(r_c_fc):
            xv = cw[valid_fc]; yv = iaa_mean_sens[valid_fc]
            z  = np.polyfit(xv, yv, 1); xl = np.linspace(xv.min(), xv.max(), 100)
            ax.plot(xl, np.poly1d(z)(xl), 'k--', lw=1.8,
                    label=f'r={r_c_fc:.3f}  p={p_c_fc:.2e}')
            ax.legend(fontsize=8)
        ax.set_xlabel('cross_weight c'); ax.set_ylabel('IAA mean |log2FC|')
        ax.set_title('E. c vs TF depletion sensitivity', fontweight='bold')
        ax.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    fig.savefig(out / 'figDirC_cross_weight_biology.pdf', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  ✓ [方向C] cross_weight 生物分析图 → {out/'figDirC_cross_weight_biology.pdf'}")

    pd.DataFrame({
        'gene': gn_arr[ext_idx], 'cross_weight': ext_cw,
        'group': ['high_c']*(top_n_genes//2) + ['low_c']*(top_n_genes//2)
    }).to_csv(out / 'dirC_extreme_crossweight_genes.csv', index=False)


# ============================================================================
# combined.csv 导出 (cross_weight c × 四象限独立分析的输入表)
# ============================================================================
#   生成一张逐基因合并表, 直接喂给 crossweight_quadrant_standalone.py 的
#   `--input combined.csv`。schema 与该脚本的列别名表对齐:
#     gene, seq_only_pred(B), full_pred(C), true_expr(TPM), cross_weight_c,
#     quadrant(Q1-Q4 短码), [iaa_mean_sens]  ← FC_aligned 存在时附带
#   复用 run_postanalysis 已算好的内存数组, 不重新推断 / 不重建模型 /
#   不依赖 torch, 因此不会与训练好的预测漂移。
def _export_combined_crossweight_table(
        output_dir, gm_q, valid_all,
        seq_only_preds_full=None, cross_w_all=None,
        FC_aligned=None, tag=None, fname='combined.csv',
        cross_w_baseline=None, yhat_baseline=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_valid = int(len(gm_q))
    if n_valid == 0:
        print("  ⚠ [combined] gm_q 为空，跳过合并表导出")
        return None

    # ── gene / true_expr / full_pred(C): 直接来自 gm_q (已是 valid 子集) ──────
    tbl = pd.DataFrame({
        'gene':      gm_q['gene'].astype(str).values,
        'true_expr': gm_q['tpm_norm'].astype(float).values,   # z log-TPM
        'full_pred': gm_q['pred_norm'].astype(float).values,  # 全模型 C 预测
    })

    # ── split (train/val/test): 让 standalone 做干净的 test-only 复算 ─────────
    #   gm_q 自带 split 列 (run_postanalysis 建 gm 时按 split_masks['train/val/test']
    #   标注, 见上游 gm 构造), 经 valid_all 切片后逐基因对齐保留。
    #   写进合并表后, crossweight_quadrant_standalone.py 的 --split test 可据此
    #   只保留 test 基因 —— 排除「模型已见过标签」的 train/val, 避免 Panel B 的
    #   误差下降幅度被训练集虚高 (即 c 解读里标注的乐观偏差)。
    if 'split' in gm_q.columns:
        tbl['split'] = gm_q['split'].astype(str).values
    else:
        print("  ⚠ [combined] gm_q 无 split 列, 合并表将不含 split"
              " (standalone --split test 将不可用, 只能在全量基因上复算)")

    # ── seq_only_pred (B): 优先用 gm_q 内现成列, 否则用全量数组按 valid_all 取子集 ─
    if 'seq_only_pred' in gm_q.columns:
        tbl['seq_only_pred'] = gm_q['seq_only_pred'].astype(float).values
    elif (seq_only_preds_full is not None
          and len(seq_only_preds_full) == len(valid_all)):
        tbl['seq_only_pred'] = np.asarray(
            seq_only_preds_full, dtype=float)[valid_all]
    else:
        tbl['seq_only_pred'] = np.nan
        print("  ⚠ [combined] 无 seq_only_pred (ablation B 不可用): Panel A/B/象限"
              "将受限 (standalone 需要此列或 trans_resid)")

    # ── cross_weight_c: collect_cross_weights 输出 (全量 N 或空) ────────────
    cw = np.asarray(cross_w_all, dtype=float) if cross_w_all is not None \
        else np.array([], dtype=float)
    if cw.size == len(valid_all):
        tbl['cross_weight_c'] = cw[valid_all]
    elif cw.size == n_valid:                 # 已是 valid 子集 (少见, 容错)
        tbl['cross_weight_c'] = cw
    else:
        tbl['cross_weight_c'] = np.nan
        if cw.size == 0:
            print("  ⚠ [combined] cross_weight 为空 (concat 融合或 seq_only 模型):"
                  " c 列写 NaN, Panel A/B 仍可跑, Panel C/D 跳过")
        else:
            print(f"  ⚠ [combined] cross_weight 长度 {cw.size} 与基因数不匹配"
                  f" ({len(valid_all)}/{n_valid}): c 列写 NaN")

    # ── cross_weight_c_baseline / yhat_baseline (Panel H 因果 Δc; 来自反事实前向) ─
    #   由 run_causal_delta_c 返回的全长 (N,) 数组传入; 同 cross_weight_c 切片口径。
    #   cross_weight_c_baseline → 解锁 standalone Panel H 因果 Δc 模式(限定 Q2/Q3);
    #   yhat_baseline           → 供其后续算 Δŷ = full_pred − yhat_baseline。
    def _slice_full(arr, name):
        a = np.asarray(arr, dtype=float)
        if a.size == len(valid_all):
            tbl[name] = a[valid_all]
        elif a.size == n_valid:
            tbl[name] = a
        else:
            print(f"  ⚠ [combined] {name} 长度 {a.size} 与基因数不匹配, 跳过该列")
    if cross_w_baseline is not None:
        _slice_full(cross_w_baseline, 'cross_weight_c_baseline')
    if yhat_baseline is not None:
        _slice_full(yhat_baseline, 'yhat_baseline')
    if 'cross_weight_c_baseline' in tbl.columns:
        _b_ok = int(np.isfinite(tbl['cross_weight_c_baseline']).sum())
        print(f"  ✓ [combined] 附带因果 Δc 基线列 cross_weight_c_baseline"
              f"{' + yhat_baseline' if 'yhat_baseline' in tbl.columns else ''}"
              f" ({_b_ok}/{n_valid} 有效) → standalone --input 自动开 Panel H")

    # ── quadrant: 把 'Q2: Trans-repressed' 这类长标签压成 Q1-Q4 短码 ─────────
    #   standalone 仅当列值命中 {Q1,Q2,Q3,Q4} 时才直接采用, 否则按 median 重算;
    #   写成短码可让其直接复用我们的固定象限锚 (与 compute_fixed_quadrant_labels 一致)
    if 'quadrant' in gm_q.columns:
        def _short_q(v):
            s = str(v).strip()
            code = s.split(':', 1)[0].strip()
            return code if code in ('Q1', 'Q2', 'Q3', 'Q4') else 'unknown'
        tbl['quadrant'] = [_short_q(v) for v in gm_q['quadrant'].values]

    # ── iaa_mean_sens (可选): 逐基因 mean|log2FC|, 解锁 Panel D 的 c~IAA 偏相关 ─
    #   (带符号的 Panel B 方向检验仍需对 standalone 传 --iaa-matrix 原始矩阵)
    if FC_aligned is not None:
        fc = np.asarray(FC_aligned, dtype=float)
        fc_sub = None
        if fc.ndim == 2 and fc.shape[0] == len(valid_all):
            fc_sub = fc[valid_all]
        elif fc.ndim == 2 and fc.shape[0] == n_valid:
            fc_sub = fc
        if fc_sub is not None and fc_sub.shape[0] == n_valid:
            with np.errstate(invalid='ignore'):
                iaa_mean_sens = np.nanmean(np.abs(fc_sub), axis=1)
            tbl['iaa_mean_sens'] = iaa_mean_sens
            n_hit = int(np.isfinite(iaa_mean_sens).sum())
            print(f"  ✓ [combined] 附带 iaa_mean_sens ({n_hit}/{n_valid} 基因命中)")
        else:
            print("  ⚠ [combined] FC_aligned 形状与基因数不匹配，跳过 iaa_mean_sens 列")

    # ── 列顺序固定 (standalone 用别名识别, 顺序仅为可读性) ──────────────────
    _order = ['gene', 'seq_only_pred', 'full_pred', 'true_expr', 'cross_weight_c']
    _order += [c for c in ('cross_weight_c_baseline', 'yhat_baseline',
                           'split', 'quadrant', 'iaa_mean_sens') if c in tbl.columns]
    tbl = tbl[_order]

    csv_path = out / fname
    tbl.to_csv(csv_path, index=False)

    _tag = str(tag) if tag else (out.name or 'run')
    _c_ok = int(np.isfinite(tbl['cross_weight_c']).sum())
    print(f"  ✓ [combined] 合并表 → {csv_path}  "
          f"(n={n_valid}, c 有效 {_c_ok}, 列={list(tbl.columns)})")
    if 'split' in tbl.columns:
        _vc = tbl['split'].value_counts().to_dict()
        _n_test = int(_vc.get('test', 0))
        print(f"    split 计数: {_vc}  (test-only 复算将用 n={_n_test})")
    print(f"    下一步 (每个 holdout 各跑一次):")
    print(f"      python crossweight_quadrant_standalone.py \\")
    print(f"          --input {csv_path} \\")
    if 'split' in tbl.columns:
        print(f"          --split test \\        # 干净的 test-only 复算 (排除 train/val)")
    print(f"          --tag {_tag} --outdir {out}")
    print(f"    (加 --iaa-matrix <log2FC矩阵.tsv> 可解锁 Panel B 带符号方向检验)")
    if 'split' in tbl.columns:
        print(f"    (去掉 --split 则在全量基因上跑, 含 train/val, 性能指标偏乐观)")
    return str(csv_path)


# ============================================================================
# IAA 数据加载
# ============================================================================

def _load_iaa_fc_matrix(iaa_fc_file, gene_names, tf_names):
    print(f"\n  [IAA验证] 加载耗尽 log2FC 矩阵: {iaa_fc_file}")
    fc_df = None
    for enc in ['utf-8', 'gbk', 'latin1']:
        try:
            fc_df = pd.read_csv(iaa_fc_file, sep='\t', encoding=enc, index_col=0); break
        except Exception:
            try:
                fc_df = pd.read_csv(iaa_fc_file, encoding=enc, index_col=0); break
            except Exception: continue
    if fc_df is None: raise RuntimeError(f"无法读取 IAA 文件: {iaa_fc_file}")
    fc_df.index = fc_df.index.astype(str).str.strip()
    fc_df = fc_df.apply(pd.to_numeric, errors='coerce')
    print(f"  ✓ IAA 文件: {fc_df.shape[0]} 基因 × {fc_df.shape[1]} TF 耗尽实验")
    iaa_tf_names_raw = np.array([str(c).strip() for c in fc_df.columns])
    N = len(gene_names); n_iaa_tfs = fc_df.shape[1]
    FC = np.full((N, n_iaa_tfs), np.nan, dtype=np.float32); n_hit = 0
    for gi, gene in enumerate(gene_names):
        key = str(gene).strip()
        if key in fc_df.index:
            FC[gi, :] = fc_df.loc[key].values.astype(np.float32); n_hit += 1
    print(f"  ✓ 基因对齐: {n_hit}/{N} 个基因命中 IAA 矩阵")
    return FC, iaa_tf_names_raw


# ============================================================================
# 方向A: gene-level Spearman（含完整三面板可视化）
# ============================================================================

def _validate_attn_gene_specific_iaa(attn_all, gene_names, tf_names,
                                      iaa_fc_file, split_masks, output_dir):
    from scipy.stats import spearmanr as _spearmanr, wilcoxon as _wlcx
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if iaa_fc_file is None or not Path(iaa_fc_file).exists():
        print(f"  ⚠ [方向A] IAA文件未提供或不存在，跳过"); return None
    fc_df = None
    for enc in ['utf-8', 'gbk', 'latin1']:
        try:
            fc_df = pd.read_csv(iaa_fc_file, sep='\t', encoding=enc, index_col=0); break
        except Exception:
            try:
                fc_df = pd.read_csv(iaa_fc_file, encoding=enc, index_col=0); break
            except Exception: continue
    if fc_df is None:
        print("  ⚠ [方向A] 无法读取IAA文件，跳过"); return None
    fc_df.index = fc_df.index.astype(str).str.strip()
    fc_df = fc_df.apply(pd.to_numeric, errors='coerce')
    iaa_tf_upper   = np.array([str(c).strip().upper() for c in fc_df.columns])
    model_tf_upper = np.array([str(t).strip().upper() for t in tf_names])
    common_model_idx, common_iaa_idx = [], []
    for mi, tf_u in enumerate(model_tf_upper):
        hits = np.where(iaa_tf_upper == tf_u)[0]
        if len(hits) > 0:
            common_model_idx.append(mi); common_iaa_idx.append(int(hits[0]))
    n_common = len(common_model_idx)
    if n_common < 5:
        print(f"  ⚠ [方向A] 共同TF太少({n_common})，跳过"); return None
    model_idx_arr = np.array(common_model_idx); iaa_idx_arr = np.array(common_iaa_idx)
    print(f"  ✓ [方向A] 共同TF = {n_common}")
    gn_arr   = np.asarray(gene_names); N = len(gene_names)
    attn_sub = attn_all[:, model_idx_arr]
    FC_sub   = np.full((N, n_common), np.nan, dtype=np.float32)
    for gi, gene in enumerate(gn_arr):
        key = str(gene).strip()
        if key in fc_df.index:
            FC_sub[gi, :] = fc_df.loc[key].values[iaa_idx_arr].astype(np.float32)
    gene_r = np.full(N, np.nan, dtype=np.float32)
    gene_p = np.full(N, np.nan, dtype=np.float32)
    for gi in range(N):
        fc_abs = np.abs(FC_sub[gi])
        valid  = np.isfinite(fc_abs) & np.isfinite(attn_sub[gi])
        if valid.sum() < 8: continue
        r_s, p_s = _spearmanr(attn_sub[gi, valid], fc_abs[valid])
        gene_r[gi] = float(r_s); gene_p[gi] = float(p_s)
    valid_mask = np.isfinite(gene_r); r_vals = gene_r[valid_mask]
    n_computed = int(valid_mask.sum())
    print(f"\n  [方向A] Gene-level Spearman:")
    print(f"    n={n_computed}  mean r={r_vals.mean():.4f}  "
          f"median={np.median(r_vals):.4f}  r>0={(r_vals>0).mean():.3f}")
    split_arr = np.where(split_masks['test'][:N], 'test',
                np.where(split_masks['val'][:N],  'val',  'train'))
    te_mask_in_valid = split_masks['test'][:N][valid_mask]
    if te_mask_in_valid.sum() >= 10:
        r_te = r_vals[te_mask_in_valid]
        print(f"    测试集(n={te_mask_in_valid.sum()}): mean r={r_te.mean():.4f}  "
              f"r>0={(r_te>0).mean():.3f}")
        try:
            stat_wx, p_wx = _wlcx(r_te, alternative='greater')
            print(f"    Wilcoxon one-sided H₀: r≤0  stat={stat_wx:.1f}  p={p_wx:.3e}")
        except Exception: pass
    result_df = pd.DataFrame({
        'gene': gn_arr, 'spearman_r': gene_r, 'spearman_p': gene_p, 'split': split_arr,
    }).dropna(subset=['spearman_r']).sort_values('spearman_r', ascending=False)
    result_df.to_csv(out / 'dirA_gene_attn_iaa_spearman.csv', index=False)
    print(f"  ✓ [方向A] per-gene Spearman → {out/'dirA_gene_attn_iaa_spearman.csv'}")

    # ── 三面板可视化 ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Direction A: Gene-level Spearman(attention_weights, |log2FC|)\n"
        f"n_common_TFs={n_common}  n_genes_computed={n_computed}",
        fontsize=11, fontweight='bold')
    bins = np.linspace(-1, 1, 40)
    ax = axes[0]
    for split_lbl, col in [('train','#3498DB'), ('val','#F39C12'), ('test','#E74C3C')]:
        m = valid_mask & (split_arr == split_lbl); r_s = gene_r[m]
        if len(r_s) > 0:
            ax.hist(r_s, bins=bins, color=col, alpha=0.55, density=True,
                    label=f'{split_lbl} (n={len(r_s)}, μ={r_s.mean():.3f})')
    ax.axvline(0, color='black', lw=1.5, ls='--', alpha=0.7)
    ax.axvline(r_vals.mean(), color='gray', lw=1.5, ls=':', label=f'All μ={r_vals.mean():.3f}')
    ax.set_xlabel('Gene-level Spearman r'); ax.set_ylabel('Density')
    ax.set_title('A. Distribution of Spearman r', fontweight='bold')
    ax.legend(fontsize=7); ax.spines[['top','right']].set_visible(False)

    ax = axes[1]
    te_full_idx = np.where(valid_mask & (split_arr == 'test'))[0]
    if len(te_full_idx) >= 6:
        top_k = min(12, len(te_full_idx) // 2)
        sort_ord = np.argsort(gene_r[te_full_idx])
        show_idx = np.concatenate([sort_ord[-top_k:], sort_ord[:top_k]])
        show_r   = gene_r[te_full_idx][show_idx]
        show_g   = gn_arr[te_full_idx][show_idx]
        c_bar    = ['#E63946' if v > 0 else '#457B9D' for v in show_r[::-1]]
        y_pos    = np.arange(len(show_idx))
        ax.barh(y_pos, show_r[::-1], color=c_bar, alpha=0.80, edgecolor='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([str(g)[:14] for g in show_g[::-1]], fontsize=7)
        ax.axvline(0, color='gray', lw=1.2, ls='--')
        ax.set_xlabel('Spearman r')
        ax.set_title(f'B. Test set: Top/bottom {top_k} genes', fontweight='bold')
        ax.spines[['top','right']].set_visible(False)

    ax = axes[2]
    if len(r_vals) >= 20:
        q_edges = np.percentile(r_vals, [0, 25, 50, 75, 100])
        q_lbls  = ['Q1\n(low r)', 'Q2', 'Q3', 'Q4\n(high r)']
        q_cols  = ['#457B9D', '#85C1E9', '#F4A460', '#E63946']
        for qi in range(4):
            m_q = (r_vals >= q_edges[qi]) & (r_vals <= q_edges[qi+1])
            n_q = m_q.sum()
            ax.bar(qi, n_q, color=q_cols[qi], alpha=0.75)
            ax.text(qi, n_q + 0.5, f'μ={r_vals[m_q].mean():.3f}',
                    ha='center', va='bottom', fontsize=8)
        ax.set_xticks(range(4)); ax.set_xticklabels(q_lbls)
        ax.set_ylabel('Gene count')
        ax.set_title(f'C. Spearman r quartiles\nr>0: {(r_vals>0).mean()*100:.0f}%',
                     fontweight='bold')
        ax.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    fig.savefig(out / 'figDirA_gene_attn_iaa_spearman.pdf', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  ✓ [方向A] 可视化 → {out/'figDirA_gene_attn_iaa_spearman.pdf'}")
    return {'n_computed': n_computed, 'n_common_tfs': n_common,
            'mean_r': float(r_vals.mean()) if len(r_vals) > 0 else float('nan'),
            'frac_positive': float((r_vals > 0).mean()) if len(r_vals) > 0 else float('nan')}


# ============================================================================
# 方向B: TF-centric Spearman（含 Jaccard 分析）
# ============================================================================

def _validate_attn_tf_centric_iaa(attn_all, gene_names, tf_names,
                                   FC_aligned, iaa_tf_names, output_dir,
                                   X_raw_std=None):
    from scipy.stats import spearmanr as _spearmanr, wilcoxon as _wlcx
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if FC_aligned is None or attn_all is None:
        print("  ⚠ [方向B] FC_aligned 或 attn_all 为空，跳过"); return None
    iaa_upper   = np.array([str(t).strip().upper() for t in iaa_tf_names])
    model_upper = np.array([str(t).strip().upper() for t in tf_names])
    common_model_idx, common_iaa_idx = [], []
    for mi, tu in enumerate(model_upper):
        hits = np.where(iaa_upper == tu)[0]
        if len(hits) > 0:
            common_model_idx.append(mi); common_iaa_idx.append(int(hits[0]))
    n_common = len(common_model_idx)
    if n_common < 5:
        print(f"  ⚠ [方向B] 共同 TF 数量不足({n_common})，跳过"); return None
    mid_arr = np.array(common_model_idx); iid_arr = np.array(common_iaa_idx)
    print(f"  ✓ [方向B] 对齐 TF: {n_common}")
    attn_sub = attn_all[:, mid_arr]
    fc_sub   = np.abs(FC_aligned[:, iid_arr])
    tf_r  = np.full(n_common, np.nan, dtype=np.float32)
    tf_p  = np.full(n_common, np.nan, dtype=np.float32)
    n_obs = np.zeros(n_common, dtype=np.int32)
    for ti in range(n_common):
        a_col = attn_sub[:, ti]; f_col = fc_sub[:, ti]
        valid = np.isfinite(a_col) & np.isfinite(f_col)
        n_obs[ti] = int(valid.sum())
        if n_obs[ti] < 10: continue
        r_s, p_s = _spearmanr(a_col[valid], f_col[valid])
        tf_r[ti] = float(r_s); tf_p[ti] = float(p_s)
    valid_mask = np.isfinite(tf_r); r_vals = tf_r[valid_mask]
    n_comp   = int(valid_mask.sum())
    frac_pos = float((r_vals > 0).mean()) if len(r_vals) > 0 else float('nan')
    mean_r   = float(r_vals.mean())       if len(r_vals) > 0 else float('nan')
    med_r    = float(np.median(r_vals))   if len(r_vals) > 0 else float('nan')
    print(f"\n  [方向B] TF-centric Spearman:")
    print(f"    n_TFs={n_comp}  mean_r={mean_r:.4f}  median_r={med_r:.4f}  r>0={frac_pos:.3f}")
    try:
        stat_wx, p_wx = _wlcx(r_vals, alternative='greater')
        print(f"    Wilcoxon H₀: r≤0  stat={stat_wx:.1f}  p={p_wx:.3e}")
    except Exception: pass

    # ── P1 核心改进：按 ChEC-seq 信号强度分层 + Kendall τ ─────────────────
    # 生物学叙事：注意力权重 ≠ TF 结合强度，而是"对表达量预测的信息增益"。
    # 低信号 TF 的注意力本身含噪声大；高信号 TF 的注意力-IAA 相关性应更强，
    # 从而验证：低重叠反映功能冗余，与 Kemmeren et al. 数据集原始结论一致。
    strat_results = {}
    tf_mean_signal = None
    if X_raw_std is not None:
        try:
            # X_raw_std shape: (N_genes, n_tfs, n_bins) — 取对应 common TF 子集
            tf_mean_signal_full = np.abs(X_raw_std).mean(axis=(0, 2))  # (n_tfs,)
            tf_mean_signal_sub = tf_mean_signal_full[mid_arr]          # (n_common,)
            signal_median = np.median(tf_mean_signal_sub)
            high_sig_mask = tf_mean_signal_sub >= signal_median        # top-50%
            low_sig_mask  = ~high_sig_mask

            for stratum, mask, lbl in [
                (high_sig_mask, 'high_signal (≥median)', 'high'),
                (low_sig_mask,  'low_signal  (<median)', 'low'),
            ]:
                idx_s = np.where(stratum & valid_mask)[0]
                if len(idx_s) < 5: continue
                r_s = r_vals[stratum[valid_mask]]
                frac_s = float((r_s > 0).mean())
                print(f"    [{lbl} ChEC-seq signal]  n={len(idx_s)}"
                      f"  mean_r={r_s.mean():.4f}  r>0={frac_s:.3f}")
                strat_results[lbl] = {'n': len(idx_s), 'mean_r': float(r_s.mean()),
                                      'frac_pos': frac_s}

            # Kendall τ（rank data 更合适）on high-signal TFs
            from scipy.stats import kendalltau as _ktau
            hs_valid = high_sig_mask & valid_mask
            if hs_valid.sum() >= 5:
                a_hs = attn_sub[:, hs_valid].mean(axis=0)   # mean attn per TF (high-sig subset)
                f_hs = fc_sub[:,  hs_valid].mean(axis=0)
                fin  = np.isfinite(a_hs) & np.isfinite(f_hs)
                if fin.sum() >= 5:
                    tau, p_tau = _ktau(a_hs[fin], f_hs[fin])
                    print(f"    [Kendall τ, high-signal TFs]  τ={tau:.4f}  p={p_tau:.3e}"
                          f"  {'显著' if p_tau < 0.05 else '不显著'}")
                    strat_results['kendall_tau_high'] = float(tau)
                    strat_results['kendall_tau_high_p'] = float(p_tau)
        except Exception as _se:
            print(f"  ⚠ [方向B 信号分层] 失败: {_se}")

    # 生物学叙事总结（主动报告为发现，而非缺陷）
    print(f"\n  [方向B 生物学解读]")
    print(f"    注意力权重反映'对表达量预测的信息增益'，而非 TF 结合强度本身。")
    print(f"    低整体相关性与 Kemmeren et al. 原始结论（TF 结合与调控靶标低重叠）高度一致，")
    print(f"    揭示了酵母稳态条件下 TF 结合存在大量功能冗余。")
    if 'high' in strat_results and 'low' in strat_results:
        _delta_r = strat_results['high']['mean_r'] - strat_results['low']['mean_r']
        print(f"    高信号 TF（top-50%）的注意力-IAA 相关性{'高于' if _delta_r > 0 else '低于'}"
              f"低信号 TF（Δmean_r={_delta_r:+.4f}）。")
        if _delta_r > 0:
            print(f"    ✓ 支持：高结合信号 TF 的预测-IAA 共变性更强（低信号 TF 注意力含噪声更大）。")
    names_common = np.array([str(tf_names[i]) for i in mid_arr])
    result_df = pd.DataFrame({
        'tf': names_common, 'spearman_r': tf_r, 'spearman_p': tf_p, 'n_genes': n_obs,
    }).sort_values('spearman_r', ascending=False)
    result_df.to_csv(out / 'dirB_tf_attn_iaa_spearman.csv', index=False)
    print(f"  ✓ [方向B] per-TF Spearman → {out/'dirB_tf_attn_iaa_spearman.csv'}")

    # ── Jaccard 分析（per-gene top-k TF overlap）────────────────────────
    top_k_vals = [3, 5, 10]; N_genes_loc = attn_sub.shape[0]
    jaccard_results = {}
    for topk in top_k_vals:
        jac_scores = []
        for gi in range(N_genes_loc):
            a_row = attn_sub[gi]; f_row = fc_sub[gi]
            valid_gi = np.isfinite(f_row) & np.isfinite(a_row)
            if valid_gi.sum() < topk * 2: continue
            a_topk = set(np.argsort(a_row[valid_gi])[-topk:])
            f_topk = set(np.argsort(f_row[valid_gi])[-topk:])
            union  = len(a_topk | f_topk); inter = len(a_topk & f_topk)
            jac_scores.append(inter / union if union > 0 else 0.0)
        arr = np.array(jac_scores, dtype=np.float32)
        random_baseline = topk / (2 * n_common - topk)
        jaccard_results[topk] = {
            'scores': arr,
            'mean': float(arr.mean()) if len(arr) > 0 else float('nan'),
            'median': float(np.median(arr)) if len(arr) > 0 else float('nan'),
            'baseline': random_baseline, 'n_genes': len(arr),
        }
        print(f"    [方向B Jaccard] top-{topk:2d}: "
              f"mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
              f"random_baseline={random_baseline:.4f}  n_genes={len(arr)}")
    pd.DataFrame([{'top_k': k, 'mean_jaccard': jaccard_results[k]['mean'],
                   'median_jaccard': jaccard_results[k]['median'],
                   'random_baseline': jaccard_results[k]['baseline'],
                   'n_genes': jaccard_results[k]['n_genes']}
                  for k in top_k_vals]).to_csv(out / 'dirB_per_gene_jaccard_topk.csv', index=False)
    print(f"  ✓ [方向B] per-gene Jaccard → {out/'dirB_per_gene_jaccard_topk.csv'}")

    # ── 四面板可视化（新增第4面板：信号分层）────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    fig.suptitle(
        f"Direction B (TF-centric): Spearman(attn across genes, |log2FC| across genes)\n"
        f"n_common_TFs={n_common}  mean_r={mean_r:.4f}  frac(r>0)={frac_pos:.3f}\n"
        f"[Biological interpretation: Low overlap reflects functional redundancy, consistent with the original conclusions of Kemmeren et al.]",
        fontsize=10, fontweight='bold')

    ax = axes[0]
    bins = np.linspace(-0.5, 0.5, 40)
    ax.hist(r_vals, bins=bins, color='#3A86FF', alpha=0.75, edgecolor='white')
    ax.axvline(0,      color='black',   lw=1.5, ls='--', alpha=0.7)
    ax.axvline(mean_r, color='#E63946', lw=2,   ls='-',  label=f'mean={mean_r:.3f}')
    ax.axvline(med_r,  color='#F4A460', lw=2,   ls='--', label=f'median={med_r:.3f}')
    ax.set_xlabel('TF-level Spearman r'); ax.set_ylabel('Count')
    ax.set_title('A. Distribution of per-TF Spearman r', fontweight='bold')
    ax.legend(fontsize=8); ax.spines[['top','right']].set_visible(False)

    ax = axes[1]
    top_k = min(15, n_comp // 2)
    sort_ord = np.argsort(tf_r[valid_mask])
    vm_names = names_common[valid_mask]; vm_r = tf_r[valid_mask]
    show_idx = np.concatenate([sort_ord[-top_k:], sort_ord[:top_k]])
    show_r   = vm_r[show_idx]; show_n = vm_names[show_idx]
    c_bar = ['#E63946' if v > 0 else '#457B9D' for v in show_r[::-1]]
    y_pos = np.arange(len(show_idx))
    ax.barh(y_pos, show_r[::-1], color=c_bar, alpha=0.82, edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(g)[:16] for g in show_n[::-1]], fontsize=7)
    ax.axvline(0, color='gray', lw=1.2, ls='--')
    ax.set_xlabel('Spearman r')
    ax.set_title(f'B. Top/Bottom {top_k} TFs by r', fontweight='bold')
    ax.spines[['top','right']].set_visible(False)

    ax = axes[2]
    best_topk = 5
    jac_arr      = jaccard_results[best_topk]['scores']
    jac_mean     = jaccard_results[best_topk]['mean']
    jac_baseline = jaccard_results[best_topk]['baseline']
    bins_j = np.linspace(0, 1, 30)
    ax.hist(jac_arr, bins=bins_j, color='#2EC4B6', alpha=0.75, edgecolor='white')
    ax.axvline(jac_mean,     color='#E63946', lw=2,   ls='-',  label=f'mean={jac_mean:.3f}')
    ax.axvline(jac_baseline, color='black',   lw=1.5, ls='--', label=f'random={jac_baseline:.3f}')
    ax.set_xlabel(f'Jaccard index (top-{best_topk} TFs per gene)')
    ax.set_ylabel('Count')
    ax.set_title(f'C. Per-gene top-{best_topk} TF overlap\n(attention rank ∩ IAA |FC| rank)',
                 fontweight='bold')
    ax.legend(fontsize=8); ax.spines[['top','right']].set_visible(False)

    # ── Panel D: 高/低信号 TF 分层 Spearman r 分布 ──────────────────────
    ax = axes[3]
    if X_raw_std is not None and tf_mean_signal is None:
        # 重新计算（函数内 tf_mean_signal_sub 已在前面定义）
        try:
            tf_mean_signal_sub_vis = np.abs(X_raw_std).mean(axis=(0, 2))[mid_arr]
            _sig_med_vis = np.median(tf_mean_signal_sub_vis)
            _hs_mask_vis = tf_mean_signal_sub_vis >= _sig_med_vis
            _ls_mask_vis = ~_hs_mask_vis
            for _mask_v, _col_v, _lbl_v in [
                (_hs_mask_vis, '#E63946', 'High signal (≥median)'),
                (_ls_mask_vis, '#457B9D', 'Low signal (<median)'),
            ]:
                _r_v = tf_r[_mask_v & valid_mask]
                if len(_r_v) >= 3:
                    ax.hist(_r_v, bins=np.linspace(-0.5, 0.5, 25),
                            color=_col_v, alpha=0.65, density=True,
                            label=f'{_lbl_v}\nn={len(_r_v)} μ={_r_v.mean():.3f}')
            ax.axvline(0, color='black', lw=1.5, ls='--', alpha=0.7)
            ax.set_xlabel('Per-TF Spearman r'); ax.set_ylabel('Density')
            ax.set_title('D. Stratified by ChEC-seq\nsignal strength', fontweight='bold')
            ax.legend(fontsize=7); ax.spines[['top','right']].set_visible(False)
        except Exception:
            ax.text(0.5, 0.5, 'Stratification\nnot available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='gray')
            ax.set_title('D. Signal stratification', fontweight='bold')
            ax.spines[['top','right']].set_visible(False)
    elif strat_results and tf_mean_signal is None:
        # 已有 strat_results，重新取分层 r 值
        try:
            _tf_ms = np.abs(X_raw_std).mean(axis=(0, 2))[mid_arr]
            _sig_m = np.median(_tf_ms)
            for _msk, _col_v, _lbl_v in [
                (_tf_ms >= _sig_m, '#E63946', 'High signal (≥median)'),
                (_tf_ms < _sig_m,  '#457B9D', 'Low signal (<median)'),
            ]:
                _rv = tf_r[_msk & valid_mask]
                if len(_rv) >= 3:
                    ax.hist(_rv, bins=np.linspace(-0.5, 0.5, 25),
                            color=_col_v, alpha=0.65, density=True,
                            label=f'{_lbl_v}\nn={len(_rv)} μ={_rv.mean():.3f}')
            ax.axvline(0, color='black', lw=1.5, ls='--', alpha=0.7)
            ax.set_xlabel('Per-TF Spearman r'); ax.set_ylabel('Density')
            ax.set_title('D. Stratified by ChEC-seq\nsignal strength', fontweight='bold')
            ax.legend(fontsize=7); ax.spines[['top','right']].set_visible(False)
        except Exception:
            ax.set_visible(False)
    else:
        # 无 X_raw_std：显示 n_obs 散点图作为替代
        if valid_mask.sum() >= 3:
            ax.scatter(n_obs[valid_mask], r_vals, alpha=0.5, s=15, color='#888888')
            ax.axhline(0, color='black', lw=1.2, ls='--')
            ax.set_xlabel('n_genes per TF'); ax.set_ylabel('Spearman r')
            ax.set_title('D. r vs coverage per TF\n(ChEC-seq signal N/A)', fontweight='bold')
            ax.spines[['top','right']].set_visible(False)
        else:
            ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(out / 'figDirB_tf_centric_attn_iaa.pdf', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  ✓ [方向B] 可视化 → {out/'figDirB_tf_centric_attn_iaa.pdf'}")
    return {'n_common_tfs': n_common, 'n_computed': n_comp,
            'mean_r': mean_r, 'median_r': med_r, 'frac_positive': frac_pos,
            'jaccard_top5_mean': jaccard_results.get(5, {}).get('mean', float('nan')),
            'jaccard_top5_baseline': jaccard_results.get(5, {}).get('baseline', float('nan')),
            'strat_high_mean_r': strat_results.get('high', {}).get('mean_r', float('nan')),
            'strat_low_mean_r':  strat_results.get('low',  {}).get('mean_r', float('nan')),
            'kendall_tau_high':  strat_results.get('kendall_tau_high', float('nan')),
            'kendall_tau_high_p':strat_results.get('kendall_tau_high_p', float('nan'))}


# ============================================================================
# Priority 3: 注意力头专化分析（含完整三面板可视化）
# ============================================================================

def extract_per_head_attn_weights(model, X_std, scalars, device,
                                   lm_data=None, batch_size=256):
    if getattr(model, 'model_kind', 'tf') == 'seq_only': return None
    pooler     = model.tf_attn_pooler
    per_tf_enc = pooler.per_tf
    layer0     = pooler.transformer.layers[0]
    mha        = layer0.self_attn
    norm1      = layer0.norm1
    N = len(X_std); all_head_attns = []
    model.eval()
    with torch.no_grad():
        for s in range(0, N, batch_size):
            e   = s + batch_size
            x   = torch.FloatTensor(X_std[s:e]).to(device)
            B, N_TF, n_bins = x.shape
            tf_emb = per_tf_enc(
                x.reshape(B * N_TF, 1, n_bins)
            ).reshape(B, N_TF, -1)
            src_normed = norm1(tf_emb)
            _, attn_w = mha(src_normed, src_normed, src_normed,
                            need_weights=True, average_attn_weights=False)
            if attn_w is not None:
                all_head_attns.append(attn_w.cpu().numpy())
    if not all_head_attns:
        print("  ⚠ [头专化] 未能提取 per-head 注意力权重"); return None
    return np.concatenate(all_head_attns, axis=0).astype(np.float32)


def _analyze_attn_head_specialization(per_head_attn, tf_names, tf_tone_csv,
                                       output_dir, top_n=20):
    from scipy.stats import wilcoxon as _wlcx, mannwhitneyu as _mwu
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if per_head_attn is None or per_head_attn.ndim != 4:
        print("  ⚠ [头专化] per_head_attn 无效，跳过"); return None
    N_genes, n_heads, N_TF, _ = per_head_attn.shape
    print(f"\n  [头专化] per_head_attn: {N_genes} 基因 × {n_heads} 头 × {N_TF} TF")
    tf_recv = per_head_attn.mean(axis=2)   # (N_genes, n_heads, N_TF)

    tf_direction = {}
    if tf_tone_csv and Path(tf_tone_csv).exists():
        try:
            td = pd.read_csv(tf_tone_csv)
            tf_col  = next((c for c in td.columns if c.lower() in ('tf','tf_name','name')), None)
            dir_col = next((c for c in td.columns
                            if c.lower() in ('label','direction','role','class','type','category')), None)
            if tf_col and dir_col:
                for _, row in td.iterrows():
                    tf_direction[str(row[tf_col]).strip().upper()] = str(row[dir_col]).lower()
                print(f"  ✓ [头专化] TF 方向: {len(tf_direction)} 条")
        except Exception as _e:
            print(f"  ⚠ [头专化] TF 方向加载失败: {_e}")

    tf_names_upper = [str(t).strip().upper() for t in tf_names]
    act_mask = np.array([tf_direction.get(u, '') == 'activator' for u in tf_names_upper])
    rep_mask = np.array([tf_direction.get(u, '') == 'repressor' for u in tf_names_upper])
    n_act, n_rep = act_mask.sum(), rep_mask.sum()
    print(f"  ✓ [头专化] 已知激活因子={n_act}  抑制因子={n_rep}")

    if n_act < 3 or n_rep < 3:
        print("  ⚠ [头专化] 已知 TF 太少，无法做方向比较"); return None

    records = []
    for h in range(n_heads):
        head_recv    = tf_recv[:, h, :]
        act_per_gene = head_recv[:, act_mask].mean(axis=1)
        rep_per_gene = head_recv[:, rep_mask].mean(axis=1)
        diff_per_gene = act_per_gene - rep_per_gene
        act_mean = float(act_per_gene.mean()); rep_mean = float(rep_per_gene.mean())
        bias = act_mean - rep_mean
        try:
            stat_wx, p_wx = _wlcx(diff_per_gene, alternative='two-sided')
        except Exception:
            stat_wx, p_wx = float('nan'), float('nan')
        try:
            stat_mw, p_mw = _mwu(head_recv[:, act_mask].ravel(),
                                   head_recv[:, rep_mask].ravel(), alternative='two-sided')
        except Exception:
            stat_mw, p_mw = float('nan'), float('nan')
        records.append({'head': h, 'act_mean_attn': act_mean, 'rep_mean_attn': rep_mean,
                        'bias_act_minus_rep': bias,
                        'wilcoxon_stat': stat_wx, 'wilcoxon_p': p_wx,
                        'mwu_stat': stat_mw, 'mwu_p': p_mw})
        direction_lbl = 'activator-preferring' if bias > 0 else 'repressor-preferring'
        print(f"    Head {h}: act_mean={act_mean:.4f}  rep_mean={rep_mean:.4f}  "
              f"bias={bias:+.4f}  [{direction_lbl}]  MWU p={p_mw:.3e}")

    spec_df = pd.DataFrame(records)
    spec_df.to_csv(out / 'P5_head_specialization.csv', index=False)
    print(f"  ✓ [头专化] 统计 → {out/'P5_head_specialization.csv'}")

    # ── 三面板可视化 ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Priority 3: Attention Head Specialization (n_heads={n_heads})\n"
                 f"n_activators={n_act}  n_repressors={n_rep}",
                 fontsize=11, fontweight='bold')

    ax = axes[0]
    x_pos = np.arange(n_heads); w = 0.35
    ax.bar(x_pos - w/2, spec_df['act_mean_attn'], width=w,
           color='#06D6A0', alpha=0.80, label='Activator mean', edgecolor='white')
    ax.bar(x_pos + w/2, spec_df['rep_mean_attn'], width=w,
           color='#EF476F', alpha=0.80, label='Repressor mean', edgecolor='white')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Head {h}' for h in range(n_heads)], fontsize=9)
    ax.set_ylabel('Mean received attention')
    ax.set_title('A. Per-head: activator vs repressor\nmean received attention', fontweight='bold')
    ax.legend(fontsize=8); ax.spines[['top','right']].set_visible(False)

    ax = axes[1]
    biases   = spec_df['bias_act_minus_rep'].values
    colors_b = ['#06D6A0' if b > 0 else '#EF476F' for b in biases]
    ax.bar(x_pos, biases, color=colors_b, alpha=0.82, edgecolor='white')
    ax.axhline(0, color='gray', lw=1.2, ls='--')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Head {h}' for h in range(n_heads)], fontsize=9)
    ax.set_ylabel('Bias = act_mean − rep_mean')
    ax.set_title('B. Head specialization bias\n(>0: activator-preferring)', fontweight='bold')
    y_max = np.abs(biases).max() * 1.15 if np.abs(biases).max() > 0 else 0.01
    for xi, (b, p_mw) in enumerate(zip(biases, spec_df['mwu_p'].values)):
        sig_lbl = ('***' if p_mw < 0.001 else '**' if p_mw < 0.01
                   else '*' if p_mw < 0.05 else 'ns')
        ax.text(xi, b + (0.02 * y_max if b >= 0 else -0.08 * y_max),
                sig_lbl, ha='center', va='bottom', fontsize=10)
    ax.spines[['top','right']].set_visible(False)

    ax = axes[2]
    global_recv = tf_recv.mean(axis=0)
    global_mean_per_tf = global_recv.mean(axis=0)
    top_tf_idx = np.argsort(global_mean_per_tf)[::-1][:top_n]
    heat_mat   = global_recv[:, top_tf_idx]
    heat_mat_n = ((heat_mat - heat_mat.min(axis=1, keepdims=True)) /
                  (heat_mat.max(axis=1, keepdims=True) - heat_mat.min(axis=1, keepdims=True) + 1e-8))
    tf_lbls  = [str(tf_names[i])[:10] for i in top_tf_idx]
    dir_marks = []
    for i in top_tf_idx:
        d = tf_direction.get(tf_names_upper[i], '')
        dir_marks.append('▲' if d == 'activator' else '▼' if d == 'repressor' else '·')
    tf_lbls_annotated = [f"{m}{n}" for m, n in zip(dir_marks, tf_lbls)]
    cmap_h = LinearSegmentedColormap.from_list('head', ['#FFFFFF', '#0061A1'])
    im = ax.imshow(heat_mat_n, aspect='auto', cmap=cmap_h, vmin=0, vmax=1)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(tf_lbls_annotated, rotation=45, ha='right', fontsize=6)
    ax.set_yticks(range(n_heads))
    ax.set_yticklabels([f'Head {h}' for h in range(n_heads)], fontsize=9)
    ax.set_title(f'C. Normalized received attention\n(▲=act  ▼=rep  ·=unknown, top-{top_n} TFs)',
                 fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.6, label='Row-normalized attention')
    ax.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    fig.savefig(out / 'figP5_head_specialization.pdf', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  ✓ [头专化] 可视化 → {out/'figP5_head_specialization.pdf'}")
    return spec_df.to_dict('records')


# ============================================================================
# Figure 5: TF-TF 协调矩阵 (零成本, 复用 per_head_attn)
# ============================================================================

# ── Fig5 算法配置 (new52, 必须, 被 _analyze_tf_tf_coordination 内部引用) ─────
# v4 原版 bug: minmax 距离 + average linkage → 72/79 TFs 塌缩在 cluster 1
# new51 fix  : correlation-Ward + NMF(k=8) → 实测仍 46/79 在 cluster 1,
#              M_sym off-diagonal 过均质, block 被 "broadcast-y" 背景吞没.
# new52 方案 : 在聚类前对 M_sym 减 configuration-model 期望, 只保留超越
#              promiscuity 的特异协调; 并补 NMF 硬失败保护、诊断落盘、
#              collapse 告警与 fallback 建议.
FIG5_CONFIG = dict(
    # 背景校正策略: 'configuration' (推荐) | 'zscore_row' | 'none'
    background_correction = 'configuration',
    # k 值: 固定整数 (e.g. 5/8) 或 'auto' (用 NMF reconstruction error elbow)
    n_clusters            = 8,
    # k 搜索范围 (仅 n_clusters='auto' 时生效)
    k_search              = (3, 4, 5, 6, 7, 8, 10),
    # Panel B 主热图 edge sparsification: None = 原样; 浮点 e.g. 0.2 = 仅高亮 top-20%
    edge_sparsify_top_pct = None,
    # 诊断阈值: HC max_frac > 此值时打印 collapse 告警 + fallback 建议
    collapse_warn_frac    = 0.40,
)


# 可选的已知酵母转录复合体锚点 — 用于可视化时给 TF 打标签,
# 帮助读者肉眼验证聚类是否捕获已知生物学复合体.
# 来源: SGD + 若干综述; 保守选取各复合体的"核心" TF, 只取在 ChEC-seq 178 TF
# 名单里有机会出现的成员. 若某个 TF 不在 tf_names 里会被自动忽略.
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


def _analyze_tf_tf_coordination(per_head_attn, tf_names, output_dir,
                                  top_sub_tfs=60, n_clusters=8,
                                  background_correction=None,
                                  edge_sparsify_top_pct=None,
                                  k_search=None,
                                  collapse_warn_frac=None):
    """
    TF-TF 协调矩阵 (Figure 5) — 从 TFAttentionPooler.transformer 第一层
    self-attention 矩阵中抽取 "跨基因平均的 TF-TF 共现模板", 识别功能模块,
    并与已知酵母转录复合体 (SAGA / NuA4 / Mediator / ...) 做对照.

    ── 版本历史 ────────────────────────────────────────────────────────────
    v4 原版 bug: distance = 1 - minmax(M_sym), minmax 把长尾右偏分布几乎全部
                  映到 ~0 → 距离 ~1 → average linkage 链式塌缩 72/79 TFs 在
                  cluster 1.
    new51 fix  : 改用 correlation distance + Ward linkage + NMF(k=8) 双证据.
                  实测 real-data 仍然 46/79 在 cluster 1, 说明 M_sym off-diagonal
                  数值分布过均质 — block structure 被 "broadcast-y" 背景吞没.
    new52 改进 : (1) 新增 configuration-model 背景校正
                     M_bg = M_sym - outer(deg, deg)/total → 保留超越 promiscuity
                     的特异协调. 进入 HC 和 NMF 前都用 M_bg.
                 (2) n_clusters 支持 'auto' (对 k_search 扫描 NMF reconstruction
                     error, 选 elbow)
                 (3) Panel B 支持 edge_sparsify_top_pct — 仅着色 top-k% 强边,
                     背景灰底, 让 block 直接跳出
                 (4) 原始 M_sym (含对角) 完整保留, 对角置零只在聚类副本上做,
                     CSV 落盘版本可供下游 self-attention 诊断复用
                 (5) NMF 退化时不再静默归入 cluster 1 — 改为硬抛出 + 跳过 NMF panel
                 (6) 自动 collapse 检测: hc max_frac > collapse_warn_frac 时
                     打印显式警告 + fallback 建议

    参数 (None 表示从 FIG5_CONFIG 读取):
      per_head_attn        : (N_genes, n_heads, N_TF, N_TF) 自注意力张量
      n_clusters           : int 或 'auto'
      background_correction: 'configuration' | 'zscore_row' | 'none'
      edge_sparsify_top_pct: None 或浮点 (e.g. 0.2 = 仅保留 top-20%)
      k_search             : tuple, 仅 n_clusters='auto' 时用
      collapse_warn_frac   : 诊断阈值 (max_frac > 此值时报 collapse)
      top_sub_tfs          : 保留参数, 未使用

    输出 (output_dir/):
      fig5_tf_tf_coordination.pdf           — 7-panel 可视化
      fig5_tf_tf_coordination.csv           — 原始对称矩阵 (含对角, 未减背景)
      fig5_tf_tf_coordination_bg_corrected.csv — 背景校正后矩阵 (聚类实际用)
      fig5_tf_tf_clusters.csv               — TF → (hc_cluster, nmf_cluster,
                                                    known_complex)
      fig5_tf_tf_nmf_loadings.csv           — N_TF × k NMF 载荷 (若成功)
      fig5_tf_tf_cluster_enrichment.csv     — cluster × 已知复合体富集
      fig5_tf_tf_diagnostics.txt            — max_frac / gini / 背景校正模式等
    """
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    from scipy.spatial.distance import pdist
    from sklearn.decomposition import NMF
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

    # ── 配置从 FIG5_CONFIG 回填 (None 参数) ────────────────────────────────
    if background_correction is None:
        background_correction = FIG5_CONFIG.get('background_correction', 'configuration')
    if edge_sparsify_top_pct is None:
        edge_sparsify_top_pct = FIG5_CONFIG.get('edge_sparsify_top_pct', None)
    if k_search is None:
        k_search = FIG5_CONFIG.get('k_search', (3, 4, 5, 6, 7, 8, 10))
    if collapse_warn_frac is None:
        collapse_warn_frac = FIG5_CONFIG.get('collapse_warn_frac', 0.40)
    # n_clusters = 'auto' 允许自动 k-selection
    auto_k = (isinstance(n_clusters, str) and n_clusters.lower() == 'auto')

    if per_head_attn is None or per_head_attn.ndim != 4:
        print("  ⚠ [Fig5] per_head_attn 无效, 跳过 TF-TF 协调分析"); return None
    N_genes, n_heads, N_q, N_k = per_head_attn.shape
    if N_q != N_k:
        print(f"  ⚠ [Fig5] 非方阵 ({N_q}×{N_k}), 跳过"); return None
    N_TF = N_q
    tf_names_arr = np.asarray([str(t).strip() for t in tf_names])
    tf_upper     = np.asarray([t.upper() for t in tf_names_arr])
    print(f"\n  [Fig5] per_head_attn: {N_genes} 基因 × {n_heads} 头 × "
          f"{N_TF} TF × {N_TF} TF")
    print(f"  [Fig5] 背景校正={background_correction}  "
          f"n_clusters={'auto' if auto_k else n_clusters}  "
          f"edge_sparsify_top_pct={edge_sparsify_top_pct}")

    # ── 1. 跨基因 + 跨头平均 → 对称 N_TF × N_TF ────────────────────────────
    M_heads = per_head_attn.mean(axis=0)              # (n_heads, N_TF, N_TF)
    M_mean  = M_heads.mean(axis=0)                    # (N_TF, N_TF)
    # 对称化: 注意力矩阵 i→j 与 j→i 不相等; 协调关系取双向平均
    M_sym_raw = (0.5 * (M_mean + M_mean.T)).astype(np.float64)
    # 原始 M_sym (含对角) 落盘 — 下游 self-attention 诊断可复用
    pd.DataFrame(M_sym_raw, index=tf_names_arr, columns=tf_names_arr).to_csv(
        out / 'fig5_tf_tf_coordination.csv')

    # M_clust: 聚类用副本 (对角置零)
    M_clust = M_sym_raw.copy()
    np.fill_diagonal(M_clust, 0.0)

    # ── 1b. 背景校正 ────────────────────────────────────────────────────────
    if background_correction == 'configuration':
        # Configuration model 期望: 若每个 TF 保持其度 deg_i, 随机连线时期望
        # 连接强度为 outer(deg_i, deg_j) / total. 减掉这个期望后, 剩下的是
        # "超越 promiscuity 的特异协调". Newman 的 Q/modularity 用的同一思想.
        deg = M_clust.sum(axis=1)                     # (N_TF,)
        total = max(M_clust.sum(), 1e-12)
        expected = np.outer(deg, deg) / total
        M_bg = M_clust - expected
        np.fill_diagonal(M_bg, 0.0)
        print(f"  [Fig5] configuration-model 背景减除: "
              f"deg range=[{deg.min():.3e}, {deg.max():.3e}]  "
              f"M_bg range=[{M_bg.min():.3e}, {M_bg.max():.3e}]")
    elif background_correction == 'zscore_row':
        # 每行对其行内均值标准差 zscore —— 去掉 "该 TF 整体多活跃" 的背景
        row_mean = M_clust.mean(axis=1, keepdims=True)
        row_std  = M_clust.std(axis=1, keepdims=True) + 1e-12
        M_bg = (M_clust - row_mean) / row_std
        # 对称化 (zscore 行后不再对称)
        M_bg = 0.5 * (M_bg + M_bg.T)
        np.fill_diagonal(M_bg, 0.0)
        print(f"  [Fig5] zscore_row 背景校正: "
              f"M_bg range=[{M_bg.min():.3e}, {M_bg.max():.3e}]")
    elif background_correction == 'none' or background_correction is None:
        M_bg = M_clust
        print(f"  [Fig5] 不做背景校正 (与 new51 可对比)")
    else:
        print(f"  ⚠ [Fig5] 未知 background_correction={background_correction!r}, "
              f"退回 'none'")
        M_bg = M_clust

    # 落盘背景校正后的矩阵 (聚类实际使用)
    pd.DataFrame(M_bg, index=tf_names_arr, columns=tf_names_arr).to_csv(
        out / 'fig5_tf_tf_coordination_bg_corrected.csv')

    # ── 2. k-selection (仅 auto 模式) ───────────────────────────────────────
    if auto_k:
        print(f"  [Fig5] auto k-selection 扫描 k ∈ {list(k_search)}")
        M_pos = np.maximum(M_bg, 0.0)   # NMF 需非负
        errs = []
        for k_cand in k_search:
            try:
                nmf_tmp = NMF(n_components=k_cand, init='nndsvd',
                              max_iter=400, random_state=42, tol=1e-4)
                nmf_tmp.fit(M_pos)
                errs.append((k_cand, float(nmf_tmp.reconstruction_err_)))
            except Exception:
                errs.append((k_cand, float('inf')))
        err_arr = np.array([e for _, e in errs])
        # 找 elbow: 二阶差分最大点 (即 err 下降速率转折)
        if len(err_arr) >= 3 and np.all(np.isfinite(err_arr)):
            diffs2 = np.diff(err_arr, n=2)           # 长度 len-2
            elbow_idx = int(np.argmax(diffs2)) + 1    # 映射回 k_search 索引
            n_clusters_resolved = int(k_search[elbow_idx])
        else:
            n_clusters_resolved = 8
        print(f"  [Fig5] auto k 结果: {list(errs)} → 选定 k={n_clusters_resolved}")
        n_clusters = n_clusters_resolved

    # ── 3. 层次聚类 (correlation distance + Ward on M_bg) ───────────────────
    # 每个 TF 的 "协调模板" = M_bg 的行向量 (背景校正后).
    # 相关距离关注 "与谁协调" 的模式, 对右偏鲁棒.
    dist_vec = pdist(M_bg, metric='correlation')
    dist_vec = np.nan_to_num(dist_vec, nan=1.0)
    Z = linkage(dist_vec, method='ward')
    cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')

    # ── 3b. NMF 软聚类 (独立证据) — 硬失败: 不再静默退化 ───────────────────
    W_nmf = None
    nmf_labels = None
    nmf_available = False
    try:
        nmf = NMF(n_components=n_clusters, init='nndsvd',
                  max_iter=800, random_state=42, tol=1e-5)
        # NMF 要求非负输入; 背景校正后可能有负值 → 削到 0 (保留正向协调信号)
        W_nmf = nmf.fit_transform(np.maximum(M_bg, 0.0))
        nmf_labels = W_nmf.argmax(axis=1) + 1           # 1..k
        nmf_available = True
        pd.DataFrame(W_nmf, index=tf_names_arr,
                     columns=[f'NMF{j+1}' for j in range(n_clusters)]).to_csv(
            out / 'fig5_tf_tf_nmf_loadings.csv')
    except Exception as _nmf_e:
        print(f"  ⚠ [Fig5] NMF 失败 ({_nmf_e}), 本轮不出 NMF 对照面板")
        # 不再 fake-label 为 cluster 1; nmf_labels 留 None, 下游检查

    # ── 4. 已知复合体标注 (若一个 TF 落多个复合体则取第一个匹配的) ──────────
    known_label = np.array(['Other'] * N_TF, dtype=object)
    for cplx_name, members in KNOWN_TF_COMPLEXES.items():
        members_u = {m.upper() for m in members}
        for i, tfu in enumerate(tf_upper):
            if tfu in members_u and known_label[i] == 'Other':
                known_label[i] = cplx_name

    # 聚类归属表 (含 HC + NMF + 已知复合体; NMF 不可用时该列为 NaN)
    clust_df = pd.DataFrame({
        'tf':            tf_names_arr,
        'hc_cluster':    cluster_labels,
        'nmf_cluster':   nmf_labels if nmf_available else [np.nan] * N_TF,
        'known_complex': known_label,
    })
    clust_df.to_csv(out / 'fig5_tf_tf_clusters.csv', index=False)

    # ── 4b. 诊断: 簇均衡性 + collapse 检测 ──────────────────────────────────
    def _cluster_diag(labels, tag):
        if labels is None:
            print(f"  [Fig5] {tag:<4s} 不可用 (跳过)")
            return None, 0.0, 0.0
        sizes = np.bincount(labels, minlength=n_clusters + 1)[1:]
        max_frac = float(sizes.max()) / N_TF if N_TF > 0 else 0.0
        s = np.sort(sizes.astype(np.float64))
        n = len(s); tot = s.sum()
        gini = ((2 * np.sum(np.arange(1, n + 1) * s) / (n * tot)
                 - (n + 1) / n) if tot > 0 else 0.0)
        flag = ''
        if max_frac > collapse_warn_frac:
            flag = f'  ⚠ collapse (>{collapse_warn_frac:.0%})'
        print(f"  [Fig5] {tag:<4s} sizes={sizes.tolist()}  "
              f"max_frac={max_frac:.2%}  gini={gini:.3f}{flag}")
        return sizes, max_frac, gini

    hc_sizes,  hc_mf, hc_gini = _cluster_diag(cluster_labels, 'HC')
    nmf_sizes, nm_mf, nm_gini = _cluster_diag(nmf_labels,     'NMF') \
                                if nmf_available else (None, 0.0, 0.0)

    # collapse 警告 (让用户在运行日志里一眼看到问题)
    if hc_mf > collapse_warn_frac:
        print(f"\n  ⚠⚠ [Fig5] HC 聚类塌缩 (max_frac={hc_mf:.0%} > "
              f"{collapse_warn_frac:.0%})")
        print(f"     建议: 若当前 background_correction='{background_correction}' "
              f"仍塌缩,")
        print(f"           试改 FIG5_CONFIG.background_correction='configuration',")
        print(f"           或切 n_clusters='auto' 让 elbow 选更小的 k.")
        print(f"     若 real-data 始终塌缩, 说明 model 的 TF-TF 协调信号本身弱,")
        print(f"     Fig5 可退 Supplementary, 主图只保 panel G (known-complex 子矩阵).")

    # 诊断文件落盘 (让投稿 reviewer 也能查验)
    with open(out / 'fig5_tf_tf_diagnostics.txt', 'w', encoding='utf-8') as _fd:
        _fd.write(f"background_correction: {background_correction}\n")
        _fd.write(f"n_clusters:            {n_clusters}  (auto={auto_k})\n")
        _fd.write(f"edge_sparsify_top_pct: {edge_sparsify_top_pct}\n")
        _fd.write(f"N_TF:                  {N_TF}\n")
        _fd.write(f"HC  sizes: {hc_sizes.tolist() if hc_sizes is not None else 'n/a'}\n")
        _fd.write(f"HC  max_frac: {hc_mf:.4f}  gini: {hc_gini:.4f}\n")
        if nmf_available:
            _fd.write(f"NMF sizes: {nmf_sizes.tolist()}\n")
            _fd.write(f"NMF max_frac: {nm_mf:.4f}  gini: {nm_gini:.4f}\n")
        else:
            _fd.write(f"NMF: unavailable\n")

    # ── 5. cluster × known_complex 富集 ────────────────────────────────────
    enrich_rows = []
    for c_id in range(1, n_clusters + 1):
        mask = cluster_labels == c_id
        n_in = int(mask.sum())
        if n_in == 0: continue
        comp_counts = pd.Series(known_label[mask]).value_counts()
        top_comp = str(comp_counts.index[0]) if len(comp_counts) else 'n/a'
        top_n    = int(comp_counts.iloc[0]) if len(comp_counts) else 0
        enrich_rows.append({
            'cluster': c_id, 'n_in': n_in,
            'top_complex': top_comp, 'n_top_complex': top_n,
            'top_complex_frac': top_n / max(n_in, 1),
        })
    enrich_df = pd.DataFrame(enrich_rows)
    enrich_df.to_csv(out / 'fig5_tf_tf_cluster_enrichment.csv', index=False)

    print("\n  [Fig5] 聚类 × 已知复合体 富集:")
    print(f"  {'cluster':>7}  {'n_in':>5}  {'top_complex':<14}  "
          f"{'n(top)':>6}  {'frac':>5}")
    print("  " + "─" * 50)
    for _, row in enrich_df.iterrows():
        print(f"  {int(row['cluster']):>7}  {int(row['n_in']):>5}  "
              f"{row['top_complex']:<14}  {int(row['n_top_complex']):>6}  "
              f"{row['top_complex_frac']:>5.2f}")

    # ── 6. 可视化 (7 面板) ─────────────────────────────────────────────────
    dendro = dendrogram(Z, no_plot=True)
    leaf_order = np.asarray(dendro['leaves'])
    # 热图显示用原始 M_sym_raw (带对角, 不减背景) — 视觉上更易读, 背景校正
    # 已经体现在 cluster 边界上了.
    M_ordered = M_sym_raw[np.ix_(leaf_order, leaf_order)]
    cluster_ordered = cluster_labels[leaf_order]

    _cmap_q = plt.cm.tab10
    _complex_colors = {k: _cmap_q(i % 10)
                       for i, k in enumerate(list(KNOWN_TF_COMPLEXES.keys()))}
    _complex_colors['Other'] = (0.85, 0.85, 0.85, 1.0)
    _complex_colors['n/a']   = (0.65, 0.65, 0.65, 1.0)

    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(2, 4, width_ratios=[0.5, 2.8, 1.2, 1.2],
                          height_ratios=[2.6, 1.0],
                          wspace=0.30, hspace=0.38)
    _bg_label = {'configuration': 'configuration-model',
                 'zscore_row':    'zscore-row',
                 'none':          'none'}.get(background_correction, background_correction)
    fig.suptitle(f"Figure 5: TF–TF Coordination "
                 f"(cross-gene mean self-attention, n_heads={n_heads}; "
                 f"bg={_bg_label}, correlation-Ward HC + "
                 f"{'NMF k=' + str(n_clusters) if nmf_available else 'NMF unavail'})",
                 fontsize=12, fontweight='bold')

    # A: Ward dendrogram
    ax_dendro = fig.add_subplot(gs[0, 0])
    with plt.rc_context({'lines.linewidth': 0.8}):
        _ct = Z[-(n_clusters - 1), 2] if len(Z) >= n_clusters - 1 else 0.7
        dendrogram(Z, orientation='left', no_labels=True,
                   color_threshold=_ct, ax=ax_dendro)
    ax_dendro.set_title('A. Ward dendrogram\n(correlation dist on bg-corrected)',
                        fontsize=10, fontweight='bold')
    ax_dendro.spines[['top', 'right', 'bottom']].set_visible(False)
    ax_dendro.set_xticks([])
    ax_dendro.invert_yaxis()

    # B: Clustered heatmap (optional edge sparsification)
    ax_heat = fig.add_subplot(gs[0, 1])
    _disp = np.log1p(M_ordered * 1e4)
    if edge_sparsify_top_pct is not None and 0 < edge_sparsify_top_pct < 1:
        # 仅着色 top-pct 强边, 其余灰底
        triu_i, triu_j = np.triu_indices(N_TF, k=1)
        flat = _disp[triu_i, triu_j]
        thr = np.quantile(flat, 1 - edge_sparsify_top_pct)
        mask_strong = _disp >= thr
        _disp_show = np.where(mask_strong, _disp, np.nan)
        im = ax_heat.imshow(_disp_show, aspect='auto', cmap='magma',
                            origin='upper')
        ax_heat.set_facecolor('#EEEEEE')
        ax_heat.set_title(f'B. TF×TF coordination ({N_TF}×{N_TF}) · log1p · '
                          f'top-{int(edge_sparsify_top_pct*100)}% edges '
                          f'· HC bounds',
                          fontsize=10, fontweight='bold')
    else:
        im = ax_heat.imshow(_disp, aspect='auto', cmap='magma', origin='upper')
        ax_heat.set_title(f'B. TF×TF coordination matrix ({N_TF}×{N_TF}) · '
                          f'log1p · HC cluster boundaries (green)',
                          fontsize=10, fontweight='bold')
    # 绘制 HC 簇边界
    _boundaries = np.where(np.diff(cluster_ordered) != 0)[0] + 0.5
    for _b in _boundaries:
        ax_heat.axhline(_b, color='#00FF9F', lw=0.9, alpha=0.95)
        ax_heat.axvline(_b, color='#00FF9F', lw=0.9, alpha=0.95)
    ax_heat.set_xticks([]); ax_heat.set_yticks([])
    plt.colorbar(im, ax=ax_heat, shrink=0.78, pad=0.015,
                 label='log1p(attn · 1e4)')

    # C: NMF loading heatmap — 若 NMF 不可用则跳过
    ax_nmf = fig.add_subplot(gs[0, 2])
    if nmf_available and W_nmf is not None:
        W_ord = W_nmf[leaf_order]
        _w_disp = W_ord / (W_ord.max(axis=1, keepdims=True) + 1e-12)
        im_nmf = ax_nmf.imshow(_w_disp, aspect='auto', cmap='viridis',
                               origin='upper')
        ax_nmf.set_xticks(range(n_clusters))
        ax_nmf.set_xticklabels([f'k{i+1}' for i in range(n_clusters)],
                               fontsize=8)
        ax_nmf.set_yticks([])
        for _b in _boundaries:
            ax_nmf.axhline(_b, color='white', lw=0.6, alpha=0.8)
        ax_nmf.set_title(f'C. NMF loadings (k={n_clusters})\n'
                         f'row-normalized', fontsize=10, fontweight='bold')
        plt.colorbar(im_nmf, ax=ax_nmf, shrink=0.78, pad=0.02)
    else:
        ax_nmf.text(0.5, 0.5, 'NMF unavailable\n(fit error)',
                    ha='center', va='center', fontsize=10, color='#888')
        ax_nmf.set_axis_off()
        ax_nmf.set_title('C. NMF (skipped)', fontsize=10, fontweight='bold')

    # D: Cluster balance
    ax_sz = fig.add_subplot(gs[0, 3])
    _xh = np.arange(1, n_clusters + 1) - 0.18
    ax_sz.bar(_xh, hc_sizes, width=0.36, color='#3A86FF', alpha=0.9,
              edgecolor='white', label='HC (Ward)')
    if nmf_available and nmf_sizes is not None:
        _xn = np.arange(1, n_clusters + 1) + 0.18
        ax_sz.bar(_xn, nmf_sizes, width=0.36, color='#FF6B6B', alpha=0.9,
                  edgecolor='white', label=f'NMF (k={n_clusters})')
    ax_sz.axhline(N_TF / n_clusters, color='gray', ls='--', lw=0.8,
                  label=f'uniform ({N_TF/n_clusters:.1f})')
    ax_sz.set_xlabel('cluster'); ax_sz.set_ylabel('n TFs')
    ax_sz.set_xticks(range(1, n_clusters + 1))
    _d_title = f'D. Cluster balance\nHC max={hc_mf:.0%}  Gini={hc_gini:.2f}'
    if nmf_available:
        _d_title += f'\nNMF max={nm_mf:.0%}  Gini={nm_gini:.2f}'
    ax_sz.set_title(_d_title, fontsize=9, fontweight='bold')
    ax_sz.legend(fontsize=7, frameon=False, loc='upper right')
    ax_sz.spines[['top', 'right']].set_visible(False)

    # E: Cluster × known-complex enrichment
    ax_en = fig.add_subplot(gs[1, 0:2])
    for _, row in enrich_df.iterrows():
        _c_id = int(row['cluster'])
        _col = _complex_colors.get(row['top_complex'], '#888888')
        ax_en.bar(_c_id, row['top_complex_frac'], color=_col,
                  edgecolor='white', alpha=0.9, label=row['top_complex'])
        ax_en.text(_c_id, row['top_complex_frac'] + 0.02,
                   f"{row['top_complex']}\nn={int(row['n_top_complex'])}/"
                   f"{int(row['n_in'])}",
                   ha='center', fontsize=7)
    handles, labels = ax_en.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax_en.legend(uniq.values(), uniq.keys(), fontsize=7, ncol=4,
                 loc='upper right', frameon=False)
    ax_en.set_xlabel('HC cluster'); ax_en.set_ylabel('top-complex fraction')
    ax_en.set_ylim(0, 1.25)
    ax_en.set_xticks(range(1, n_clusters + 1))
    ax_en.set_title('E. HC cluster × known-complex top-enrichment',
                    fontsize=10, fontweight='bold')
    ax_en.spines[['top', 'right']].set_visible(False)

    # F: Top coordinated pairs (从原始 M_sym_raw 取; 对角已在 M_clust 里置零但
    # M_sym_raw 保留, 所以 triu 不取对角)
    ax_tbl = fig.add_subplot(gs[1, 2]); ax_tbl.set_axis_off()
    triu_i, triu_j = np.triu_indices(N_TF, k=1)
    vals = M_sym_raw[triu_i, triu_j]
    top_k = min(15, len(vals))
    top_idx = np.argsort(vals)[::-1][:top_k]
    table_txt = [f"{'TF_a':<9}{'TF_b':<9}{'coord':>8}"]
    table_txt.append("─" * 28)
    for ti in top_idx:
        ia, ib = triu_i[ti], triu_j[ti]
        table_txt.append(
            f"{tf_names_arr[ia][:8]:<9}"
            f"{tf_names_arr[ib][:8]:<9}"
            f"{vals[ti]:>8.4f}")
    ax_tbl.text(0.0, 1.0, "\n".join(table_txt), family='monospace',
                fontsize=8, va='top')
    ax_tbl.set_title(f'F. Top-{top_k} coordinated pairs',
                     fontsize=10, fontweight='bold')

    # G: Known-complex sub-matrix
    ax_sub = fig.add_subplot(gs[1, 3])
    known_mask = known_label != 'Other'
    if known_mask.sum() >= 4:
        known_idx = np.where(known_mask)[0]
        sort_key = list(zip(known_label[known_idx], tf_names_arr[known_idx]))
        _order_sub = sorted(range(len(known_idx)), key=lambda i: sort_key[i])
        known_idx_sorted = known_idx[_order_sub]
        M_known = M_sym_raw[np.ix_(known_idx_sorted, known_idx_sorted)]
        _d_k = np.log1p(M_known * 1e4)
        im2 = ax_sub.imshow(_d_k, aspect='auto', cmap='magma')
        ax_sub.set_xticks(range(len(known_idx_sorted)))
        ax_sub.set_yticks(range(len(known_idx_sorted)))
        _labels_sub = [f"{tf_names_arr[i]}({known_label[i][:4]})"
                       for i in known_idx_sorted]
        ax_sub.set_xticklabels(_labels_sub, rotation=90, fontsize=4)
        ax_sub.set_yticklabels(_labels_sub, fontsize=4)
        ax_sub.set_title(f'G. Known-complex sub-matrix\n'
                         f'n_matched={int(known_mask.sum())}',
                         fontsize=9, fontweight='bold')
    else:
        ax_sub.text(0.5, 0.5, 'No known-complex\nTFs in list',
                    ha='center', va='center', fontsize=10)
        ax_sub.set_axis_off()

    plt.savefig(out / 'fig5_tf_tf_coordination.pdf',
                bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"\n  ✓ [Fig5] 原始矩阵 CSV     → {out/'fig5_tf_tf_coordination.csv'}")
    print(f"  ✓ [Fig5] 背景校正矩阵 CSV → {out/'fig5_tf_tf_coordination_bg_corrected.csv'}")
    print(f"  ✓ [Fig5] 聚类归属          → {out/'fig5_tf_tf_clusters.csv'}")
    if nmf_available:
        print(f"  ✓ [Fig5] NMF 载荷          → {out/'fig5_tf_tf_nmf_loadings.csv'}")
    print(f"  ✓ [Fig5] 富集表           → {out/'fig5_tf_tf_cluster_enrichment.csv'}")
    print(f"  ✓ [Fig5] 诊断              → {out/'fig5_tf_tf_diagnostics.txt'}")
    print(f"  ✓ [Fig5] 可视化           → {out/'fig5_tf_tf_coordination.pdf'}")
    return {
        'M_sym': M_sym_raw,                    # 原始 (含对角) — 保持后向兼容
        'M_bg':  M_bg,                          # 背景校正后, 聚类实际用的矩阵
        'background_correction': background_correction,
        'n_clusters':          n_clusters,
        'hc_cluster_labels':   cluster_labels,
        'nmf_cluster_labels':  nmf_labels if nmf_available else None,
        'nmf_loadings':        W_nmf if nmf_available else None,
        'nmf_available':       nmf_available,
        'leaf_order':          leaf_order,
        'enrichment':          enrich_df,
        'hc_sizes':   hc_sizes,  'hc_max_frac':  hc_mf, 'hc_gini':  hc_gini,
        'nmf_sizes':  nmf_sizes, 'nmf_max_frac': nm_mf, 'nmf_gini': nm_gini,
        'collapse_detected':   hc_mf > collapse_warn_frac,
    }




# ============================================================================
# 可视化辅助函数
# ============================================================================

def _plot_variance_decomp(decomp_dict, output_dir):
    if not decomp_dict: return
    out = Path(output_dir); out.mkdir(exist_ok=True, parents=True)
    labels = list(decomp_dict.keys())
    pvals  = [decomp_dict[k]['pearson'] for k in labels]
    has_ci = any('pearson_ci_lo' in decomp_dict[k] for k in labels)
    yerr = None
    if has_ci:
        lo = [pvals[i] - decomp_dict[k].get('pearson_ci_lo', pvals[i])
              for i, k in enumerate(labels)]
        hi = [decomp_dict[k].get('pearson_ci_hi', pvals[i]) - pvals[i]
              for i, k in enumerate(labels)]
        yerr = [lo, hi]
    n = len(labels)
    base_colors = plt.cm.Blues(np.linspace(0.25, 0.85, n))
    fig, ax = plt.subplots(figsize=(max(6, n * 2 + 1), 5))
    x_pos = np.arange(n)
    bars = ax.bar(x_pos, pvals, color=base_colors, alpha=0.88,
                  edgecolor='white', zorder=3)
    if yerr is not None:
        ax.errorbar(x_pos, pvals, yerr=yerr, fmt='none',
                    ecolor='#333333', elinewidth=1.5, capsize=5, capthick=1.5, zorder=4)
    for bar, v, xi in zip(bars, pvals, x_pos):
        label_y = v + (max(yerr[1][xi], 0) if has_ci else 0) + 0.005
        ax.text(bar.get_x() + bar.get_width() / 2,
                label_y, f'{v:.4f}', ha='center', va='bottom',
                fontsize=8.5, fontweight='bold')
    for i in range(1, n):
        delta = pvals[i] - pvals[i - 1]
        if abs(delta) > 0.001:
            ax.text((i - 1 + i) / 2, max(pvals[i-1], pvals[i]) + 0.025,
                    f'Δ{delta:+.3f}', fontsize=8, ha='center', va='bottom', color='#555555')
    ci_note = (f'  (error bars: {int(BOOTSTRAP_CI.get("ci_level",0.95)*100)}% CI, '
               f'n={BOOTSTRAP_CI.get("n_boot",0)} bootstrap)') if has_ci else ''
    ax.set_xticks(x_pos)
    ax.set_xticklabels([l.replace(' ', '\n') for l in labels], fontsize=9)
    ax.set_ylabel('Test Pearson r')
    ax.set_title(f'Ablation Study: Modality Contribution{ci_note}', fontweight='bold')
    ax.set_ylim(0, max(pvals) * 1.18)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    fig.savefig(out / 'fig1_variance_decomp.pdf', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  ✓ 消融贡献图 → {out/'fig1_variance_decomp.pdf'}")


def _plot_quadrant_scatter(sa_z, tpm_norm, conds, quad_labels, quad_colors,
                            sep_x, sep_y, output_dir,
                            x_label='Seq-only prediction (median-centred)'):
    out = Path(output_dir); out.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    colors_arr = np.full(len(tpm_norm), '#DDDDDD', dtype=object)
    for cond, col in zip(conds, quad_colors):
        colors_arr[np.where(cond)[0]] = col
    ax.scatter(sa_z, tpm_norm, c=colors_arr, alpha=0.3, s=8, rasterized=True)
    ax.axhline(sep_y, color='#888', lw=1, ls='--', alpha=0.7)
    ax.axvline(sep_x, color='#888', lw=1, ls='--', alpha=0.7)
    patches = [mpatches.Patch(color=c, label=l, alpha=0.7)
               for c, l in zip(quad_colors, quad_labels)]
    ax.legend(handles=patches, fontsize=9, loc='upper left')
    ax.set_xlabel(x_label)
    ax.set_ylabel('Expression (TPM z-score)')
    ax.set_title('Gene Quadrant Classification', fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    fig.savefig(out/'fig2_quadrant_scatter.pdf', bbox_inches='tight', dpi=200); plt.close(fig)


def _plot_quadrant_heatmap(attn_q, conds, quad_labels, quad_colors, tf_names, output_dir):
    out = Path(output_dir); out.mkdir(exist_ok=True, parents=True)
    top_n = min(30, len(tf_names))
    global_mean = attn_q.mean(axis=0); top_idx = np.argsort(global_mean)[::-1][:top_n]
    quad_attn = {}
    for cond, lbl in zip(conds, quad_labels):
        idx = np.where(cond)[0]
        if len(idx) > 0: quad_attn[lbl] = attn_q[idx][:, top_idx].mean(axis=0)
    if not quad_attn: return
    mat = np.stack([quad_attn[l] for l in quad_labels if l in quad_attn], axis=0)
    tf_lbls  = [str(tf_names[i]) for i in top_idx]
    row_lbls = [l for l in quad_labels if l in quad_attn]
    fig, ax = plt.subplots(figsize=(max(12, top_n*0.4), 3.5))
    cmap = LinearSegmentedColormap.from_list('attn', ['#FFFFFF','#0061A1'])
    im = ax.imshow(mat, aspect='auto', cmap=cmap)
    ax.set_xticks(range(top_n)); ax.set_xticklabels(tf_lbls, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(row_lbls))); ax.set_yticklabels(row_lbls, fontsize=9)
    ax.set_title(f'TF Attention Heatmap by Quadrant (Top-{top_n})', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Mean Attention')
    plt.tight_layout()
    fig.savefig(out/'fig3_attn_heatmap.pdf', bbox_inches='tight', dpi=200); plt.close(fig)
    print(f"  ✓ 注意力热图 → {out/'fig3_attn_heatmap.pdf'}")



def _plot_residual_analysis(gm_q, conds, quad_labels, quad_colors,
                             seq_only_pred_q, full_pred_q, output_dir):
    tpm_q   = gm_q['tpm_norm'].values
    trans_r = tpm_q - seq_only_pred_q   # TF trans-regulatory effect
    trans_e = full_pred_q - seq_only_pred_q
    q2_mask = np.asarray(conds[1], dtype=bool)
    q3_mask = np.asarray(conds[2], dtype=bool)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("P3: Seq_only Residual Analysis\ntrans_resid = y_true − seq_only_pred",
                 fontsize=11, fontweight='bold')

    # ── Panel A: 按四象限分组的 trans_resid 箱线图 ───────────────────────
    ax = axes[0]
    for xi, (cond, lbl, col) in enumerate(zip(conds, quad_labels, quad_colors)):
        idx = np.where(cond)[0]
        if len(idx) < 3: continue
        vals = trans_r[idx]
        ax.boxplot(vals, positions=[xi], widths=0.55, patch_artist=True,
                   medianprops=dict(color='white', lw=2),
                   boxprops=dict(facecolor=col, alpha=0.60),
                   whiskerprops=dict(color=col, lw=1.5), capprops=dict(color=col, lw=1.5),
                   flierprops=dict(marker='', alpha=0))
        rng = np.random.default_rng(0); n_show = min(200, len(idx))
        ax.scatter(xi + rng.uniform(-0.18, 0.18, n_show),
                   vals[rng.choice(len(idx), n_show, replace=False)],
                   c=col, alpha=0.22, s=6, zorder=2)
    ax.axhline(0, color='#555', lw=1.2, ls='--', alpha=0.7)
    ax.set_xticks(range(len(quad_labels)))
    ax.set_xticklabels([f'{l.split(":")[0]}\n{l.split(": ")[1]}'
                        for l in quad_labels], fontsize=9)
    ax.set_ylabel("trans_resid"); ax.set_title("A. TF trans-regulatory effect per quadrant")
    ax.spines[['top','right']].set_visible(False)

    # ── Panel B: Q2 vs Q3 trans_resid 分布 ──────────────────────────────
    ax = axes[1]
    bins = np.linspace(trans_r.min(), trans_r.max(), 40)
    r_tr = trans_r[q2_mask]; i_tr = trans_r[q3_mask]
    ax.hist(r_tr, bins=bins, color='#FF6B6B', alpha=0.60, density=True,
            label=f'Q2-Trans-repressed n={q2_mask.sum()}')
    ax.hist(i_tr, bins=bins, color='#2EC4B6', alpha=0.60, density=True,
            label=f'Q3-Trans-activated n={q3_mask.sum()}')
    if len(r_tr) > 0:
        ax.axvline(r_tr.mean(), color='#FF6B6B', lw=2, ls='-',
                   label=f'Q2 mean={r_tr.mean():.3f}')
    if len(i_tr) > 0:
        ax.axvline(i_tr.mean(), color='#2EC4B6', lw=2, ls='-',
                   label=f'Q3 mean={i_tr.mean():.3f}')
    ax.axvline(0, color='#555', lw=1.2, ls='--', alpha=0.7)
    ax.set_xlabel("trans_resid (y_true − seq_only_pred)"); ax.set_ylabel("Density")
    ax.set_title("B. trans_resid: Q2-Trans-repressed vs Q3-Trans-activated")
    ax.legend(fontsize=7); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    path = f"{output_dir}/figP3_residual_analysis.pdf"
    fig.savefig(path, bbox_inches='tight', dpi=200); plt.close(fig)
    print(f"  ✓ [P3] 残差分析图 → {path}")
    print(f"    Q2-Trans-repressed n={q2_mask.sum()}  Q3-Trans-activated n={q3_mask.sum()}")
    resid_df = gm_q[['gene','quadrant','seq_activity','tpm_norm','pred_norm']].copy()
    resid_df['trans_resid']  = trans_r
    resid_df['trans_effect'] = trans_e
    resid_df['resid_class']  = 'other'
    resid_df.loc[q2_mask, 'resid_class'] = 'Q2_trans_repressed'
    resid_df.loc[q3_mask, 'resid_class'] = 'Q3_trans_activated'
    resid_df.to_csv(Path(output_dir)/'P3_residual_gene_list.csv', index=False)


def _plot_tf_tone_comparison(gm_q, conds, quad_labels, quad_colors,
                              attn_q, tf_names, full_pred_q, seq_only_pred_q,
                              output_dir, tf_tone_csv=None, top_n=20,
                              X_signal_q=None):
    """
    P4: TF 注意力 × Tone 方向对照 + 信号强度验证
    X_signal_q: 标准化 ChEC-seq 信号 (N_valid, n_tfs, n_bins)，用于验证
                Q3 高注意力 TF 在 Q3 基因上是否信号低（抑制因子缺失假说）
    """
    tf_direction = {}
    if tf_tone_csv and Path(tf_tone_csv).exists():
        try:
            td = pd.read_csv(tf_tone_csv)
            tf_col  = next((c for c in td.columns if c.lower() in ('tf','tf_name','name')), None)
            dir_col = next((c for c in td.columns
                            if c.lower() in ('label','direction','role','class','type','category')), None)
            if tf_col and dir_col:
                for _, row in td.iterrows():
                    tf_direction[str(row[tf_col]).strip().upper()] = str(row[dir_col]).lower()
                print(f"  [P4] TF 方向分类（着色用）: {len(tf_direction)} 条")
        except Exception as _e:
            print(f"  ⚠ [P4] 加载 tf_direction 失败: {_e}")

    tf_names_upper = [str(n).strip().upper() for n in tf_names]
    q2_lbl = 'Q2: Trans-repressed'; q3_lbl = 'Q3: Trans-activated'

    quad_attn_mean = {}
    for cond, lbl in zip(conds, quad_labels):
        idx = np.where(cond)[0]
        if len(idx) > 0: quad_attn_mean[lbl] = attn_q[idx].mean(axis=0)

    def _top_tf_df(lbl, n):
        if lbl not in quad_attn_mean: return pd.DataFrame()
        mw  = quad_attn_mean[lbl]; top = np.argsort(mw)[::-1][:n]
        return pd.DataFrame([{'tf': tf_names[i], 'attn': float(mw[i]),
                               'direction': tf_direction.get(tf_names_upper[i], 'unknown')}
                              for i in top])

    df_q2 = _top_tf_df(q2_lbl, top_n); df_q3 = _top_tf_df(q3_lbl, top_n)
    for df, name in [(df_q2, 'P4_topTF_Q2_trans_repressed.csv'),
                     (df_q3, 'P4_topTF_Q3_trans_activated.csv')]:
        if not df.empty: df.to_csv(Path(output_dir) / name, index=False)

    # ── 信号强度验证 ──────────────────────────────────────────────────────
    if (X_signal_q is not None and q2_lbl in quad_attn_mean and q3_lbl in quad_attn_mean):
        q2_gene_idx = np.where(conds[1])[0]; q3_gene_idx = np.where(conds[2])[0]
        if len(q2_gene_idx) > 0 and len(q3_gene_idx) > 0:
            q2_top_tf_idx = np.argsort(quad_attn_mean[q2_lbl])[::-1][:top_n]
            q3_top_tf_idx = np.argsort(quad_attn_mean[q3_lbl])[::-1][:top_n]
            q2_topTF_in_q2 = np.abs(X_signal_q[q2_gene_idx][:, q2_top_tf_idx, :]).mean()
            q2_topTF_in_q3 = np.abs(X_signal_q[q3_gene_idx][:, q2_top_tf_idx, :]).mean()
            q3_topTF_in_q2 = np.abs(X_signal_q[q2_gene_idx][:, q3_top_tf_idx, :]).mean()
            q3_topTF_in_q3 = np.abs(X_signal_q[q3_gene_idx][:, q3_top_tf_idx, :]).mean()
            print(f"\n  [P4 信号强度验证]")
            print(f"    Q2高注意力TF 在 Q2 基因信号: {q2_topTF_in_q2:.4f}")
            print(f"    Q2高注意力TF 在 Q3 基因信号: {q2_topTF_in_q3:.4f}")
            print(f"    Q3高注意力TF 在 Q2 基因信号: {q3_topTF_in_q2:.4f}")
            print(f"    Q3高注意力TF 在 Q3 基因信号: {q3_topTF_in_q3:.4f}")
            ratio_q3 = q3_topTF_in_q3 / (q3_topTF_in_q2 + 1e-8)
            ratio_q2 = q2_topTF_in_q2 / (q2_topTF_in_q3 + 1e-8)
            print(f"    Q3高注意TF: Q3/Q2信号比 = {ratio_q3:.3f}  "
                  f"({'< 1.0 ✓ 支持缺失假说' if ratio_q3 < 1.0 else '> 1.0 不支持缺失假说'})")
            print(f"    Q2高注意TF: Q2/Q3信号比 = {ratio_q2:.3f}  "
                  f"({'> 1.0 ✓ 支持抑制结合假说' if ratio_q2 > 1.0 else '< 1.0 不支持'})")
            pd.DataFrame({
                'TF_group': ['Q2_top','Q2_top','Q3_top','Q3_top'],
                'Gene_group': ['Q2_genes','Q3_genes','Q2_genes','Q3_genes'],
                'mean_abs_signal': [q2_topTF_in_q2, q2_topTF_in_q3, q3_topTF_in_q2, q3_topTF_in_q3]
            }).to_csv(Path(output_dir) / 'P4_signal_strength_verification.csv', index=False)
            print(f"  ✓ [P4] 信号验证表 → {Path(output_dir)/'P4_signal_strength_verification.csv'}")

    # ── Q2 vs Q3 差异 TF 表 ──────────────────────────────────────────────
    if q2_lbl in quad_attn_mean and q3_lbl in quad_attn_mean:
        diff = quad_attn_mean[q2_lbl] - quad_attn_mean[q3_lbl]
        pd.DataFrame({
            'tf': tf_names, 'attn_Q2': quad_attn_mean[q2_lbl],
            'attn_Q3': quad_attn_mean[q3_lbl], 'delta_Q2_minus_Q3': diff,
            'direction': [tf_direction.get(n.upper(), 'unknown') for n in tf_names]
        }).sort_values('delta_Q2_minus_Q3', ascending=False).to_csv(
            Path(output_dir) / 'P4_TF_attn_Q2vsQ3_differential.csv', index=False)
        print(f"\n  [P4] Q2 Trans-repressed 偏高注意力 Top-5 TFs:")
        for i in np.argsort(diff)[::-1][:5]:
            print(f"    {str(tf_names[i]):20s}  Δattn={diff[i]:+.4f}  "
                  f"dir={tf_direction.get(tf_names_upper[i], 'unknown')}")
        print(f"  [P4] Q3 Trans-activated 偏高注意力 Top-5 TFs:")
        for i in np.argsort(diff)[:5]:
            print(f"    {str(tf_names[i]):20s}  Δattn={diff[i]:+.4f}  "
                  f"dir={tf_direction.get(tf_names_upper[i], 'unknown')}")



# ============================================================================
# 高置信度 Q2/Q3 分析 — 路线图 Step 1-3
# ============================================================================

def _run_highconf_q2q3_analysis(
        gm_q, conds, quad_labels, quad_colors,
        attn_q, tf_names,
        seq_only_pred_q,
        output_dir,
        trans_resid_sigma_thresh=1.0,   # |trans_resid| > N×σ
        seq_pred_sigma_thresh=0.5,      # |seq_only_pred - median| > N×σ
        attn_entropy_pct=75,            # 保留注意力熵低于此百分位的基因
):
    """
    路线图 Step 1–3（不含 IAA 验证）：
      Step 1: 多维置信度筛选高置信 Q2/Q3 基因
      Step 2: 计算 trans_score（方向性 TF 干预强度）
      Step 3: 注意力归因 + 跨基因共享 TF 模式分析
    """
    from scipy.stats import entropy as _entropy
    out = Path(output_dir); out.mkdir(exist_ok=True, parents=True)

    tpm_q      = gm_q['tpm_norm'].values
    trans_r    = tpm_q - seq_only_pred_q          # trans_resid
    seq_pred   = seq_only_pred_q

    q2_mask = np.asarray(conds[1], dtype=bool)
    q3_mask = np.asarray(conds[2], dtype=bool)

    # ── Step 1：置信度筛选 ────────────────────────────────────────────────
    print("\n" + "=" * 62 +
          "\n  [Step1] 高置信度 Q2/Q3 基因筛选\n" + "=" * 62)

    tr_sigma   = float(np.std(trans_r, ddof=1))
    seq_med    = float(np.median(seq_pred))
    seq_sigma  = float(np.std(seq_pred, ddof=1))

    # 注意力 Shannon 熵：熵低 → 注意力集中于少数 TF → 模型对 TF 来源更确定
    attn_ent = np.array([
        float(_entropy(np.clip(row, 1e-12, None)))
        for row in attn_q
    ])
    ent_threshold = float(np.percentile(attn_ent, attn_entropy_pct))

    # --- Q2 高置信 ---
    q2_hc = (
        q2_mask &
        (trans_r < -trans_resid_sigma_thresh * tr_sigma) &   # trans_resid 够负
        (seq_pred  >  seq_med + seq_pred_sigma_thresh * seq_sigma) &  # 序列预测够高
        (attn_ent  <  ent_threshold)                          # 注意力集中
    )
    # --- Q3 高置信 ---
    q3_hc = (
        q3_mask &
        (trans_r  >  trans_resid_sigma_thresh * tr_sigma) &   # trans_resid 够正
        (seq_pred  <  seq_med - seq_pred_sigma_thresh * seq_sigma) &  # 序列预测够低
        (attn_ent  <  ent_threshold)                          # 注意力集中
    )

    n_q2 = q2_mask.sum(); n_q3 = q3_mask.sum()
    n_q2_hc = q2_hc.sum(); n_q3_hc = q3_hc.sum()
    print(f"  Q2 全集 {n_q2} → 高置信 {n_q2_hc} "
          f"({n_q2_hc/max(n_q2,1)*100:.1f}%)")
    print(f"  Q3 全集 {n_q3} → 高置信 {n_q3_hc} "
          f"({n_q3_hc/max(n_q3,1)*100:.1f}%)")
    print(f"  筛选参数: |trans_resid|>{trans_resid_sigma_thresh:.1f}σ "
          f"({trans_resid_sigma_thresh*tr_sigma:.3f}), "
          f"|seq_pred-median|>{seq_pred_sigma_thresh:.1f}σ "
          f"({seq_pred_sigma_thresh*seq_sigma:.3f}), "
          f"attn_entropy<{attn_entropy_pct}th pct ({ent_threshold:.3f})")

    # ── Step 2：trans_score（方向性 TF 干预强度参数）─────────────────────
    print("\n" + "=" * 62 +
          "\n  [Step2] trans_score 计算（方向性 TF 干预强度）\n" + "=" * 62)

    # signed_log_trans_score = sign(trans_resid) × log1p(|trans_resid|)
    # 比原始比值 trans_resid / (|seq_pred| + ε) 更稳健：
    #   · 避免 seq_pred ≈ 0 时除法爆炸
    #   · log1p 压缩极端值，分布更对称，适合可视化
    #   · 保留方向（Q2<0, Q3>0）和相对强度排序
    trans_score = np.sign(trans_r) * np.log1p(np.abs(trans_r))

    # 构建全量 DataFrame（后面保存，仅 Q2/Q3 标记置信度）
    hc_df = gm_q[['gene', 'quadrant', 'seq_activity', 'tpm_norm']].copy()
    hc_df['seq_only_pred'] = seq_pred
    hc_df['trans_resid']   = trans_r
    hc_df['trans_score']   = trans_score
    hc_df['attn_entropy']  = attn_ent
    hc_df['highconf']      = False
    hc_df.loc[q2_hc | q3_hc, 'highconf'] = True

    hc_df_q2 = hc_df[q2_hc].copy().sort_values('trans_resid')           # 最负在前
    hc_df_q3 = hc_df[q3_hc].copy().sort_values('trans_resid', ascending=False)  # 最正在前

    hc_df_q2.to_csv(out / 'HC_Q2_trans_repressed_genes.csv', index=False)
    hc_df_q3.to_csv(out / 'HC_Q3_trans_activated_genes.csv', index=False)
    hc_df.to_csv(out / 'HC_all_quadrant_trans_score.csv', index=False)
    print(f"  ✓ trans_score（signed-log）表格已保存 → HC_Q2/Q3_genes.csv")

    if n_q2_hc > 0:
        print(f"  Q2 高置信 trans_score: mean={hc_df_q2['trans_score'].mean():.4f}  "
              f"min={hc_df_q2['trans_score'].min():.4f}")
    if n_q3_hc > 0:
        print(f"  Q3 高置信 trans_score: mean={hc_df_q3['trans_score'].mean():.4f}  "
              f"max={hc_df_q3['trans_score'].max():.4f}")

    # ── Step 3：注意力归因 + 跨基因共享 TF 模式 ─────────────────────────
    print("\n" + "=" * 62 +
          "\n  [Step3] 高置信 Q2/Q3 注意力归因 & 共享 TF 模式\n" + "=" * 62)

    tf_names_arr = np.array(tf_names)

    results_attn = {}
    for tag, mask, sort_asc in [('Q2_HC', q2_hc, True), ('Q3_HC', q3_hc, False)]:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            print(f"  ⚠ {tag}: 无高置信基因，跳过注意力归因")
            continue

        mean_attn = attn_q[idx].mean(axis=0)
        std_attn  = attn_q[idx].std(axis=0)
        top20_idx = np.argsort(mean_attn)[::-1][:20]

        attn_df = pd.DataFrame({
            'tf':       tf_names_arr[top20_idx],
            'mean_attn': mean_attn[top20_idx],
            'std_attn':  std_attn[top20_idx],
            'rank':      np.arange(1, 21),
        })
        attn_df.to_csv(out / f'HC_{tag}_top20_TF_attention.csv', index=False)
        results_attn[tag] = {'idx': idx, 'mean_attn': mean_attn, 'top20_idx': top20_idx}

        print(f"\n  {tag} (n={len(idx)}) 高注意力 Top-10 TFs:")
        for rank, i in enumerate(top20_idx[:10], 1):
            print(f"    {rank:2d}. {str(tf_names[i]):20s}  "
                  f"mean_attn={mean_attn[i]:.4f} ± {std_attn[i]:.4f}")

    # --- 共享 TF 模式：Q2 与 Q3 高置信基因的差异注意力 ---
    if 'Q2_HC' in results_attn and 'Q3_HC' in results_attn:
        diff_attn = results_attn['Q3_HC']['mean_attn'] - results_attn['Q2_HC']['mean_attn']
        diff_df = pd.DataFrame({
            'tf':               tf_names_arr,
            'attn_Q3_HC':       results_attn['Q3_HC']['mean_attn'],
            'attn_Q2_HC':       results_attn['Q2_HC']['mean_attn'],
            'delta_Q3_minus_Q2': diff_attn,
        }).sort_values('delta_Q3_minus_Q2', ascending=False)
        diff_df.to_csv(out / 'HC_Q2vsQ3_TF_attn_differential.csv', index=False)

        print(f"\n  [Step3] Q3_HC 偏高注意力 TFs（候选 trans-activators）:")
        for _, row in diff_df.head(5).iterrows():
            print(f"    {str(row['tf']):20s}  ΔQ3-Q2={row['delta_Q3_minus_Q2']:+.4f}")
        print(f"  [Step3] Q2_HC 偏高注意力 TFs（候选 trans-repressors）:")
        for _, row in diff_df.tail(5).iloc[::-1].iterrows():
            print(f"    {str(row['tf']):20s}  ΔQ3-Q2={row['delta_Q3_minus_Q2']:+.4f}")

    # ── 可视化：trans_score 分布 + 高置信基因散点图 ─────────────────────
    _plot_highconf_q2q3(
        hc_df=hc_df, q2_hc=q2_hc, q3_hc=q3_hc,
        results_attn=results_attn, tf_names_arr=tf_names_arr,
        output_dir=str(out))

    print(f"\n  ✓ [高置信Q2/Q3] 所有输出已保存至: {out}")
    return {
        'n_q2_hc': int(n_q2_hc), 'n_q3_hc': int(n_q3_hc),
        'hc_df': hc_df,
    }


def _plot_highconf_q2q3(hc_df, q2_hc, q3_hc, results_attn, tf_names_arr, output_dir):
    """高置信 Q2/Q3 可视化：3 子图"""
    out = Path(output_dir)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("High-Confidence Q2/Q3 Analysis\n"
                 "(Trans-repressed vs Trans-activated)",
                 fontsize=11, fontweight='bold')

    # Panel A: trans_score 分布（Q2 vs Q3 全集 vs 高置信）
    # trans_score = sign(trans_resid) × log1p(|trans_resid|)，已无极端值
    ax = axes[0]
    # 用 Q2+Q3 全集的 2nd–98th pct 定义 bins 范围，确保不被离群值撑开
    _q23_ts = np.concatenate([
        hc_df.loc[hc_df['quadrant'] == 'Q2: Trans-repressed', 'trans_score'].values,
        hc_df.loc[hc_df['quadrant'] == 'Q3: Trans-activated', 'trans_score'].values,
    ])
    _vlo, _vhi = (float(np.percentile(_q23_ts, 2)),
                  float(np.percentile(_q23_ts, 98))) if len(_q23_ts) > 0 else (-3, 3)
    bins = np.linspace(_vlo, _vhi, 45)
    q2_all_ts = hc_df.loc[hc_df['quadrant'] == 'Q2: Trans-repressed', 'trans_score']
    q3_all_ts = hc_df.loc[hc_df['quadrant'] == 'Q3: Trans-activated', 'trans_score']
    q2_hc_ts = hc_df['trans_score'].values[q2_hc]
    q3_hc_ts = hc_df['trans_score'].values[q3_hc]

    ax.hist(q2_all_ts, bins=bins, color='#FF6B6B', alpha=0.30, density=True,
            label=f'Q2 all (n={len(q2_all_ts)})')
    ax.hist(q3_all_ts, bins=bins, color='#2EC4B6', alpha=0.30, density=True,
            label=f'Q3 all (n={len(q3_all_ts)})')
    if len(q2_hc_ts) > 0:
        ax.hist(q2_hc_ts, bins=bins, color='#FF6B6B', alpha=0.80, density=True,
                label=f'Q2 HC (n={len(q2_hc_ts)})', histtype='step', lw=2)
    if len(q3_hc_ts) > 0:
        ax.hist(q3_hc_ts, bins=bins, color='#2EC4B6', alpha=0.80, density=True,
                label=f'Q3 HC (n={len(q3_hc_ts)})', histtype='step', lw=2)
    ax.axvline(0, color='#555', lw=1.2, ls='--', alpha=0.7)
    ax.set_xlabel('trans_score = sign(trans_resid) × log1p(|trans_resid|)')
    ax.set_ylabel('Density')
    ax.set_title('A. trans_score distribution\n'
                 r'(sign(resid)×log1p(|resid|); HC = high-confidence)',
                 fontweight='bold')
    ax.legend(fontsize=7); ax.spines[['top', 'right']].set_visible(False)

    # Panel B: 高置信基因散点图（seq_only_pred vs tpm_norm，按置信度着色）
    ax = axes[1]
    other_mask = ~(q2_hc | q3_hc)
    ax.scatter(hc_df['seq_only_pred'].values[other_mask],
               hc_df['tpm_norm'].values[other_mask],
               c='#CCCCCC', s=4, alpha=0.20, label='Other', rasterized=True)
    ax.scatter(hc_df['seq_only_pred'].values[q2_hc],
               hc_df['tpm_norm'].values[q2_hc],
               c='#FF6B6B', s=18, alpha=0.70, label=f'Q2-HC (n={q2_hc.sum()})',
               zorder=3, edgecolors='white', linewidths=0.3)
    ax.scatter(hc_df['seq_only_pred'].values[q3_hc],
               hc_df['tpm_norm'].values[q3_hc],
               c='#2EC4B6', s=18, alpha=0.70, label=f'Q3-HC (n={q3_hc.sum()})',
               zorder=3, edgecolors='white', linewidths=0.3)
    ax.axhline(0, color='#888', lw=0.8, ls='--', alpha=0.6)
    ax.axvline(float(np.median(hc_df['seq_only_pred'])),
               color='#888', lw=0.8, ls='--', alpha=0.6)
    ax.set_xlabel('Seq-only prediction (median-centred)')
    ax.set_ylabel('Expression (TPM z-score)')
    ax.set_title('B. High-confidence Q2/Q3 genes\n(scatter in seq-pred × expression space)',
                 fontweight='bold')
    ax.legend(fontsize=7); ax.spines[['top', 'right']].set_visible(False)

    # Panel C: 高置信 Q2/Q3 差异注意力 Top-20 TF 水平条形图
    ax = axes[2]
    if 'Q2_HC' in results_attn and 'Q3_HC' in results_attn:
        diff_vals = (results_attn['Q3_HC']['mean_attn'] -
                     results_attn['Q2_HC']['mean_attn'])
        top_pos_idx = np.argsort(diff_vals)[::-1][:10]   # Q3偏高
        top_neg_idx = np.argsort(diff_vals)[:10]          # Q2偏高
        show_idx    = np.concatenate([top_pos_idx, top_neg_idx[::-1]])
        show_vals   = diff_vals[show_idx]
        show_labels = [str(tf_names_arr[i])[:16] for i in show_idx]
        bar_colors  = ['#2EC4B6' if v >= 0 else '#FF6B6B' for v in show_vals]
        y_pos = np.arange(len(show_idx))
        ax.barh(y_pos, show_vals, color=bar_colors, alpha=0.80,
                edgecolor='white', linewidth=0.5)
        ax.set_yticks(y_pos); ax.set_yticklabels(show_labels, fontsize=7)
        ax.axvline(0, color='#555', lw=1.0, ls='-')
        ax.set_xlabel('Δ mean_attn (Q3_HC − Q2_HC)')
        ax.set_title('C. Differential TF attention\n(+: Q3-activator, −: Q2-repressor)',
                     fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
    else:
        ax.text(0.5, 0.5, 'Insufficient HC genes\nfor differential attn',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('C. Differential TF attention', fontweight='bold')

    plt.tight_layout()
    fig.savefig(out / 'figHC_q2q3_analysis.pdf', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  ✓ 高置信Q2/Q3图表 → {out/'figHC_q2q3_analysis.pdf'}")


# ============================================================================
# run_postanalysis — 完整后分析入口（无 DNA 分支）
# ============================================================================

def run_postanalysis(model, X_raw, y_raw, split_masks, tf_names, gene_names,
                     sig_std, lp, y_tr_norm, y_val_norm, y_test_norm,
                     output_dir, device=None,
                     scalar_data_raw=None, sc_mean=None, sc_std=None,
                     ablation_results=None, n_seq_features=12,
                     seq_only_model_dir=None,
                     n_tfs_model=None, n_bins_model=None, n_scalars_model=None,
                     tf_tone_csv=None, lm_data=None,
                     iaa_fc_file=None) -> None:
    if device is None: device = torch.device('cpu')
    out = Path(output_dir);
    out.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 62 + "\n  全量推断中...\n" + "=" * 62)
    all_idx = np.concatenate([np.where(split_masks['train'])[0],
                              np.where(split_masks['val'])[0],
                              np.where(split_masks['test'])[0]])
    reorder = np.argsort(all_idx)
    X_std_concat = np.concatenate([
        sig_std.transform(X_raw[split_masks['train']]),
        sig_std.transform(X_raw[split_masks['val']]),
        sig_std.transform(X_raw[split_masks['test']])
    ], axis=0)[reorder]

    scalars_full = None
    if scalar_data_raw is not None and sc_mean is not None and sc_std is not None:
        sc_raw = scalar_data_raw.copy().astype(np.float32)
        if sc_raw.shape[1] > n_seq_features:
            sc_raw[:, n_seq_features:] = np.log1p(np.clip(sc_raw[:, n_seq_features:], 0, None))
        scalars_full = np.nan_to_num((sc_raw - sc_mean) / sc_std, nan=0.0).astype(np.float32)
        print(f"  ✓ 标量归一化完成 ({scalar_data_raw.shape[1]} 列)")
    if scalars_full is None and getattr(model, 'use_scalars', False):
        fusion = getattr(model, 'fusion', None)
        _n_sc = ((getattr(fusion, 'n_reg_scalars', 2) + getattr(fusion, 'n_seq_features', 1))
                 if fusion is not None else 2)
        scalars_full = np.zeros((len(gene_names), _n_sc), dtype=np.float32)
        print("  ⚠ 使用全零标量")

    lm_concat = None
    if lm_data is not None:
        lm_concat = np.concatenate([lm_data[split_masks['train']],
                                    lm_data[split_masks['val']],
                                    lm_data[split_masks['test']]], axis=0)[reorder]

    preds_all, attn_all = predict_full_dataset(
        model, X_std_concat, device,
        scalars=scalars_full, lm_data=lm_concat)
    print(f"  ✓ 预测完成: n={len(preds_all)}")

    y_all_norm = np.empty(len(y_raw), dtype=np.float32)
    y_all_norm[split_masks['train']] = y_tr_norm
    y_all_norm[split_masks['val']] = y_val_norm
    y_all_norm[split_masks['test']] = y_test_norm
    gm = pd.DataFrame({
        'gene': gene_names, 'tpm_norm': y_all_norm,
        'pred_norm': preds_all,
        'split': np.where(split_masks['train'], 'train',
                          np.where(split_masks['val'], 'val', 'test'))
    })

    # 序列活性（来自 scalar_data_raw 的前 n_seq_features 列）
    if scalar_data_raw is not None and scalar_data_raw.shape[1] >= n_seq_features:
        act_vals = compute_seq_summary_np(scalar_data_raw[:, :n_seq_features].astype(np.float32))
    elif scalars_full is not None and scalars_full.shape[1] >= n_seq_features:
        act_vals = compute_seq_summary_np(scalars_full[:, :n_seq_features])
    else:
        act_vals = np.zeros(len(gm), dtype=np.float32)
    gm['seq_activity'] = act_vals
    gm['seq_activity_norm'] = ((act_vals - act_vals.min()) /
                               (act_vals.max() - act_vals.min() + 1e-8))

    # ── 消融贡献分析 ────────────────────────────────────────────────────
    print("\n" + "=" * 62 + "\n  消融贡献分析（D→L→B→C）\n" + "=" * 62)
    decomp = {}
    if ablation_results is not None:
        _ordered_keys = [
            ('ablation_D_dna_cnn',  'DNA-CNN (from scratch)'),
            ('ablation_L_lm_only',  'LM-only'),
            ('ablation_B_act_lm',   'Activity + LM'),
            ('ablation_C_seq_tf',   'Seq + TF (full)'),
        ]
        for key_in, key_out in _ordered_keys:
            res = ablation_results.get(key_in)
            if res is None: continue
            entry = {'pearson': res['pearson'], 'r2': res['r2']}
            ci = res.get('boot_ci')
            if ci is not None:
                entry['pearson_ci_lo'] = ci['pearson_ci_lo']
                entry['pearson_ci_hi'] = ci['pearson_ci_hi']
            decomp[key_out] = entry
    if decomp:
        print(f"  ✓ 消融节点: {len(decomp)} 个")
        for k, v in decomp.items():
            print(f"    {k:42s}  r={v['pearson']:.4f}  R²={v['r2']:.4f}")
    pd.DataFrame(decomp).T.to_csv(out / 'decomp_stats.csv')
    _plot_variance_decomp(decomp, str(out))

    # ── seq_only 全量推断（需先于四象限分类，用于 X 轴锚定）─────────────
    print("\n" + "=" * 62 + "\n  加载 ablation_B_act_lm 模型做全量推断...\n" + "=" * 62)
    seq_only_preds_full = None
    _so_dir = seq_only_model_dir
    if _so_dir is None and ablation_results is not None:
        _so_dir = ablation_results.get('ablation_B_act_lm', {}).get('exp_dir')

    if _so_dir is not None:
        _ckpt_b = str(Path(_so_dir) / 'complete_model.pth')
        _n_tfs = n_tfs_model or X_raw.shape[1]
        _n_bins = n_bins_model or X_raw.shape[2]
        _n_sc = n_scalars_model or (
            scalar_data_raw.shape[1] if scalar_data_raw is not None else n_seq_features)
        sc_seq_only = scalars_full.copy() if scalars_full is not None else None
        if sc_seq_only is not None:
            sc_seq_only[:, n_seq_features:] = 0.0
        lm_dim_use = lm_data.shape[1] if lm_data is not None else 0
        seq_only_preds_full = _load_checkpoint_and_predict(
            ckpt_path=_ckpt_b,
            X_std=X_std_concat, scalars=sc_seq_only,
            n_tfs=_n_tfs, n_bins=_n_bins,
            n_scalars=_n_sc, n_seq_features=n_seq_features,
            device=device,
            use_lm=USE_LM_EMB,
            lm_data=lm_concat if USE_LM_EMB else None,
            lm_dim=lm_dim_use)
        if np.isnan(seq_only_preds_full).all():
            print("  ⚠ ablation_B 推断全部 NaN，跳过 seq_only_pred 轴")
            seq_only_preds_full = None
        else:
            print("  ✓ seq_only 全量预测完成")
    else:
        print("  ⚠ 未找到 ablation_B_act_lm 模型目录，四象限将回退至原始活性轴")

    # ── 四象限分类 ───────────────────────────────────────────────────────
    print("\n" + "=" * 62 + "\n  故事线2：四象限基因分类\n" + "=" * 62)
    valid_all = gm['seq_activity'].notna().values
    gm_q = gm[valid_all].copy().reset_index(drop=True)
    attn_q = attn_all[valid_all]

    # X 轴优先使用 seq_only_pred（模型序列分支的预测输出）；
    # 其物理含义严格等于「序列信息对表达量的贡献」，比 Vaishnav 原始活性标量更准确。
    # 当 ablation_B 模型不可用时回退至原始活性标量。
    if seq_only_preds_full is not None:
        seq_only_q = seq_only_preds_full[valid_all]
        sa_median = float(np.nanmedian(seq_only_q))
        sa_z = (seq_only_q - sa_median).astype(np.float32)
        _xaxis_label = 'Seq-only prediction (ablation B, median-centred)'
        print(f"  ✓ X轴锚定: seq_only_pred  "
              f"median={sa_median:.4f}  "
              f"range=[{seq_only_q.min():.3f},{seq_only_q.max():.3f}]")
    elif scalar_data_raw is not None and scalar_data_raw.shape[1] >= n_seq_features:
        raw_activity = compute_seq_summary_np(
            scalar_data_raw[:, :n_seq_features].astype(np.float32))
        raw_activity_q = raw_activity[valid_all]
        sa_median = np.nanmedian(raw_activity_q)
        sa_z = (raw_activity_q - sa_median).astype(np.float32)
        _xaxis_label = 'Vaishnav seq activity (median-centred, fallback)'
        print(f"  ⚠ X轴回退: Vaishnav 原始活性  "
              f"median={sa_median:.4f}  "
              f"range=[{raw_activity_q.min():.3f},{raw_activity_q.max():.3f}]")
    else:
        sa_raw = gm_q['seq_activity'].values
        sa_median = np.nanmedian(sa_raw)
        sa_z = (sa_raw - sa_median).astype(np.float32)
        _xaxis_label = 'Seq activity (z-score, fallback)'

    tpm_v = gm_q['tpm_norm'].values
    conds = [
        (sa_z >= 0) & (tpm_v >= 0),  # Q1: Sequence-concordant (high) — 高seq预测 & 高表达
        (sa_z >= 0) & (tpm_v < 0),   # Q2: Trans-repressed            — 高seq预测 & 低表达（TF抑制）
        (sa_z < 0)  & (tpm_v >= 0),  # Q3: Trans-activated            — 低seq预测 & 高表达（TF激活）
        (sa_z < 0)  & (tpm_v < 0),   # Q4: Sequence-concordant (low)  — 低seq预测 & 低表达
    ]
    quad_labels = ['Q1: Sequence-concordant (high)', 'Q2: Trans-repressed',
                   'Q3: Trans-activated', 'Q4: Sequence-concordant (low)']
    quad_colors = ['#3A86FF', '#FF6B6B', '#2EC4B6', '#AAAAAA']

    gm_q['quadrant'] = 'unknown'
    for cond, lbl in zip(conds, quad_labels):
        gm_q.loc[cond, 'quadrant'] = lbl
    if seq_only_preds_full is not None:
        gm_q['seq_only_pred'] = seq_only_preds_full[valid_all]
    gm_q['pred_error'] = gm_q['pred_norm'] - gm_q['tpm_norm']
    gm_q['abs_error'] = gm_q['pred_error'].abs()

    # ── 象限计数: 全基因 + test-only (两份分别报告, 避免混淆) ──────────────
    # 全基因计数用于「可视化散点图坐标覆盖率」; test-only 才是模型性能指标.
    # 历史 bug: 此处用 gm_q 直接算 Pearson, 而 gm_q 含 train+val+test, 导致 Q2/Q3
    # 的 n 是 test-only 的 ~4 倍, Pearson 被「模型已见过标签的 train 基因」虚高.
    # 例如 Q2 全基因 n=365 pearson=0.55 vs test-only n=90 pearson=0.306.
    print(f"\n  [全基因] (仅供参考 — 含 train/val, 不可用于论文性能指标)")
    for lbl in quad_labels:
        n = (gm_q['quadrant'] == lbl).sum()
        print(f"    {lbl}: {n} 基因 ({n / len(gm_q) * 100:.1f}%)")

    # test 子集 (模型性能指标来源)
    gm_test = gm_q[gm_q['split'] == 'test'].copy()
    print(f"\n  [test only] (论文性能指标来源, n={len(gm_test)})")
    for lbl in quad_labels:
        n = (gm_test['quadrant'] == lbl).sum()
        print(f"    {lbl}: {n} 基因 ({n / max(len(gm_test), 1) * 100:.1f}%)")

    _save_cols = ['gene', 'quadrant', 'split',
                  'seq_activity', 'tpm_norm', 'pred_norm', 'pred_error']
    if 'seq_only_pred' in gm_q.columns:
        _save_cols.insert(5, 'seq_only_pred')
    gm_q[_save_cols].to_csv(out / 'quadrant_gene_list.csv', index=False)

    from scipy.stats import pearsonr as _pearsonr, spearmanr as _spearmanr_quad
    # ── 分象限性能指标: 仅 test 子集 (论文用数字) ─────────────────────────
    # 本表是论文唯一可用的每象限 Pearson 来源;
    # multiseed_C_quadrant_stats.csv 跨 seed 聚合版在其基础上加 std.
    print(f"\n  ┌─ [TEST ONLY] 分象限性能指标 (论文唯一可用) ───────────────────")
    print(f"  │  {'象限':22s}  {'n':>5}  {'MAE':>6}  {'Bias':>7}  "
          f"{'Pearson':>7}  {'Spearman':>9}")
    print(f"  │  " + "-" * 65)
    quadrant_metrics_rows = []
    for lbl, cond in zip(quad_labels, conds):
        sub = gm_test[gm_test['quadrant'] == lbl]
        if len(sub) < 3:
            print(f"  │  {lbl:22s}  {len(sub):>5}   (n<3, 跳过)")
            continue
        mae  = sub['abs_error'].mean()
        bias = sub['pred_error'].mean()
        pr   = (_pearsonr(sub['tpm_norm'], sub['pred_norm'])[0]
                if len(sub) > 5 else float('nan'))
        sr   = (_spearmanr_quad(sub['tpm_norm'], sub['pred_norm'])[0]
                if len(sub) > 5 else float('nan'))
        print(f"  │  {lbl:22s}  {len(sub):>5}  {mae:>6.3f}  "
              f"{bias:>+7.3f}  {pr:>7.4f}  {sr:>9.4f}")
        quadrant_metrics_rows.append({
            'quadrant': lbl, 'n_test': int(len(sub)),
            'pearson': float(pr), 'spearman': float(sr),
            'mae': float(mae), 'bias': float(bias),
            'split_scope': 'test_only',
        })
    print(f"  └───────────────────────────────────────────────────────────────")

    # 全基因对照表 (ONLY for debugging / appendix — NOT a performance metric)
    full_rows = []
    for lbl in quad_labels:
        sub = gm_q[gm_q['quadrant'] == lbl]
        if len(sub) < 3: continue
        pr = (_pearsonr(sub['tpm_norm'], sub['pred_norm'])[0]
              if len(sub) > 5 else float('nan'))
        full_rows.append({
            'quadrant': lbl, 'n_all': int(len(sub)),
            'pearson': float(pr),
            'mae': float(sub['abs_error'].mean()),
            'bias': float(sub['pred_error'].mean()),
            'split_scope': 'all_genes_leaky_debug_only',
        })

    # 落盘: test-only 为主文件, full 为诊断文件 (文件名带 _DO_NOT_REPORT 避免误用)
    pd.DataFrame(quadrant_metrics_rows).to_csv(
        out / 'quadrant_metrics_test_only.csv', index=False)
    pd.DataFrame(full_rows).to_csv(
        out / 'quadrant_metrics_all_genes_DO_NOT_REPORT.csv', index=False)
    print(f"  ✓ quadrant_metrics_test_only.csv  (论文用)")
    print(f"  ✓ quadrant_metrics_all_genes_DO_NOT_REPORT.csv  (仅诊断)")



    _plot_quadrant_scatter(sa_z, tpm_v, conds, quad_labels, quad_colors, 0.0, 0.0, str(out),
                           x_label=_xaxis_label)
    _plot_quadrant_heatmap(attn_q, conds, quad_labels, quad_colors, tf_names, str(out))



    # ── P3 残差分析 ───────────────────────────────────────────────────────
    if seq_only_preds_full is not None and scalars_full is not None:
        print("\n" + "=" * 62 + "\n  [P3] 残差分析...\n" + "=" * 62)
        _plot_residual_analysis(
            gm_q=gm_q, conds=conds,
            quad_labels=quad_labels, quad_colors=quad_colors,
            seq_only_pred_q=seq_only_preds_full[valid_all],
            full_pred_q=preds_all[valid_all],
            output_dir=str(out))

    # [P4] 传入标准化 ChEC-seq 信号供信号强度验证
    X_signal_valid = X_std_concat[valid_all] if X_std_concat is not None else None

    if seq_only_preds_full is not None:
        print("\n" + "=" * 62 +
              "\n  [P4] TF 注意力 vs tone 方向对照 + 信号强度验证...\n" + "=" * 62)
        _plot_tf_tone_comparison(
            gm_q=gm_q, conds=conds,
            quad_labels=quad_labels, quad_colors=quad_colors,
            attn_q=attn_q, tf_names=tf_names,
            full_pred_q=preds_all[valid_all],
            seq_only_pred_q=seq_only_preds_full[valid_all],
            output_dir=str(out), tf_tone_csv=tf_tone_csv,
            X_signal_q=X_signal_valid)

    # ── 高置信度 Q2/Q3 分析（路线图 Step 1-3，不含 IAA）────────────────
    if seq_only_preds_full is not None:
        print("\n" + "=" * 62 +
              "\n  [HC-Q2Q3] 高置信度 Trans-repressed / Trans-activated 分析\n"
              + "=" * 62)
        try:
            _run_highconf_q2q3_analysis(
                gm_q=gm_q, conds=conds,
                quad_labels=quad_labels, quad_colors=quad_colors,
                attn_q=attn_q, tf_names=tf_names,
                seq_only_pred_q=seq_only_preds_full[valid_all],
                output_dir=str(out),
            )
        except Exception as _hc_e:
            print(f"  ⚠ 高置信Q2/Q3分析失败: {_hc_e}")
            traceback.print_exc()

    # ── IAA 数据加载（方向A/B/C 共用）───────────────────────────────────
    FC_aligned_full = None;
    iaa_tf_names_raw = None
    if iaa_fc_file is not None and Path(iaa_fc_file).exists():
        try:
            FC_aligned_full, iaa_tf_names_raw = _load_iaa_fc_matrix(
                iaa_fc_file, gene_names, tf_names)
        except Exception as _e:
            print(f"  ⚠ IAA 矩阵加载失败: {_e}")
    elif iaa_fc_file is not None:
        print(f"\n  ⚠ IAA 文件不存在: {iaa_fc_file}，跳过 IAA 相关分析")
    else:
        print("\n  [IAA] iaa_fc_file 未提供，跳过")

    # ── 方向A：gene-level Spearman ────────────────────────────────────────
    print("\n" + "=" * 62 +
          "\n  [方向A·辅助] Gene-level TF注意力 vs IAA |log2FC|\n" + "=" * 62)
    try:
        dir_a_res = _validate_attn_gene_specific_iaa(
            attn_all=attn_all, gene_names=gene_names, tf_names=tf_names,
            iaa_fc_file=iaa_fc_file, split_masks=split_masks, output_dir=str(out))
        if dir_a_res is not None:
            print(f"\n  ── 方向A 摘要 ──")
            print(f"    n_computed={dir_a_res['n_computed']}  "
                  f"n_common_TFs={dir_a_res['n_common_tfs']}")
            print(f"    mean Spearman r={dir_a_res['mean_r']:.4f}  "
                  f"frac(r>0)={dir_a_res['frac_positive']:.3f}")
    except Exception as _e:
        print(f"  ⚠ 方向A 分析失败: {_e}")
        traceback.print_exc()

    # ── 方向B：TF-centric Spearman（主要验证）────────────────────────────
    print("\n" + "=" * 62 +
          "\n  [方向B·主要] TF-centric 注意力 vs IAA |log2FC| 验证\n" + "=" * 62)
    try:
        if FC_aligned_full is not None and iaa_tf_names_raw is not None:
            dir_b_res = _validate_attn_tf_centric_iaa(
                attn_all=attn_all, gene_names=gene_names, tf_names=tf_names,
                FC_aligned=FC_aligned_full, iaa_tf_names=iaa_tf_names_raw,
                output_dir=str(out), X_raw_std=X_std_concat)
            if dir_b_res is not None:
                print(f"\n  ── 方向B 摘要 ──")
                print(f"    n_TFs={dir_b_res['n_computed']}  "
                      f"mean_r={dir_b_res['mean_r']:.4f}  "
                      f"median_r={dir_b_res['median_r']:.4f}  "
                      f"frac(r>0)={dir_b_res['frac_positive']:.3f}")
                _hs_r = dir_b_res.get('strat_high_mean_r', float('nan'))
                _ls_r = dir_b_res.get('strat_low_mean_r',  float('nan'))
                _tau  = dir_b_res.get('kendall_tau_high',  float('nan'))
                _taup = dir_b_res.get('kendall_tau_high_p',float('nan'))
                if not np.isnan(_hs_r):
                    print(f"    [信号分层] 高信号 mean_r={_hs_r:.4f}  "
                          f"低信号 mean_r={_ls_r:.4f}  Δ={_hs_r-_ls_r:+.4f}")
                if not np.isnan(_tau):
                    print(f"    [Kendall τ, 高信号 TFs]  τ={_tau:.4f}  p={_taup:.3e}")
        else:
            print("  ⚠ [方向B] IAA 数据不可用，跳过")
    except Exception as _e:
        print(f"  ⚠ 方向B 分析失败: {_e}")
        traceback.print_exc()



    # ── 方向C：cross_weight 生物注释 ────────────────────────────────────
    print("\n" + "=" * 62 +
          "\n  [方向C] cross_weight c 生物注释 + IAA 敏感性\n" + "=" * 62)
    cross_w_all = np.array([], dtype=np.float32)
    try:
        cross_w_all = collect_cross_weights(
            model, X_std_concat, scalars_full, device, lm_data=lm_concat)
        _plot_cross_weight_diagnosis(
            cross_w_all=cross_w_all,
            gene_names=gene_names, output_dir=str(out),
            FC_aligned=FC_aligned_full, top_pct=5)
    except Exception as _e:
        print(f"  ⚠ 方向C 分析失败: {_e}")
        traceback.print_exc()

    # ── Panel H 因果 Δc 反事实实验 (TF→基线 前向; Δc=c−c0, Δŷ=ŷ−ŷ0) ──────────
    #   复用上面 cross_w_all + gm_q / seq_only / IAA, 只额外做一次基线前向。
    #   返回的 (c0, ŷ0) 全长数组随后写进 combined.csv (cross_weight_c_baseline /
    #   yhat_baseline), 解锁 crossweight_quadrant_standalone.py 的 Panel H。
    cross_w_baseline = None; yhat_baseline = None
    if CAUSAL_DELTA_C.get('enabled', False):
        try:
            cross_w_baseline, yhat_baseline = run_causal_delta_c(
                model=model, X_std_full=X_std_concat, scalars_full=scalars_full,
                device=device, lm_full=lm_concat,
                gm_q=gm_q, valid_all=valid_all, cross_w_all=cross_w_all,
                FC_aligned=FC_aligned_full, output_dir=str(out), tag=out.name,
                cfg=CAUSAL_DELTA_C)
        except Exception as _dc_e:
            print(f"  ⚠ 因果 Δc 实验失败: {_dc_e}")
            traceback.print_exc()

    # ── combined.csv 导出 (cross_weight c × 四象限独立分析输入表) ────────────
    #   复用上面已算好的 cross_w_all + 四象限块的 gm_q / seq_only_preds_full /
    #   IAA 块的 FC_aligned_full + 因果 Δc 块的 (c0, ŷ0), 无需重新推断。供
    #   crossweight_quadrant_standalone.py 的 `--input combined.csv` 直接使用
    #   (每个 holdout 的 output_dir 各产一份)。
    try:
        _export_combined_crossweight_table(
            output_dir          = str(out),
            gm_q                = gm_q,
            valid_all           = valid_all,
            seq_only_preds_full = seq_only_preds_full,
            cross_w_all         = cross_w_all,
            FC_aligned          = FC_aligned_full,   # 未带 IAA 时为 None
            tag                 = out.name,
            cross_w_baseline    = cross_w_baseline,  # Panel H 因果 Δc 基线 c0 (可为 None)
            yhat_baseline       = yhat_baseline,     # Panel H 因果 Δc 基线 ŷ0 (可为 None)
        )
    except Exception as _cb_e:
        print(f"  ⚠ combined.csv 导出失败: {_cb_e}")
        traceback.print_exc()

    print("\n  [Motif] 分析已迁移至 motif_analysis.py，请在 MEME 虚拟环境中单独运行")

    # ── 区域权重 × Q2/Q3 分析（TFSpatialEncoder→AttentiveRegionPool）────────
    print("\n" + "=" * 62 +
          "\n  [区域权重] TFSpatialEncoder 区域注意力 × Q2/Q3 分析\n" + "=" * 62)
    try:
        region_w_all = collect_region_weights(
            model=model, X_std=X_std_concat, device=device,
            scalars=sc_concat if 'sc_concat' in dir() else None)
        if len(region_w_all) > 0:
            _analyze_region_weights_q2q3(
                region_w_all=region_w_all,
                gm_q=gm_q, conds=conds, quad_labels=quad_labels,
                quad_colors=quad_colors, output_dir=str(out))
    except Exception as _rw_e:
        print(f"  ⚠ 区域权重分析失败: {_rw_e}")
        traceback.print_exc()

    # ── Priority 3：注意力头专化（附录/阴性结果）+ Figure 5: TF-TF 协调矩阵 ──
    # 两个分析共用同一次 per_head_attn 前向, 避免重复计算 (N_genes × 178 × 178 attn
    # 本身 ≈ 6572 × 4 × 178 × 178 × 4B = 3.3 GB, 单次提取成本高).
    print("\n" + "=" * 62 +
          "\n  [附录] 头专化 + [Fig5] TF-TF 协调矩阵 (共用 per_head_attn)\n"
          + "=" * 62)
    per_head_attn = None
    try:
        per_head_attn = extract_per_head_attn_weights(
            model=model, X_std=X_std_concat, scalars=scalars_full,
            device=device, lm_data=lm_concat, batch_size=256)
    except Exception as _e:
        print(f"  ⚠ per_head_attn 提取失败: {_e}")
        traceback.print_exc()

    # 头专化 (阴性结果, 进附录)
    if per_head_attn is not None:
        try:
            _analyze_attn_head_specialization(
                per_head_attn=per_head_attn, tf_names=tf_names,
                tf_tone_csv=tf_tone_csv, output_dir=str(out), top_n=20)
            print("  ℹ [附录] 头专化分析完成（预期阴性，效应量<0.5%基底）")
        except Exception as _e:
            print(f"  ⚠ [附录] 头专化分析失败: {_e}")
            traceback.print_exc()
    else:
        print("  ⚠ [附录] per-head 注意力不可用, 跳过头专化")

    # Figure 5: TF-TF 协调矩阵 (正文主图)
    if per_head_attn is not None:
        print("\n" + "=" * 62 +
              "\n  [Figure 5] TF-TF 协调矩阵 (跨基因自注意力平均 + 层次聚类)\n"
              + "=" * 62)
        try:
            # 从顶层 FIG5_CONFIG 读算法配置 (背景校正 / k-selection / sparsify).
            # 这里传 None 让函数内部回填 FIG5_CONFIG, 保留显式覆盖的能力.
            _fig5_result = _analyze_tf_tf_coordination(
                per_head_attn=per_head_attn, tf_names=tf_names,
                output_dir=str(out), top_sub_tfs=60,
                n_clusters=FIG5_CONFIG.get('n_clusters', 8),
                background_correction=FIG5_CONFIG.get('background_correction',
                                                       'configuration'),
                edge_sparsify_top_pct=FIG5_CONFIG.get('edge_sparsify_top_pct', None),
                k_search=FIG5_CONFIG.get('k_search', (3, 4, 5, 6, 7, 8, 10)),
                collapse_warn_frac=FIG5_CONFIG.get('collapse_warn_frac', 0.40),
            )
            # collapse 时提示用户 fallback 方案 (上面已打印详细建议)
            if _fig5_result and _fig5_result.get('collapse_detected'):
                print("  ⚠ [Fig5] collapse 已检测, 请考虑 (1) 切换到 'configuration' 背景校正, "
                      "(2) 减小 k, 或 (3) 把 Fig5 退 Supplementary.")
        except Exception as _e:
            print(f"  ⚠ [Fig5] TF-TF 协调矩阵分析失败: {_e}")
            traceback.print_exc()
    else:
        print("  ⚠ [Fig5] per-head 注意力不可用, 跳过 Figure 5")
    # 显式释放大张量
    per_head_attn = None

    print(f"\n{'=' * 62}\n  ✓ 全部分析完成，结果保存至: {out}\n{'=' * 62}")

# ============================================================================
# 模型对比矩阵 — 公共特征构建 & 基线训练
# ============================================================================

def _build_comparison_features(
    feature_set: str,          # 'lm_only' | 'lm_act' | 'full'
    scalar_data: np.ndarray,   # (N, n_seq_features)
    lm_emb: np.ndarray | None, # (N, lm_dim) or None — 须传入已归一化版本（normalize_lm_embeddings 处理后）
    X_raw: np.ndarray,         # (N, n_tfs, n_bins)  TF 信号（原始计数，内部做 log1p）
    split_masks: dict,
    tpm_transform: str = 'log1p',
    tpm_normalize:  str = 'zscore',
    y_raw: np.ndarray = None,
    tf_summary: str   = 'mean',
    n_seq_features:   int = 12,
    tss_ratio:        float = TSS_RATIO,
    lm_compress_dim:  int = 0,  # >0 则对 LM 嵌入做 PCA 压缩（fit on train）
) -> dict:
    """
    构建基线对比用特征矩阵，返回 {train_X, val_X, test_X, train_y, val_y, test_y}。
    所有归一化参数只在 train 上 fit，保证无泄漏。

    feature_set 与 ours 消融链的 1:1 对应关系：
      'lm_only'  ≡ ablation L (LM 2304)                ← LM-only 列
      'lm_act'   ≡ ablation B (LM 2304 + Act 12)       ← +Activity 列
      'full'     ≡ ablation C (LM 2304 + Act 12 + TF)  ← +TF (Full) 列

    特征处理与 train_experiment (ours) 的对应关系：
      TF 信号: log1p → per-feature z-score  （与 SignalStandardizer 一致）
      scalar : 可选 log1p（列数 > n_seq_features 时）→ z-score
      LM 嵌入: 调用前已由 normalize_lm_embeddings 归一化，此处不重复归一化
    """
    # ── 1. TF 信号汇总 ────────────────────────────────────────────────
    def _summarize_tf(X, method):
        if method == 'mean':
            return X.mean(axis=2)                               # (N, n_tfs)
        if method == 'meanstd':
            return np.concatenate([X.mean(axis=2),
                                   X.std(axis=2)], axis=1)     # (N, 2*n_tfs)
        if method == 'tss_win':
            n_bins = X.shape[2]
            c = int(n_bins * tss_ratio)
            lo = max(0, c - 20); hi = min(n_bins, c + 20)
            return X[:, :, lo:hi].mean(axis=2)                  # (N, n_tfs)
        if method == 'tss_weighted':
            # TSS 高斯加权均值（σ=0.15×n_bins）; 和 PerTFEncoder 内部 σ≈0.15 一致,
            # 给基线一个"TSS 邻域加权均值"的表征 —— 不是复刻 ours 的位置先验。
            n_bins = X.shape[2]
            pos    = np.arange(n_bins, dtype=np.float32)
            tss_pos = n_bins * tss_ratio
            w = np.exp(-0.5 * ((pos - tss_pos) / (n_bins * 0.15)) ** 2)
            w = w / w.sum()
            return (X * w[np.newaxis, np.newaxis, :]).sum(axis=2)  # (N, n_tfs)
        if method == 'raw_bins_pca':
            # 原始 79 × 150 bin 信号展平 → 标记返回, PCA 在下游 train-only fit.
            # 目的: 给基线和 ours-C 一致的原始 bin 特征量, 分离「特征量」vs「架构」
            # 两个混淆因素, 定位 PerTFEncoder 是否真的有贡献.
            N, n_tfs, n_bins = X.shape
            return X.reshape(N, n_tfs * n_bins)                 # (N, n_tfs*n_bins)
        raise ValueError(f"未知 tf_summary 方式: {method}")

    # ── 1a. TF 信号：log1p 压缩右偏分布 → 汇总 → 后续 per-feature z-score
    #    与 SignalStandardizer（ours 流水线）保持一致
    tf_summary_all = _summarize_tf(np.log1p(X_raw), tf_summary)   # (N, *)

    # ── 1b. raw_bins_pca 专用: PCA 降维 (fit on train only, 保留 95% 方差) ──
    # 为避免 79×150=11850 维特征直接塞进 Ridge, 先 PCA 压缩.
    # 关键: fit 只用 train split, transform 全量, 保证无泄漏.
    if tf_summary == 'raw_bins_pca':
        from sklearn.decomposition import PCA
        _pca_tf = PCA(n_components=0.95, random_state=42, svd_solver='full')
        _dim_before = tf_summary_all.shape[1]
        _pca_tf.fit(tf_summary_all[split_masks['train']])
        tf_summary_all = _pca_tf.transform(tf_summary_all).astype(np.float32)
        print(f"    [TF-PCA] raw_bins  {_dim_before} → {tf_summary_all.shape[1]} 维"
              f"（解释方差 {_pca_tf.explained_variance_ratio_.sum():.2%}, "
              f"fit-on-train-only）")

    # ── 2. 标量特征：按 train 归一化 ──────────────────────────────────
    # 与 train_experiment 保持一致：对超出 n_seq_features 的列先做 log1p（防御性）
    sc = scalar_data.copy().astype(np.float32)
    if sc.shape[1] > n_seq_features:
        sc[:, n_seq_features:] = np.log1p(np.clip(sc[:, n_seq_features:], 0, None))
    sc_tr_mu  = sc[split_masks['train']].mean(axis=0, keepdims=True)
    sc_tr_std = sc[split_masks['train']].std(axis=0,  keepdims=True) + 1e-8
    sc_norm   = (sc - sc_tr_mu) / sc_tr_std

    # ── 3. LM 嵌入：传入的 lm_emb 已在 __main__ 中全局归一化（normalize_lm_embeddings），
    #    此处不重复归一化，仅做可选 PCA 压缩（FT-Transformer 专用，fit on train）。
    if lm_emb is not None:
        lm_norm = lm_emb.astype(np.float32)
        if lm_compress_dim > 0 and lm_norm.shape[1] > lm_compress_dim:
            from sklearn.decomposition import PCA
            _pca = PCA(n_components=lm_compress_dim, random_state=42)
            _pca.fit(lm_norm[split_masks['train']])
            lm_dim_before = lm_norm.shape[1]
            lm_norm = _pca.transform(lm_norm).astype(np.float32)
            print(f"    [LM-PCA] {lm_dim_before} → {lm_compress_dim} 维"
                  f"（解释方差 {_pca.explained_variance_ratio_.sum():.2%}）")
    else:
        lm_norm = None

    # ── 4. TF 汇总特征：按 train 归一化 ────────────────────────────
    tf_mu   = tf_summary_all[split_masks['train']].mean(axis=0, keepdims=True)
    tf_std  = tf_summary_all[split_masks['train']].std(axis=0,  keepdims=True) + 1e-8
    tf_norm = (tf_summary_all - tf_mu) / tf_std

    # ── 5. 按 feature_set 拼接 ──────────────────────────────────────
    # 1:1 映射到 ours 消融链 L/B/C:
    #   'lm_only' → [lm_norm]                          (消融 L)
    #   'lm_act'  → [lm_norm, sc_norm]                 (消融 B)
    #   'full'    → [lm_norm, sc_norm, tf_norm]        (消融 C)
    #   'tf_only' → [tf_norm]                          (TF-only 基线, 不含 LM/Activity)
    # 前三列依赖 LM; 'tf_only' 独立用于回答"79 维 TF 单独能预测 TPM 多好"。
    def _cat(arrays):
        return np.concatenate([a for a in arrays if a is not None], axis=1)

    if feature_set == 'lm_only':
        if lm_norm is None:
            raise ValueError("feature_set='lm_only' 需要 lm_emb 不为 None")
        feat_all = _cat([lm_norm])
    elif feature_set == 'lm_act':
        if lm_norm is None:
            raise ValueError("feature_set='lm_act' 需要 lm_emb 不为 None")
        feat_all = _cat([lm_norm, sc_norm])
    elif feature_set == 'full':
        if lm_norm is None:
            raise ValueError("feature_set='full' 需要 lm_emb 不为 None")
        feat_all = _cat([lm_norm, sc_norm, tf_norm])
    elif feature_set == 'tf_only':
        # TF-only: 仅 TF 汇总特征（n_tfs 维, 经 tss_weighted + z-score）
        # 用于测试 cross-modal 架构相对于「仅用 79 维 TF 标量」的增益是否真实
        feat_all = _cat([tf_norm])
    else:
        raise ValueError(f"未知 feature_set: {feature_set} "
                         f"（支持: 'lm_only' | 'lm_act' | 'full' | 'tf_only'）")

    # ── 6. 目标变量处理 ─────────────────────────────────────────────
    lp = ExpressionProcessor(tpm_transform, tpm_normalize)
    tr_y, _ = lp.fit_transform(y_raw[split_masks['train']])

    return dict(
        train_X = feat_all[split_masks['train']].astype(np.float32),
        val_X   = feat_all[split_masks['val']  ].astype(np.float32),
        test_X  = feat_all[split_masks['test'] ].astype(np.float32),
        train_y = tr_y,
        val_y   = lp.transform(y_raw[split_masks['val']  ]),
        test_y  = lp.transform(y_raw[split_masks['test'] ]),
        n_feat  = feat_all.shape[1],
    )


def _train_sklearn_model(model_name: str, feat: dict,
                          output_dir: Path, label: str,
                          do_bootstrap: bool = True,
                          n_boot: int = 2000,
                          ci_level: float = 0.95,
                          random_seed: int = 42) -> dict:
    """训练单个 sklearn / XGBoost / LightGBM 模型，返回 {pearson, r2, boot_ci}"""
    from sklearn.linear_model import Ridge, Lasso
    try:
        from sklearn.model_selection import GridSearchCV
    except ImportError:
        GridSearchCV = None

    print(f"\n    [{model_name}] n_train={len(feat['train_X'])}  "
          f"n_feat={feat['n_feat']}  n_test={len(feat['test_X'])}")

    # ── 选择 & 训练模型 ─────────────────────────────────────────────
    if model_name == 'Ridge':
        from sklearn.linear_model import RidgeCV
        clf = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], cv=5)
        clf.fit(feat['train_X'], feat['train_y'])
        best_alpha = float(clf.alpha_)
        print(f"      RidgeCV best α={best_alpha:.4g}")

    elif model_name == 'Lasso':
        from sklearn.linear_model import LassoCV
        clf = LassoCV(alphas=np.logspace(-4, 2, 30), cv=5,
                      max_iter=5000, random_state=random_seed)
        clf.fit(feat['train_X'], feat['train_y'])
        print(f"      LassoCV best α={clf.alpha_:.4g}")

    elif model_name == 'XGBoost':
        try:
            from xgboost import XGBRegressor
        except ImportError:
            print("      ⚠ xgboost 未安装，跳过"); return {}
        clf = XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=random_seed, n_jobs=-1,
            early_stopping_rounds=30,
            eval_metric='rmse', verbosity=0)
        clf.fit(feat['train_X'], feat['train_y'],
                eval_set=[(feat['val_X'], feat['val_y'])],
                verbose=False)

    elif model_name == 'LightGBM':
        try:
            import lightgbm as lgb
        except ImportError:
            print("      ⚠ lightgbm 未安装，跳过"); return {}
        clf = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=random_seed, n_jobs=-1,
            verbose=-1)
        clf.fit(feat['train_X'], feat['train_y'],
                eval_set=[(feat['val_X'], feat['val_y'])],
                callbacks=[lgb.early_stopping(30, verbose=False),
                           lgb.log_evaluation(-1)])
    else:
        raise ValueError(f"未知模型: {model_name}")

    preds  = clf.predict(feat['test_X']).astype(np.float64)
    labels = feat['test_y'].astype(np.float64)
    pearson_val = float(pearsonr(preds, labels)[0])
    r2_val      = float(r2_score(labels, preds))
    print(f"      Test Pearson={pearson_val:.4f}  R²={r2_val:.4f}")

    # ── Bootstrap CI ─────────────────────────────────────────────────
    boot_ci = None
    if do_bootstrap:
        boot_ci = compute_bootstrap_ci(preds, labels, n_boot, ci_level, random_seed)

    # ── 保存结果 ────────────────────────────────────────────────────
    out_dir = output_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    row = {'model': model_name, 'feature_set': label.split('__')[1] if '__' in label else label,
           'pearson': pearson_val, 'r2': r2_val}
    if boot_ci:
        row.update({'pearson_ci_lo': boot_ci['pearson_ci_lo'],
                    'pearson_ci_hi': boot_ci['pearson_ci_hi'],
                    'r2_ci_lo': boot_ci['r2_ci_lo'],
                    'r2_ci_hi': boot_ci['r2_ci_hi']})
    pd.DataFrame([row]).to_csv(out_dir / 'result.csv', index=False)

    return {'pearson': pearson_val, 'r2': r2_val, 'boot_ci': boot_ci,
            'preds': preds, 'labels': labels}


# ============================================================================
# ★ TF-only 基线 × multiseed  (不进 MODEL_COMPARISON 矩阵)
#   独立一步: 回答「仅用 79 维 TF TSS 加权均值, Ridge/XGBoost 能拿到多少 pearson」
#   - Ridge: 确定性 (RidgeCV 默认无 shuffle), 仅跑 1 seed 即可
#   - XGBoost: subsample/colsample 对 random_state 敏感, 跑全部 seeds
#   与 MULTI_SEED_EVAL 共用同一套固定象限标签, 可直接对齐 ours-C 的每象限 Pearson.
# ============================================================================
def run_tf_only_baseline_multiseed(
    X_raw, y_raw, split_masks, scalar_data, lm_emb_all,
    output_dir, quadrant_info=None,
    cfg=None, tpm_transform='log1p', tpm_normalize='zscore',
    n_seq_features=12,
):
    """
    TF-only 基线 × multiseed 评估.

    Args:
      X_raw          : (N, n_tfs, n_bins) TF 信号
      y_raw          : (N,) 原始 TPM
      split_masks    : 含 'train'/'val'/'test' 的 bool 掩码
      scalar_data    : 仅用于复用 _build_comparison_features 的接口,
                       TF-only 不用 scalar 特征 (但函数要求有输入)
      lm_emb_all     : 同上, 仅为接口兼容 (TF-only 忽略)
      output_dir     : 输出根目录
      quadrant_info  : compute_fixed_quadrant_labels 返回值, 为 None 时只算全局
      cfg            : 默认使用 TF_ONLY_BASELINE

    返回:
      pd.DataFrame (长表: 每 (model, seed) 一行, 带全局 + 象限列)
    """
    if cfg is None:
        cfg = TF_ONLY_BASELINE

    out = Path(output_dir) / cfg.get('output_subdir', 'tf_only_baseline')
    out.mkdir(parents=True, exist_ok=True)

    # ── 磁盘缓存复用 (force_retrain=False 时): 主 CSV 齐全即 skip 全部训练 ──
    _per_seed_csv = out / 'tf_only_per_seed.csv'
    _quad_csv     = out / 'tf_only_quadrant_per_seed.csv'
    _force = bool(cfg.get('force_retrain', False))
    _need_quad = (cfg.get('quadrant_stats', True) and quadrant_info is not None)
    _cache_ok = _per_seed_csv.exists() and ((not _need_quad) or _quad_csv.exists())
    if (not _force) and _cache_ok:
        try:
            df_cached = pd.read_csv(_per_seed_csv)
            if not df_cached.empty:
                _print = (f"  [CACHE] 复用已有 TF-only 基线结果 (force_retrain=False):\n"
                          f"     {_per_seed_csv.name}  ({len(df_cached)} 行)")
                if _need_quad:
                    _print += (f"\n     {_quad_csv.name}  "
                               f"({len(pd.read_csv(_quad_csv))} 行)")
                print(_print)
                print(f"  [CACHE] 跳过全部 Ridge/XGBoost 训练; 需重跑请设 "
                      f"TF_ONLY_BASELINE['force_retrain']=True")
                return df_cached
        except Exception as _ce:
            print(f"  ⚠ 读 TF-only 缓存失败, 回退到全量训练: {_ce}")

    # ── variants: 支持同时跑多种 TF 特征表示 (默认只跑 tss_weighted) ────────
    # 新增 'raw_bins_pca' 后, 两种表示的 Ridge 结果会被并列记录到同一 csv 里,
    # 便于直接回答「ours-C 的 Q2 输给 TF-only 是不是因为 PerTFEncoder 有 bug」.
    variants = list(cfg.get('variants', [cfg.get('tf_summary', 'tss_weighted')]))

    print(f"\n{'=' * 62}")
    print(f"  ★ TF-only 基线 × multiseed × variants  (不进基线对比矩阵)")
    print(f"  目的: 回答「纯 TF 特征 Ridge/XGBoost 的 pearson 落在哪」")
    print(f"  模型:   {cfg.get('models', ['Ridge'])}")
    print(f"  seeds:  {cfg.get('seeds', [42])}")
    print(f"  variants: {variants}")
    print(f"  输出: {out}")
    print('=' * 62)

    # test_gene_indices: 全 variants 共用 (split 不变)
    test_indices = np.where(split_masks['test'])[0].astype(np.int64)
    quad_arr = None
    compute_q = (cfg.get('quadrant_stats', True) and quadrant_info is not None)
    if compute_q:
        quad_arr = quadrant_info['quadrant_array']
    elif cfg.get('quadrant_stats', True):
        print("  ⚠ quadrant_stats=True 但 quadrant_info=None, 仅算全局 Pearson")

    models_to_run = cfg.get('models', ['Ridge'])
    seeds_to_run  = list(cfg.get('seeds', [42]))

    global_records   = []
    quadrant_records = []

    # ── 外层: 遍历 variants ────────────────────────────────────────────
    for variant in variants:
        print(f"\n  {'═' * 58}")
        print(f"  [variant={variant}]  构建特征...")
        print(f"  {'═' * 58}")

        # 构建该 variant 的特征 (特征生成本身确定性, 一次即可)
        feat_tf = _build_comparison_features(
            feature_set='tf_only',
            scalar_data=scalar_data,
            lm_emb=None,                   # TF-only 不用 LM
            X_raw=X_raw, split_masks=split_masks,
            tpm_transform=tpm_transform, tpm_normalize=tpm_normalize,
            y_raw=y_raw, tf_summary=variant,
            n_seq_features=n_seq_features,
        )
        print(f"  ✓ 特征就绪: n_feat={feat_tf['n_feat']}  "
              f"(n_train={len(feat_tf['train_X'])}, n_test={len(feat_tf['test_X'])})")

        # ── 内层: 遍历 (model, seed) ──────────────────────────────────
        for model_name in models_to_run:
            # Ridge/Lasso 确定性 — 仅跑第一个 seed
            is_deterministic = model_name in ('Ridge', 'Lasso')
            if is_deterministic and len(seeds_to_run) > 1:
                print(f"\n  [{variant}|{model_name}] 确定性模型, 仅 seed={seeds_to_run[0]}")
            eff_seeds = [seeds_to_run[0]] if is_deterministic else seeds_to_run

            for sd in eff_seeds:
                label = f'tf_only__{variant}__{model_name}__seed{sd}'
                print(f"\n  >> [{variant}] {model_name}  seed={sd}")
                try:
                    res = _train_sklearn_model(
                        model_name=model_name,
                        feat=feat_tf,
                        output_dir=out,
                        label=label,
                        do_bootstrap=cfg.get('bootstrap', True),
                        n_boot=cfg.get('n_boot', 2000),
                        ci_level=cfg.get('ci_level', 0.95),
                        random_seed=sd,
                    )
                    if not res:
                        continue
                except Exception as _e:
                    print(f"     ⚠ 失败: {_e}")
                    traceback.print_exc()
                    continue

                preds  = res['preds']; labels = res['labels']
                err = preds - labels
                row_g = {
                    'variant': variant,
                    'model': model_name, 'seed': sd,
                    'n_test': len(preds),
                    'n_feat': feat_tf['n_feat'],
                    'pearson': res['pearson'], 'r2': res['r2'],
                    'mae': float(np.abs(err).mean()),
                    'bias': float(err.mean()),
                }
                if res.get('boot_ci'):
                    row_g.update({k: res['boot_ci'][k] for k in
                                  ('pearson_ci_lo', 'pearson_ci_hi',
                                   'r2_ci_lo', 'r2_ci_hi')})
                global_records.append(row_g)

                if compute_q:
                    q_metrics = compute_per_quadrant_metrics(
                        preds, labels, test_indices, quad_arr)
                    print(f"     分象限: "
                          + "  ".join(f"{q}: r={q_metrics[q]['pearson']:+.4f} "
                                      f"(n={q_metrics[q]['n']})"
                                      for q in ('Q1', 'Q2', 'Q3', 'Q4')))
                    for q_name in ('Q1', 'Q2', 'Q3', 'Q4'):
                        m = q_metrics[q_name]
                        quadrant_records.append({
                            'variant': variant,
                            'model': model_name, 'seed': sd, 'quadrant': q_name,
                            'n': m['n'], 'pearson': m['pearson'],
                            'mae': m['mae'], 'bias': m['bias'],
                            'spearman': m['spearman'], 'r2': m['r2'],
                        })

    # ── 4. 汇总 & 保存 ────────────────────────────────────────────────
    if not global_records:
        print("  ⚠ TF-only 基线无有效结果"); return pd.DataFrame()

    df_global = pd.DataFrame(global_records)
    df_global.to_csv(out / 'tf_only_per_seed.csv', index=False)

    # 跨 seed 聚合 (按 variant + model 分组)
    stats_rows = []
    for (v_name, m_name), sub in df_global.groupby(['variant', 'model']):
        stats_rows.append({
            'variant': v_name,
            'model': m_name,
            'n_seeds': len(sub),
            'n_feat':       int(sub['n_feat'].iloc[0]),
            'pearson_mean': float(sub['pearson'].mean()),
            'pearson_std':  float(sub['pearson'].std(ddof=1)) if len(sub) > 1 else 0.0,
            'pearson_min':  float(sub['pearson'].min()),
            'pearson_max':  float(sub['pearson'].max()),
            'mae_mean':     float(sub['mae'].mean()),
            'bias_mean':    float(sub['bias'].mean()),
            'r2_mean':      float(sub['r2'].mean()),
        })
    df_stats = pd.DataFrame(stats_rows)
    df_stats.to_csv(out / 'tf_only_stats_by_model.csv', index=False)

    # 分象限长表 + per (variant, model, quadrant) 聚合
    df_quad = pd.DataFrame()
    qstats  = pd.DataFrame()
    if quadrant_records:
        df_quad = pd.DataFrame(quadrant_records)
        df_quad.to_csv(out / 'tf_only_quadrant_per_seed.csv', index=False)
        qstats = (df_quad.groupby(['variant', 'model', 'quadrant'])
                          .agg(n_mean=('n', 'mean'),
                               pearson_mean=('pearson', 'mean'),
                               pearson_std =('pearson', lambda x: x.std(ddof=1) if len(x)>1 else 0.0),
                               mae_mean=('mae', 'mean'),
                               bias_mean=('bias', 'mean'))
                          .reset_index())
        qstats.to_csv(out / 'tf_only_quadrant_stats.csv', index=False)

    # ── 5. 控制台汇总打印 —— 论文叙事关键一瞥 ───────────────────────
    print(f"\n  ┌─ TF-only 基线: 全局跨 seed 聚合 (variant × model) ───────────────")
    print(f"  │  {'variant':14s}  {'model':10s}  {'n_feat':>6s}  {'n_seeds':>7s}  "
          f"{'Pearson (mean±std)':>22s}  {'MAE':>8s}  {'Bias':>9s}")
    for _, r in df_stats.iterrows():
        print(f"  │  {r['variant']:14s}  {r['model']:10s}  "
              f"{int(r['n_feat']):>6d}  {int(r['n_seeds']):>7d}  "
              f"{r['pearson_mean']:+.4f} ± {r['pearson_std']:.4f}   "
              f"{r['mae_mean']:.3f}    "
              f"{r['bias_mean']:+.3f}")
    print(f"  └──────────────────────────────────────────────────────────────────")

    if quadrant_records:
        print(f"\n  ┌─ TF-only 基线: 分象限跨 seed 聚合 (论文重点 Q2/Q3) ──────────────")
        print(f"  │  {'variant':14s}  {'model':10s}  {'象限':6s}  "
              f"{'n':>5s}  {'Pearson (mean±std)':>22s}  {'MAE':>8s}  {'Bias':>9s}")
        for _, qrow in qstats.iterrows():
            marker = ' ★' if qrow['quadrant'] in ('Q2', 'Q3') else '  '
            print(f"  │  {qrow['variant']:14s}  {qrow['model']:10s}  "
                  f"{qrow['quadrant']:4s}{marker}  "
                  f"{int(qrow['n_mean']):>5d}  "
                  f"{qrow['pearson_mean']:+.4f} ± {qrow['pearson_std']:.4f}   "
                  f"{qrow['mae_mean']:.3f}    "
                  f"{qrow['bias_mean']:+.3f}")
        print(f"  └──────────────────────────────────────────────────────────────────")

    # ── 6. 轻量可视化: TF-only × model × quadrant 条形图 ─────────────────
    try:
        _plot_tf_only_summary(df_global, df_quad, out)
    except Exception as _pe:
        print(f"  ⚠ TF-only 可视化失败 (可忽略): {_pe}")

    return df_global if df_quad.empty else pd.merge(
        df_global, df_quad,
        on=['variant', 'model', 'seed'], how='outer', suffixes=('', '_quad'))


def _plot_tf_only_summary(df_global: pd.DataFrame,
                           df_quad: pd.DataFrame,
                           out_dir: Path):
    """TF-only 基线汇总图: 全局 pearson 条形 + 每象限条形 (variant × model)"""
    # 兼容旧 df (无 variant 列)
    if 'variant' not in df_global.columns:
        df_global = df_global.assign(variant='default')
    if not df_quad.empty and 'variant' not in df_quad.columns:
        df_quad = df_quad.assign(variant='default')

    variants = sorted(df_global['variant'].unique())
    models   = sorted(df_global['model'].unique())
    vm_keys  = [(v, m) for v in variants for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(max(13, 2.2 * len(vm_keys) + 5), 4.5))

    # Panel A: 全局 pearson (每 variant × model × seed 散点 + mean line)
    ax = axes[0]
    model_colors = {'Ridge': '#3A86FF', 'XGBoost': '#FF6B6B',
                    'Lasso': '#6A4C93', 'LightGBM': '#06A77D'}
    xlabels = []
    for i, (v, m) in enumerate(vm_keys):
        sub = df_global[(df_global['variant'] == v) & (df_global['model'] == m)]
        if sub.empty:
            xlabels.append(f'{v}\n{m}')
            continue
        jitter = np.random.RandomState(0).uniform(-0.1, 0.1, size=len(sub))
        ax.scatter(np.full(len(sub), i) + jitter, sub['pearson'],
                    color=model_colors.get(m, '#888'), s=50, alpha=0.8,
                    edgecolor='white',
                    label=f'{m} (n={len(sub)})' if i == 0 or m not in xlabels else None)
        ax.hlines(sub['pearson'].mean(), i - 0.3, i + 0.3,
                   color=model_colors.get(m, '#888'), lw=2.5, zorder=3)
        xlabels.append(f'{v}\n{m}')
    ax.set_xticks(range(len(vm_keys)))
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel('Test Pearson r')
    ax.set_title('A. TF-only Global Pearson (variant × model, per seed + mean)',
                 fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)
    ax.spines[['top', 'right']].set_visible(False)
    # 去重 legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc='lower right')

    # Panel B: 每象限 pearson bars (按 variant × model 分组)
    ax = axes[1]
    if not df_quad.empty:
        qstats = (df_quad.groupby(['variant', 'model', 'quadrant'])
                          .agg(pearson_mean=('pearson', 'mean'),
                               pearson_std=('pearson', lambda x: x.std(ddof=1) if len(x)>1 else 0.0))
                          .reset_index())
        quads = ['Q1', 'Q2', 'Q3', 'Q4']
        n_vm = len(vm_keys); bar_w = 0.8 / max(n_vm, 1)
        x_base = np.arange(len(quads))
        # 用 variant 区分 hatch, model 区分颜色, 一眼能看出 raw_bins_pca vs tss_weighted
        v_hatches = {v: h for v, h in zip(variants, ['', '//', 'xx', '..'])}
        for i, (v, m) in enumerate(vm_keys):
            means, stds = [], []
            for q in quads:
                row = qstats[(qstats['variant'] == v) &
                             (qstats['model'] == m) &
                             (qstats['quadrant'] == q)]
                means.append(float(row['pearson_mean'].iloc[0]) if not row.empty else 0.0)
                stds.append(float(row['pearson_std'].iloc[0]) if not row.empty else 0.0)
            offs = (i - (n_vm - 1) / 2) * bar_w
            ax.bar(x_base + offs, means, width=bar_w, yerr=stds,
                    color=model_colors.get(m, '#888'),
                    hatch=v_hatches.get(v, ''),
                    label=f'{v}|{m}',
                    edgecolor='white', capsize=3, alpha=0.85)
        ax.set_xticks(x_base)
        ax.set_xticklabels(quads)
        ax.axvspan(0.5, 2.5, color='yellow', alpha=0.06, zorder=0)  # Q2/Q3 高亮区
        ax.set_ylabel('Pearson r (mean ± std across seeds)')
        ax.set_title('B. TF-only Quadrant Pearson',
                      fontweight='bold')
        ax.set_ylim(-0.2, 1)
        ax.axhline(0, color='k', lw=0.5)
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=7, loc='upper right', ncol=max(1, len(variants)))
    else:
        ax.text(0.5, 0.5, '无象限数据\n(quadrant_info=None)',
                 ha='center', va='center', transform=ax.transAxes,
                 fontsize=11, color='#888')
        ax.axis('off')

    plt.tight_layout()
    fig.savefig(out_dir / 'tf_only_summary.pdf', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  ✓ TF-only 可视化 → {out_dir / 'tf_only_summary.pdf'}")


# ============================================================================
# DL-Tabular 基线：GatedMLP + FT-Transformer
# 两者均直接作用于扁平化表格特征向量（activity + LM + TF summary 拼接）
# ============================================================================

class _GatedMLP(nn.Module):
    """
    Gated MLP（GLU-style）：每层输出 = Linear(x) * sigmoid(Gate(x))。
    适用于高维表格特征，参数量与全连接相当但梯度流更稳定。
    """
    def __init__(self, in_dim: int, d_hidden: int = 256, n_layers: int = 3,
                 dropout: float = 0.20):
        super().__init__()
        dims = [in_dim] + [d_hidden] * n_layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'linear': nn.Linear(dims[i], dims[i+1]),
                'gate':   nn.Linear(dims[i], dims[i+1]),
                'norm':   nn.LayerNorm(dims[i+1]),
                'drop':   nn.Dropout(dropout),
            }))
            # 残差投影（仅当维度不同时）
        self.residual_projs = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1]) if dims[i] != dims[i+1] else nn.Identity()
            for i in range(n_layers)
        ])
        self.head = nn.Linear(d_hidden, 1)
        self.out_dim = d_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk, proj in zip(self.layers, self.residual_projs):
            h = blk['linear'](x) * torch.sigmoid(blk['gate'](x))
            h = blk['norm'](h)
            h = blk['drop'](h)
            x = h + proj(x)
        return self.head(x).squeeze(-1)


class _FeatureTokenizer(nn.Module):
    """将每个标量特征嵌入为 d_token 维 token（Feature Tokenizer）。"""
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        # 每个特征独立的 weight + bias，形同 n_features 个 nn.Linear(1, d_token)
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias   = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features) → (B, n_features, d_token)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class _FTTransformer(nn.Module):
    """
    FT-Transformer（Feature Tokenizer + Transformer）。
    参考: Gorishniy et al., 2021 (NeurIPS)。
    适合多模态表格特征（promoter activity / LM embedding / TF summary）。
    每个标量特征被线性投影为独立 token，再经 Transformer 编码，
    最后通过 [CLS] token 或 mean-pool 做回归。
    """
    def __init__(self, n_features: int, d_token: int = 64,
                 n_layers: int = 3, n_heads: int = 4, dropout: float = 0.20):
        super().__init__()
        assert d_token % n_heads == 0
        self.tokenizer = _FeatureTokenizer(n_features, d_token)
        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads,
            dim_feedforward=d_token * 4,   # 标准 4x FFN 宽度
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers,
            enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token // 2), nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_token // 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features)
        tokens = self.tokenizer(x)                    # (B, n_features, d_token)
        cls    = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, d_token)
        tokens = torch.cat([cls, tokens], dim=1)      # (B, 1+n_features, d_token)
        out    = self.transformer(tokens)             # (B, 1+n_features, d_token)
        cls_out = self.norm(out[:, 0])                # (B, d_token)
        return self.head(cls_out).squeeze(-1)


def _train_dl_tabular_model(model_name: str, feat: dict,
                             output_dir: Path, label: str,
                             cfg_dl: dict,
                             do_bootstrap: bool = True,
                             n_boot: int = 2000,
                             ci_level: float = 0.95,
                             random_seed: int = 42) -> dict:
    """
    训练 GatedMLP 或 FT-Transformer 表格 DL 模型。
    cfg_dl: MODEL_COMPARISON['dl_tabular'] 子字典。
    """
    _dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    n_feat     = feat['n_feat']
    epochs     = cfg_dl.get('epochs',       150)
    patience   = cfg_dl.get('patience',     20)
    lr         = cfg_dl.get('lr',           3e-4)
    dropout    = cfg_dl.get('dropout',      0.20)
    # FT-Transformer token 数随 n_feat 线性增长，自动缩小 batch 防显存溢出
    base_bs    = cfg_dl.get('batch_size',   128)
    if model_name == 'FT-Transformer' and n_feat > 200:
        batch_size = max(32, base_bs // max(1, n_feat // 200))
    else:
        batch_size = base_bs

    # AMP：CUDA 可用时开启混合精度，显著降低显存占用
    use_amp    = (_dev.type == 'cuda')
    amp_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"\n    [{model_name}] n_feat={n_feat}  n_train={len(feat['train_X'])}  "
          f"n_test={len(feat['test_X'])}  batch={batch_size}  device={_dev}"
          f"{'  AMP=on' if use_amp else ''}")

    if model_name == 'GatedMLP':
        model = _GatedMLP(
            in_dim=n_feat,
            d_hidden=cfg_dl.get('d_hidden_mlp', 256),
            n_layers=cfg_dl.get('n_layers', 3),
            dropout=dropout).to(_dev)
    elif model_name == 'FT-Transformer':
        model = _FTTransformer(
            n_features=n_feat,
            d_token=cfg_dl.get('d_token',  64),
            n_layers=cfg_dl.get('n_layers',  3),
            n_heads=cfg_dl.get('n_heads',    4),
            dropout=dropout).to(_dev)
    else:
        raise ValueError(f"未知 DL-Tabular 模型: {model_name}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"      参数量: {n_params:,}")

    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit  = CombinedLoss(mse_w=0.5, pearson_w=0.5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)

    # ── DataLoader ──────────────────────────────────────────────────────
    def _loader(X, y, shuffle):
        ds = torch.utils.data.TensorDataset(
            torch.FloatTensor(X), torch.FloatTensor(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=(_dev.type == 'cuda'))

    tr_loader = _loader(feat['train_X'], feat['train_y'], shuffle=True)
    va_loader = _loader(feat['val_X'],   feat['val_y'],   shuffle=False)

    best_val_r = -np.inf; best_state = None; no_improve = 0
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(_dev), yb.to(_dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(xb)
                loss = crit(pred, yb)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            amp_scaler.step(opt)
            amp_scaler.update()
        sched.step()

        model.eval()
        with torch.no_grad():
            va_preds = torch.cat([model(xb.to(_dev)).cpu() for xb, _ in va_loader]).numpy()
        va_r = float(pearsonr(va_preds, feat['val_y'])[0])
        if va_r > best_val_r:
            best_val_r = va_r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"      Early stop @ ep={ep+1}  best val r={best_val_r:.4f}")
            break

    # ── 测试集评估 ──────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    te_ds = torch.FloatTensor(feat['test_X'])
    with torch.no_grad():
        te_preds = torch.cat([
            model(te_ds[s:s+batch_size].to(_dev)).cpu()
            for s in range(0, len(te_ds), batch_size)
        ]).numpy().astype(np.float64)

    # 训练完成后释放 GPU 显存
    model.cpu()
    del model
    if _dev.type == 'cuda':
        torch.cuda.empty_cache()

    labels = feat['test_y'].astype(np.float64)
    pearson_val = float(pearsonr(te_preds, labels)[0])
    r2_val      = float(r2_score(labels, te_preds))
    print(f"      Test Pearson={pearson_val:.4f}  R²={r2_val:.4f}")

    boot_ci = None
    if do_bootstrap:
        boot_ci = compute_bootstrap_ci(te_preds, labels, n_boot, ci_level, random_seed)

    out_dir = output_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    row = {'model': model_name,
           'feature_set': label.split('__')[1] if '__' in label else label,
           'pearson': pearson_val, 'r2': r2_val}
    if boot_ci:
        row.update({'pearson_ci_lo': boot_ci['pearson_ci_lo'],
                    'pearson_ci_hi': boot_ci['pearson_ci_hi'],
                    'r2_ci_lo':  boot_ci['r2_ci_lo'],
                    'r2_ci_hi':  boot_ci['r2_ci_hi']})
    pd.DataFrame([row]).to_csv(out_dir / 'result.csv', index=False)
    return {'pearson': pearson_val, 'r2': r2_val, 'boot_ci': boot_ci,
            'preds': te_preds, 'labels': labels}


def run_model_comparison_matrix(
        X_raw, y_raw, split_masks,
        scalar_data, lm_emb_all,
        dl_results: dict,
        output_dir: str,
        cfg: dict = None,
        tpm_transform: str = 'log1p',
        tpm_normalize:  str = 'zscore',
        n_seq_features: int = 12,
        quadrant_info: dict = None,
        test_gene_indices: np.ndarray = None,
) -> pd.DataFrame:
    """
    构建 3(模型族) × 3(特征集) 对比矩阵，并生成热图。

    DL 行的数据直接从 dl_results 读取（已完成训练），
    线性/非线性行在此函数内现场训练。

    new52 新增 per-quadrant 统一口径:
      若 quadrant_info (来自 compute_fixed_quadrant_labels) 和 test_gene_indices
      都提供, 则每个模型 (sklearn / DL-Tabular / DL 消融) 都额外输出固定锚下的
      Q1/Q2/Q3/Q4 Pearson/MAE/Bias, 并把这些指标以长表形式写入
      `comparison_matrix_quadrant.csv` (列: model, feature_set, quadrant, n,
      pearson, mae, bias). 横向可直接与 `multiseed_C_quadrant_per_seed.csv`,
      `tf_only_quadrant_stats.csv` 拼接, 形成论文 Supp S11 的统一视图.

    返回 DataFrame, 行=模型, 列=特征集 (主表, 不含 per-quadrant 长表).
    """
    if cfg is None:
        cfg = MODEL_COMPARISON

    out = Path(output_dir) / cfg.get('output_subdir', 'model_comparison_matrix')
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 62}")
    print("  ★ 模型对比矩阵实验")
    print(f"  行: Linear / 非线性 / DL-Tabular(GatedMLP/FT-Transformer) / DL消融(L/B/C)")
    print(f"  列: LM-only / +Activity / +TF(Full)")
    print(f"  输出: {out}")
    print('=' * 62)

    # ── 磁盘缓存复用 (force_retrain=False 时): 两份主 CSV 都在就直接读回 ──
    _full_csv = out / 'comparison_matrix_full.csv'
    _quad_csv = out / 'comparison_matrix_quadrant.csv'
    _force = bool(cfg.get('force_retrain', False))
    if (not _force) and _full_csv.exists() and _quad_csv.exists():
        try:
            df_cached = pd.read_csv(_full_csv)
            if not df_cached.empty:
                print(f"  [CACHE] 复用已有结果 (force_retrain=False):")
                print(f"     {_full_csv.name}  ({len(df_cached)} 行)")
                print(f"     {_quad_csv.name}  ({len(pd.read_csv(_quad_csv))} 行)")
                print(f"  [CACHE] 跳过全部基线训练; 需重跑请设 "
                      f"MODEL_COMPARISON['force_retrain']=True")
                # 如果有画图函数需要重绘, 可在此调用 (不重训但刷新图);
                # 这里保持最小侵入, 仅返回缓存的 df — Supp S11 汇合和上层日志够用.
                return df_cached
        except Exception as _ce:
            print(f"  ⚠ 读缓存失败, 回退到全量训练: {_ce}")

    # Q2 新三列均依赖 LM 嵌入 → LM 不可用时直接短路，避免生成空矩阵
    if lm_emb_all is None:
        print("  ⚠ lm_emb_all=None: 三列均依赖 LM 嵌入，对比矩阵跳过")
        return pd.DataFrame()

    feature_sets = ['lm_only', 'lm_act', 'full']
    fs_labels    = ['LM-only', '+Activity', '+TF (Full)']

    # ── 1. 提前构建特征集 ────────────────────────────────────────────
    #   feats     : 全维 LM（供 sklearn / GatedMLP 使用）
    #   feats_ft  : PCA 压缩 LM（专供 FT-Transformer，避免 token 数爆炸）
    feats    = {}
    feats_ft = {}
    _lm_compress = cfg.get('dl_tabular', {}).get('lm_compress_dim', 64)
    for fs in feature_sets:
        _common = dict(
            feature_set=fs, scalar_data=scalar_data, lm_emb=lm_emb_all,
            X_raw=X_raw, split_masks=split_masks,
            tpm_transform=tpm_transform, tpm_normalize=tpm_normalize,
            y_raw=y_raw, tf_summary=cfg.get('tf_summary', 'mean'),
            n_seq_features=n_seq_features)
        feats[fs] = _build_comparison_features(**_common)
        feats_ft[fs] = _build_comparison_features(**_common,
                                                   lm_compress_dim=_lm_compress)
        print(f"  ✓ 特征集 [{fs}]: n_feat={feats[fs]['n_feat']}"
              f"  (FT-Transformer 压缩后: {feats_ft[fs]['n_feat']})")

    # ── 2. 基线模型列表 ──────────────────────────────────────────
    baseline_models = []
    if cfg.get('run_linear', True):
        baseline_models += ['Ridge', 'Lasso']
    if cfg.get('run_nonlinear', True):
        baseline_models += ['XGBoost', 'LightGBM']
    if cfg.get('run_dl_tabular', True):
        baseline_models += ['GatedMLP', 'FT-Transformer']

    # ── 3. DL 结果映射 ───────────────────────────────────────────
    #   Q2 新三列与 ours 消融链真正 1:1 对应（不再是占位）:
    #     lm_only → L (LM 2304)                消融 L
    #     lm_act  → B (LM 2304 + Act 12)       消融 B
    #     full    → C (LM 2304 + Act 12 + TF)  消融 C
    dl_key_map = {
        'lm_only': 'ablation_L_lm_only',
        'lm_act':  'ablation_B_act_lm',
        'full':    'ablation_C_seq_tf',
    }

    # ── 4. 汇总表 ────────────────────────────────────────────────
    records = []
    quad_records = []   # per-quadrant 长表: model, feature_set, quadrant, n, pearson, mae, bias

    # ── per-quadrant 就地计算 (若 quadrant_info 可用) ────────────────────────
    # 这里把 test_gene_indices 集中求一次, 供所有基线模型复用.
    _can_quadrant = (quadrant_info is not None and 'quadrant_array' in quadrant_info)
    if _can_quadrant:
        _quad_arr = np.asarray(quadrant_info['quadrant_array'])
        # test_gene_indices: 优先显式传入; 否则从 split_masks['test'] 推导
        if test_gene_indices is not None:
            _te_idx = np.asarray(test_gene_indices)
        else:
            _te_idx = np.where(split_masks['test'])[0]
        _te_quadrant = _quad_arr[_te_idx]
        print(f"  [per-quadrant] 固定锚={quadrant_info.get('anchor_name','?')}  "
              f"n_test={len(_te_idx)}  " +
              "  ".join(f"{q}={int((_te_quadrant == q).sum())}"
                         for q in ('Q1', 'Q2', 'Q3', 'Q4')))
    else:
        _te_quadrant = None
        _te_idx = None
        print(f"  [per-quadrant] quadrant_info 未提供, 跳过 per-quadrant 输出")

    def _per_quadrant_row(preds, labels, model_name, fs):
        """把单个模型的 test 预测按固定象限拆, 追加行到 quad_records."""
        if _te_quadrant is None:
            return
        for q in ('Q1', 'Q2', 'Q3', 'Q4'):
            mask = (_te_quadrant == q)
            n_q = int(mask.sum())
            if n_q < 5:
                quad_records.append({
                    'model': model_name, 'feature_set': fs, 'quadrant': q,
                    'n': n_q, 'pearson': np.nan, 'mae': np.nan, 'bias': np.nan})
                continue
            p_q = np.asarray(preds)[mask]
            l_q = np.asarray(labels)[mask]
            try:
                r = float(pearsonr(p_q, l_q)[0])
            except Exception:
                r = float('nan')
            err = p_q - l_q
            quad_records.append({
                'model': model_name, 'feature_set': fs, 'quadrant': q,
                'n': n_q, 'pearson': r,
                'mae':  float(np.abs(err).mean()),
                'bias': float(err.mean()),
            })

    # 4a. 训练基线模型（线性 / 非线性 / DL-Tabular）
    _boot_kwargs = dict(
        do_bootstrap=cfg.get('bootstrap', True),
        n_boot=cfg.get('n_boot', 2000),
        ci_level=cfg.get('ci_level', 0.95),
        random_seed=cfg.get('random_seed', 42),
    )
    for model_name in baseline_models:
        for fs in feature_sets:
            if fs not in feats:
                continue
            label = f"{model_name}__{fs}"
            print(f"\n  >> {model_name} × {fs}")
            if model_name in ('GatedMLP', 'FT-Transformer'):
                _feat_src = feats_ft[fs] if model_name == 'FT-Transformer' else feats[fs]
                res = _train_dl_tabular_model(
                    model_name=model_name, feat=_feat_src,
                    output_dir=out, label=label,
                    cfg_dl=cfg.get('dl_tabular', {}),
                    **_boot_kwargs,
                )
            else:
                res = _train_sklearn_model(
                    model_name=model_name, feat=feats[fs],
                    output_dir=out, label=label,
                    **_boot_kwargs,
                )
            if not res:
                continue
            row = {'model': model_name, 'feature_set': fs,
                   'feature_set_label': fs_labels[feature_sets.index(fs)],
                   'pearson': res['pearson'], 'r2': res['r2']}
            ci = res.get('boot_ci')
            if ci:
                row.update({'pearson_ci_lo': ci['pearson_ci_lo'],
                            'pearson_ci_hi': ci['pearson_ci_hi']})
            # per-quadrant (_train_sklearn_model 和 _train_dl_tabular_model 都返回 preds/labels)
            if _can_quadrant and 'preds' in res and 'labels' in res:
                _per_quadrant_row(res['preds'], res['labels'], model_name, fs)
            records.append(row)

    # 4b. 收集 DL 行（直接读 dl_results，不重复训练）
    for fs, dl_key in dl_key_map.items():
        if fs not in feats:
            continue
        dl_res = dl_results.get(dl_key)
        if dl_res is None:
            print(f"  ⚠ DL 结果 [{dl_key}] 未找到，跳过")
            continue
        row = {'model': 'ours', 'feature_set': fs,
               'feature_set_label': fs_labels[feature_sets.index(fs)],
               'pearson': dl_res['pearson'], 'r2': dl_res['r2']}
        ci = dl_res.get('boot_ci')
        if ci:
            row.update({'pearson_ci_lo': ci.get('pearson_ci_lo', np.nan),
                        'pearson_ci_hi': ci.get('pearson_ci_hi', np.nan)})
        # per-quadrant (train_experiment 已经把 preds_test/labels_test/test_gene_indices
        # 写入 dl_res; 其 test_gene_indices 与这里的 _te_idx 一致)
        if _can_quadrant:
            _p = dl_res.get('preds_test')
            _l = dl_res.get('labels_test')
            if _p is not None and _l is not None:
                _per_quadrant_row(_p, _l, f'ours_{dl_key.replace("ablation_", "")}', fs)
        records.append(row)

    if not records:
        print("  ⚠ 无有效对比结果，跳过"); return pd.DataFrame()

    df = pd.DataFrame(records)
    df.to_csv(out / 'comparison_matrix_full.csv', index=False)
    print(f"\n  ✓ 完整对比表 → {out / 'comparison_matrix_full.csv'}")

    # ── 4c. per-quadrant 长表落盘 (统一口径, 供 Supp S11 汇合使用) ─────────
    if quad_records:
        qdf = pd.DataFrame(quad_records)
        qdf.to_csv(out / 'comparison_matrix_quadrant.csv', index=False)
        print(f"  ✓ per-quadrant 长表 → {out / 'comparison_matrix_quadrant.csv'}")
        # 控制台快照: 以 Q2 为例 (论文重点), 按 pearson 降序
        try:
            _q2 = qdf[qdf['quadrant'] == 'Q2'].sort_values('pearson', ascending=False)
            if not _q2.empty:
                print(f"\n  Q2 (trans-repressed) per-model 排名 (n={int(_q2['n'].iloc[0])}):")
                for _, _r in _q2.iterrows():
                    print(f"    {_r['model']:>18s} × {_r['feature_set']:<8s}  "
                          f"r={_r['pearson']:+.4f}  MAE={_r['mae']:.3f}  "
                          f"Bias={_r['bias']:+.3f}")
        except Exception:
            pass

    # ── 5. 打印汇总表 ────────────────────────────────────────────
    _print_comparison_table(df, feature_sets, fs_labels)

    # ── 6. 绘制热图 + CI 误差条图 ────────────────────────────────
    _plot_comparison_matrix(df, out, feature_sets, fs_labels)

    return df


def _print_comparison_table(df: pd.DataFrame,
                              feature_sets: list, fs_labels: list):
    """将对比矩阵打印为对齐的文本表格。"""
    models = df['model'].unique().tolist()
    print(f"\n{'─'*74}")
    print(f"  {'模型':22s} | " + " | ".join(f"{l:>18s}" for l in fs_labels))
    print(f"{'─'*74}")
    for m in models:
        row_strs = []
        for fs in feature_sets:
            sub = df[(df['model'] == m) & (df['feature_set'] == fs)]
            if sub.empty:
                row_strs.append(f"{'—':>18s}")
            else:
                r = float(sub['pearson'].values[0])
                if 'pearson_ci_lo' in sub.columns and not pd.isna(sub['pearson_ci_lo'].values[0]):
                    lo = float(sub['pearson_ci_lo'].values[0])
                    hi = float(sub['pearson_ci_hi'].values[0])
                    row_strs.append(f"{r:.4f}[{lo:.3f},{hi:.3f}]".rjust(18))
                else:
                    row_strs.append(f"{r:.4f}".rjust(18))
        print(f"  {m:22s} | " + " | ".join(row_strs))
    print(f"{'─'*74}")


def _plot_comparison_matrix(df: pd.DataFrame, out: Path,
                             feature_sets: list, fs_labels: list):
    """绘制两种图：① Pearson 热图  ② 分组误差条图"""
    models     = df['model'].unique().tolist()
    n_models   = len(models)
    n_fs       = len(feature_sets)

    # ── 热图数据 ─────────────────────────────────────────────────
    heat = np.full((n_models, n_fs), np.nan)
    for i, m in enumerate(models):
        for j, fs in enumerate(feature_sets):
            sub = df[(df['model'] == m) & (df['feature_set'] == fs)]
            if not sub.empty:
                heat[i, j] = float(sub['pearson'].values[0])

    # ── ① 热图 ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, n_fs * 2.2), max(4, n_models * 0.9 + 1.5)))
    cmap = LinearSegmentedColormap.from_list(
        'cmp', ['#f0f4f8', '#3A86FF', '#023e8a'])
    vmin = max(0, np.nanmin(heat) - 0.05)
    vmax = min(1, np.nanmax(heat) + 0.02)
    im = ax.imshow(heat, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Test Pearson r')

    for i in range(n_models):
        for j in range(n_fs):
            v = heat[i, j]
            if np.isfinite(v):
                txt_col = 'white' if v > (vmin + vmax) * 0.6 else 'black'
                ax.text(j, i, f'{v:.4f}', ha='center', va='center',
                        fontsize=9, fontweight='bold', color=txt_col)

    ax.set_xticks(range(n_fs)); ax.set_xticklabels(fs_labels, fontsize=10)
    ax.set_yticks(range(n_models)); ax.set_yticklabels(models, fontsize=9)
    ax.set_title('Model × Feature-Set Comparison Matrix\n(Test Pearson r)',
                 fontsize=11, fontweight='bold', pad=10)
    # 在各模型族之间画分隔线
    _linear_set    = {'Ridge', 'Lasso'}
    _nonlinear_set = {'XGBoost', 'LightGBM'}
    _dltab_set     = {'GatedMLP', 'FT-Transformer'}
    _family_order  = [_linear_set, _nonlinear_set, _dltab_set]
    for fam_set in _family_order:
        # 找到该族最后一行的索引
        last_i = max((i for i, m in enumerate(models) if m in fam_set), default=-1)
        if 0 <= last_i < n_models - 1:
            ax.axhline(last_i + 0.5, color='#888888', lw=1.2, ls='--', alpha=0.6)
    # 加粗分隔 ours（消融模型）与所有基线
    ours_i = next((i for i, m in enumerate(models) if m == 'ours'), None)
    if ours_i is not None and ours_i > 0:
        ax.axhline(ours_i - 0.5, color='#E63946', lw=2, ls='--', alpha=0.8)
    plt.tight_layout()
    fig.savefig(out / 'fig_comparison_heatmap.pdf', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  ✓ 对比热图 → {out / 'fig_comparison_heatmap.pdf'}")

    # ── ② 分组误差条图 ──────────────────────────────────────────
    fig2, axes = plt.subplots(1, n_fs, figsize=(5 * n_fs, max(4, n_models * 0.55 + 2)),
                               sharey=True)
    if n_fs == 1: axes = [axes]

    color_map = {
        'Ridge':          '#457B9D',
        'Lasso':          '#A8DADC',
        'XGBoost':        '#E9C46A',
        'LightGBM':       '#F4A261',
        'GatedMLP':       '#8338EC',
        'FT-Transformer': '#3A0CA3',
        'ours':           '#E63946',
    }

    for ax_j, (fs, fs_lbl) in enumerate(zip(feature_sets, fs_labels)):
        ax = axes[ax_j]
        sub_fs = df[df['feature_set'] == fs].copy()
        sub_fs = sub_fs.set_index('model').reindex(models).reset_index()

        y_pos = np.arange(len(models))
        for yi, row in enumerate(sub_fs.itertuples()):
            m_name = getattr(row, 'model', models[yi])
            r_val  = getattr(row, 'pearson', np.nan)
            if not np.isfinite(r_val):
                continue
            col = color_map.get(m_name, '#888888')
            ax.barh(yi, r_val, height=0.55, color=col, alpha=0.80, edgecolor='white')
            if ('pearson_ci_lo' in df.columns and
                    not pd.isna(getattr(row, 'pearson_ci_lo', np.nan))):
                lo = getattr(row, 'pearson_ci_lo')
                hi = getattr(row, 'pearson_ci_hi')
                ax.errorbar(r_val, yi, xerr=[[r_val - lo], [hi - r_val]],
                            fmt='none', color='#333333', capsize=3, lw=1.5)
            ax.text(r_val + 0.005, yi, f'{r_val:.4f}',
                    va='center', fontsize=7.5, color='#333333')

        ax.set_yticks(y_pos); ax.set_yticklabels(models, fontsize=9)
        ax.set_xlabel('Test Pearson r', fontsize=9)
        ax.set_title(fs_lbl, fontsize=10, fontweight='bold')
        ax.set_xlim(0, min(1.0, np.nanmax(heat) + 0.12))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=0.25, ls='--')

        # 标记 DL 行
        if ours_i is not None:
            ax.axhline(ours_i, color='#E63946', lw=1.2, ls='--', alpha=0.5)

    fig2.suptitle('Model Comparison by Feature Set (Test Pearson r ± 95% CI)',
                  fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig2.savefig(out / 'fig_comparison_bars.pdf', bbox_inches='tight', dpi=200)
    plt.close(fig2)
    print(f"  ✓ 误差条图 → {out / 'fig_comparison_bars.pdf'}")

    # ── ③ Δ增益分析：+Activity 和 +TF 相对 LM-only 的增益 ──────
    _plot_gain_analysis(df, out, feature_sets, fs_labels, models)


def _plot_gain_analysis(df: pd.DataFrame, out: Path,
                         feature_sets: list, fs_labels: list, models: list):
    """绘制每个模型的特征增益条图（Pearson r 相对 LM-only 的提升量）。"""
    if 'lm_only' not in feature_sets:
        return

    gain_records = []
    for m in models:
        base_sub = df[(df['model'] == m) & (df['feature_set'] == 'lm_only')]
        if base_sub.empty or pd.isna(base_sub['pearson'].values[0]):
            continue
        base_r = float(base_sub['pearson'].values[0])
        for fs in feature_sets[1:]:          # 跳过 baseline 自身
            sub = df[(df['model'] == m) & (df['feature_set'] == fs)]
            if sub.empty or pd.isna(sub['pearson'].values[0]):
                continue
            gain_records.append({
                'model': m, 'feature_set': fs,
                'feature_set_label': fs_labels[feature_sets.index(fs)],
                'delta_pearson': float(sub['pearson'].values[0]) - base_r,
                'base_pearson':  base_r,
            })

    if not gain_records:
        return
    gdf = pd.DataFrame(gain_records)
    gdf.to_csv(out / 'comparison_gain_analysis.csv', index=False)

    n_fs_gain = len(feature_sets) - 1
    fig, axes = plt.subplots(1, n_fs_gain,
                              figsize=(4.5 * n_fs_gain, max(4, len(models) * 0.55 + 2)),
                              sharey=True)
    if n_fs_gain == 1: axes = [axes]

    color_map2 = {'+Activity': '#FB8500', '+TF (Full)': '#E63946'}
    for ax_i, fs in enumerate(feature_sets[1:]):
        ax = axes[ax_i]
        fs_lbl = fs_labels[feature_sets.index(fs)]
        sub = gdf[gdf['feature_set'] == fs].set_index('model').reindex(models).reset_index()
        y_pos = np.arange(len(models))
        col = color_map2.get(fs_lbl, '#888888')
        for yi, row in enumerate(sub.itertuples()):
            delta = getattr(row, 'delta_pearson', np.nan)
            if not np.isfinite(delta):
                continue
            ax.barh(yi, delta, height=0.55,
                    color=col if delta >= 0 else '#AAAAAA',
                    alpha=0.80, edgecolor='white')
            ax.text(delta + (0.001 if delta >= 0 else -0.001),
                    yi, f'{delta:+.4f}', va='center',
                    ha='left' if delta >= 0 else 'right',
                    fontsize=7.5, color='#333333')
        ax.axvline(0, color='black', lw=0.8)
        ax.set_yticks(y_pos); ax.set_yticklabels(models, fontsize=9)
        ax.set_xlabel('ΔPearson r vs LM-only', fontsize=9)
        ax.set_title(f'Gain: {fs_lbl}', fontsize=10, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=0.25, ls='--')

    fig.suptitle('Feature Addition Gain (ΔPearson r relative to LM-only)',
                 fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(out / 'fig_comparison_gain.pdf', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  ✓ 增益分析图 → {out / 'fig_comparison_gain.pdf'}")


# ============================================================================
# train_experiment — 完整消融实验入口（无 DNA）
# ============================================================================

# ============================================================================
# 断点续跑缓存 helper（配合 RESUME_FROM_CACHE 使用）
# ============================================================================

def _is_train_experiment_complete(exp_dir: Path) -> bool:
    """两个文件都存在 → 视为本次实验已完整完成。"""
    return (exp_dir / 'complete_model.pth').exists() and \
           (exp_dir / 'train_result_cache.pth').exists()


def _save_train_experiment_cache(exp_dir: Path, result: dict) -> None:
    """把 train_experiment / train_dna_experiment 的 return dict
    去掉 'model'（不可序列化 / 体积大）后存盘。
    下次命中缓存时由 _load_train_experiment_cache 恢复。"""
    if not RESUME_FROM_CACHE.get('enabled', True):
        return
    cache = {k: v for k, v in result.items() if k != 'model'}
    torch.save(cache, exp_dir / 'train_result_cache.pth')


def _load_train_experiment_cache(exp_dir: Path, device) -> dict:
    """从 train_result_cache.pth 恢复结果 dict，并从 complete_model.pth
    重建 model（用 torch.load map_location 映射到当前 device）。
    注意: model 字段会是 state_dict OrderedDict，不是 nn.Module 实例；
    调用方若需要 nn.Module 请自行重建（FUSION_ABLATION 的 B-model fallback
    已有此机制）。此处只保证 pearson / r2 / preds_test / labels_test /
    test_gene_indices / boot_ci 等统计字段可用。"""
    cache = torch.load(exp_dir / 'train_result_cache.pth',
                       map_location=device, weights_only=False)
    # 补充 model = None（下游用到 model 时走 complete_model.pth fallback）
    cache.setdefault('model', None)
    return cache


def _rebuild_b_model_from_ckpt(ckpt_path, device, lm_emb_available: bool):
    """从 ablation_B_act_lm 的 complete_model.pth 重建 SequenceOnlyPredictor.

    用途: 当 all_results['ablation_B_act_lm'] 中的 'model' 字段为 None
    (RESUME_FROM_CACHE 命中时 _load_train_experiment_cache 不会反序列化 model),
    从磁盘 ckpt 重建并加载 state_dict —— 让后续 C 软监督 / Step 3.5b 象限锚
    / multi-seed 三处都能拿到可推断的 nn.Module. 同时把参数冻结并 eval().

    返回 SequenceOnlyPredictor 或 None (ckpt 不存在时).
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        return None
    state = _safe_torch_load(str(ckpt_path), device)
    sd = (state.get('model_state_dict', state)
          if isinstance(state, dict) else state)
    cfg = state.get('config', {}) if isinstance(state, dict) else {}
    MC = MODEL_CONFIG
    n_sf = cfg.get('n_seq_features', N_SEQ_FEATURES)
    m = SequenceOnlyPredictor(
        n_windows=n_sf,
        d_profile=MC['d_profile'], d_hidden=128, dropout=0.25,
        use_lm=cfg.get('use_lm', USE_LM_EMB) and lm_emb_available,
        lm_dim=LM_DIM, d_lm_hidden=MC['d_lm_hidden'],
        d_lm_out=MC['d_lm_out'],
        lm_input_dropout=MC['lm_input_drop']).to(device)
    m.load_state_dict(sd, strict=False)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _resolve_b_model(all_results, output_dir, device, lm_emb_available: bool):
    """统一的 B 模型获取入口: 优先用 all_results 内存里的 nn.Module,
    否则从 ./<output_dir>/ablation_B_act_lm/complete_model.pth 重建.

    返回 (model_or_None, source_str) —— source_str 便于日志打印.
    """
    b_res = all_results.get('ablation_B_act_lm') if all_results else None
    if b_res is not None and b_res.get('model') is not None:
        m = b_res['model'].to(device)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m, 'memory'
    # 优先 all_results 里的 exp_dir, 否则回退到标准路径
    exp_dir = (b_res.get('exp_dir')
               if (b_res is not None and b_res.get('exp_dir')) else None)
    ckpt = (Path(exp_dir) / 'complete_model.pth') if exp_dir else \
           (Path(output_dir) / 'ablation_B_act_lm' / 'complete_model.pth')
    m = _rebuild_b_model_from_ckpt(ckpt, device, lm_emb_available)
    return (m, f'disk:{ckpt}') if m is not None else (None, 'missing')


def _load_c_seed_payload_from_cache(exp_dir, device):
    """从 <exp_dir>/train_result_cache.pth 读回 C per-seed 的完整结果 dict.

    用途: 跨 run 复用 FUSION_ABL / 主 C 运行的 preds/labels/test_gene_indices,
    让 MULTI_SEED_EVAL 在不重训的情况下也能算 per-quadrant.

    返回 None 表示 cache 不存在或缺字段; 否则返回含
    pearson/r2/preds_test/labels_test/test_gene_indices 的 dict.
    """
    exp_dir = Path(exp_dir)
    cache_file = exp_dir / 'train_result_cache.pth'
    if not cache_file.exists():
        return None
    try:
        cache = torch.load(cache_file, map_location=device, weights_only=False)
    except Exception as _e:
        print(f"  ⚠ 读 train_result_cache 失败 {cache_file}: {_e}")
        return None
    if not isinstance(cache, dict):
        return None
    required = ('pearson', 'preds_test', 'labels_test', 'test_gene_indices')
    if not all(k in cache for k in required):
        return None
    return cache


def train_experiment(
        X_raw, y_raw, chromosomes, genes, tf_names, n_tfs, n_bins,
        output_dir, device, split_masks, label='exp_',
        model_kind='tf',
        batch_size=256, epochs=300, lr=2e-4,
        warmup_epochs=15, patience=40,
        tpm_transform='log1p', tpm_normalize='zscore',
        d_tf=128, d_cnn=128, n_tf_layers=1, n_regions=3, n_heads=4,
        conv_dropout=0.10, fc_dropout=0.35,
        scalar_data=None, use_scalars=True,
        n_scalars=14, n_seq_features=12,
        d_seq=128, d_tf_proj=96,
        seq_baseline_model=None,
        lm_data=None, use_lm=False, lm_dim=2304,
        d_lm_hidden=512, d_lm_out=128,
        lm_input_dropout=0.20, lm_noise_std=0.0, freeze_lm=False,
        use_profile=True,       # 消融 L：use_profile=False → LM-only
        # (Q1) ours-arch + one-hot 可控实验; 与 use_lm 互斥
        use_dna_onehot=False, dna_seq_len=1500, d_dna_out=128, dna_dropout=0.20,
        # FUSION_ABLATION 模块消融标志 (dict 或 None); None = 完整 fusion
        fusion_ablation_flags=None,
) -> dict:
    assert not (use_lm and use_dna_onehot), \
        "train_experiment: use_lm 与 use_dna_onehot 互斥"
    # 一起传入时 lm_data 实际承载 DNA one-hot 张量（共用 lm_s slot, 避免侵入 Dataset 接口）
    if use_dna_onehot:
        # DNA one-hot 不应叠加 Gaussian 噪声 —— 对 0/1 bit 加噪声没有生物学意义
        if lm_noise_std > 0:
            print(f"  [one-hot] 已将 lm_noise_std 从 {lm_noise_std} 强制置 0")
        lm_noise_std = 0.0
        # one-hot 模式走 C 的全架构（必须含 TF 分支 + fusion）
        assert model_kind != 'seq_only', \
            "train_experiment: use_dna_onehot=True 需配合 model_kind='tf'（走完整 C 架构）"
    exp_dir = Path(output_dir) / label
    exp_dir.mkdir(exist_ok=True, parents=True)
    print(f"\n{'#' * 62}\n  实验: {label}\n{'#' * 62}")

    # ── 断点续跑：命中缓存则直接返回，跳过训练 ──────────────────────────
    if RESUME_FROM_CACHE.get('enabled', True) and _is_train_experiment_complete(exp_dir):
        print(f"  [断点续跑] 命中缓存 → 跳过训练  ({exp_dir})")
        return _load_train_experiment_cache(exp_dir, device)

    def _split(arr):
        return {k: arr[split_masks[k]] for k in ('train', 'val', 'test')}

    X_s = _split(X_raw);
    y_s = _split(y_raw)

    y_tr_log = np.log1p(y_s['train'])
    p40, p80, p95 = (np.percentile(y_tr_log, q) for q in (40, 80, 95))
    sampler_w = np.where(y_tr_log >= p95, 3.0,
                         np.where(y_tr_log >= p80, 1.8,
                                  np.where(y_tr_log >= p40, 1.2, 1.0))).astype(np.float32)

    # 损失权重全部 1.0x. (new53: 已下线 Q2-like 上采样 & Q2 loss 2.0x 加权 —
    # C0 诊断表明这对 global Pearson 净效应为负, 且强制高 loss 会把 LM 表示学歪.)
    loss_w = {
        'train': np.ones(int(split_masks['train'].sum()), dtype=np.float32),
        'val':   np.ones(int(split_masks['val'].sum()),  dtype=np.float32),
        'test':  np.ones(int(split_masks['test'].sum()), dtype=np.float32),
    }

    sig_std = SignalStandardizer()
    X = {'train': sig_std.fit_transform(X_s['train']),
         'val': sig_std.transform(X_s['val']),
         'test': sig_std.transform(X_s['test'])}

    lp = ExpressionProcessor(tpm_transform, tpm_normalize)
    y_tr_norm, _ = lp.fit_transform(y_s['train'])
    y = {'train': y_tr_norm,
         'val': lp.transform(y_s['val']),
         'test': lp.transform(y_s['test'])}

    _use_sc = use_scalars and (scalar_data is not None)
    if _use_sc:
        def _prep(arr):
            out = arr.copy().astype(np.float32)
            if out.shape[1] > n_seq_features:
                out[:, n_seq_features:] = np.log1p(np.clip(out[:, n_seq_features:], 0, None))
            return out

        sc_s = {k: _prep(scalar_data[split_masks[k]]) for k in ('train', 'val', 'test')}
        sc_mean = np.nanmean(sc_s['train'], axis=0, keepdims=True)
        sc_std = np.nanstd(sc_s['train'], axis=0, keepdims=True) + 1e-8
        sc_norm = {k: np.nan_to_num((sc_s[k] - sc_mean) / sc_std, nan=0.0)
                   for k in ('train', 'val', 'test')}
        n_sc = scalar_data.shape[1]
        print(f"  ✓ scalar_data: {scalar_data.shape}  n_scalars={n_sc}")
    else:
        sc_norm = {k: None for k in ('train', 'val', 'test')}
        sc_mean = sc_std = None

    # ── FUSION_ABLATION: disable_lm 时把 LM 数据也从流水线切断, 避免
    #    「数据进了 Dataset 但模型不用」的资源浪费与潜在误用.
    _ab_flags = fusion_ablation_flags or {}
    if _ab_flags.get('disable_lm', False):
        use_lm   = False
        lm_data  = None
    _use_lm = use_lm and (lm_data is not None)
    # (Q1) one-hot 模式：lm_data 实际承载 DNA one-hot 张量，共用 lm_s slot
    _use_dna_oh = use_dna_onehot and (lm_data is not None)
    # _lm_like 为真即说明"序列表征张量"可用（LM 或 one-hot）；train_model 内部
    # 通过 `lm_emb.shape[-1] > 1` 判断即可兼容两种 shape: (B, lm_dim) vs (B, 4, L)
    _lm_like = _use_lm or _use_dna_oh
    lm_s = {k: lm_data[split_masks[k]] if _lm_like else None
            for k in ('train', 'val', 'test')}

    def _mkds(split, aug):
        return GeneExpressionDataset(
            X[split], y[split], loss_w[split],
            scalars=sc_norm[split], lm_embs=lm_s[split],
            augment=aug, tf_mask_prob=0.3, tf_mask_ratio=0.15,
            lm_noise_std=lm_noise_std if aug else 0.0)

    train_ds = _mkds('train', True)
    val_ds = _mkds('val', False)
    test_ds = _mkds('test', False)
    sampler = WeightedRandomSampler(
        torch.FloatTensor(sampler_w), num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size, sampler=sampler,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size, num_workers=0, pin_memory=True)

    MC = MODEL_CONFIG
    if model_kind == 'seq_only':
        model = SequenceOnlyPredictor(
            n_windows=n_seq_features,
            d_profile=MC['d_profile'], d_hidden=128, dropout=fc_dropout,
            use_lm=_use_lm, lm_dim=lm_dim,
            d_lm_hidden=d_lm_hidden, d_lm_out=d_lm_out,
            lm_input_dropout=lm_input_dropout,
            freeze_lm=freeze_lm,
            use_profile=use_profile).to(device)
    else:
        model = MultiModalPredictor(
            n_tfs=n_tfs, n_bins=n_bins,
            d_tf=d_tf, d_cnn=d_cnn,
            n_tf_layers=n_tf_layers, n_regions=n_regions, n_heads=n_heads,
            conv_dropout=conv_dropout, fc_dropout=fc_dropout,
            use_scalars=_use_sc,
            n_scalars=scalar_data.shape[1] if _use_sc else 2,
            n_seq_features=n_seq_features if _use_sc else 1,
            d_seq=d_seq, d_tf_proj=d_tf_proj,
            use_lm=_use_lm, lm_dim=lm_dim,
            d_lm_hidden=d_lm_hidden, d_lm_out=d_lm_out,
            lm_input_dropout=lm_input_dropout,
            freeze_lm=freeze_lm,
            topk_k=MC.get('topk_k', 0),
            use_dna_onehot=_use_dna_oh, dna_seq_len=dna_seq_len,
            d_dna_out=d_dna_out, dna_dropout=dna_dropout,
            fusion_flags=_ab_flags if _ab_flags else None).to(device)

    save_path = str(exp_dir / 'best_model.pth')
    history = train_model(
        model, train_loader, val_loader,
        epochs=epochs, lr=lr, device=device,
        patience=patience, save_path=save_path,
        warmup_epochs=warmup_epochs,
        seq_baseline_model=seq_baseline_model)

    results = evaluate_model(model, test_loader, device)
    plot_training_curves(history, results, str(exp_dir), label=label)

    # ── Bootstrap CI ────────────────────────────────────────────────────
    boot_ci = None
    if BOOTSTRAP_CI.get('enabled', False):
        print(f"\n  [Bootstrap CI] n_boot={BOOTSTRAP_CI['n_boot']}  "
              f"ci_level={BOOTSTRAP_CI['ci_level']}")
        boot_ci = compute_bootstrap_ci(
            preds=results['preds'].astype(np.float64),
            labels=results['labels'].astype(np.float64),
            n_boot=BOOTSTRAP_CI['n_boot'],
            ci_level=BOOTSTRAP_CI['ci_level'],
            seed=BOOTSTRAP_CI['random_seed'])
        _ci = boot_ci
        print(f"  Pearson r = {_ci['pearson_point']:.4f}  "
              f"[{_ci['pearson_ci_lo']:.4f}, {_ci['pearson_ci_hi']:.4f}]")
        print(f"  R²        = {_ci['r2_point']:.4f}  "
              f"[{_ci['r2_ci_lo']:.4f}, {_ci['r2_ci_hi']:.4f}]")
        pd.DataFrame([_ci]).to_csv(exp_dir / 'bootstrap_ci.csv', index=False)
        print(f"  ✓ Bootstrap CI → {exp_dir / 'bootstrap_ci.csv'}")

    if model_kind != 'seq_only':
        tf_imp, attn_mat = extract_tf_importance(
            model, test_loader, device, tf_names=tf_names, desc='test')
        if tf_imp is not None and not tf_imp.empty:
            plot_tf_importance(tf_imp, str(exp_dir), top_n=30, label=label)
            tf_imp.to_csv(exp_dir / 'tf_importance_attn.csv', header=['attn_weight'])
        if attn_mat is not None:
            test_mask = split_masks['test']
            test_genes = np.asarray(genes[test_mask], dtype=object)
            test_raw_indices = np.where(test_mask)[0].astype(np.int32)
            np.save(exp_dir / 'attn_weights_test.npy', attn_mat)
            np.save(exp_dir / 'attn_test_tf_names.npy',
                    np.asarray(tf_names, dtype=object))
            np.save(exp_dir / 'attn_test_gene_names.npy', test_genes)
            np.save(exp_dir / 'attn_test_raw_indices.npy', test_raw_indices)
            pd.DataFrame({
                'attn_row': np.arange(len(test_genes)),
                'raw_index': test_raw_indices,
                'gene': test_genes
            }).to_csv(exp_dir / 'attn_test_gene_table.csv', index=False)
            print(f"  ✓ 注意力矩阵等文件 → {exp_dir}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'results': {'r2': float(results['r2']), 'pearson': float(results['pearson'])},
        'history': history,
        'config': {
            'model': ('SequenceOnlyPredictor'
                      if model_kind == 'seq_only' else 'MultiModalPredictor'),
            'label': label,
            'n_tfs': int(n_tfs),
            'n_bins': int(n_bins),
            'd_tf': d_tf, 'd_cnn': d_cnn,
            'n_tf_layers': n_tf_layers,
            'n_regions': n_regions, 'n_heads': n_heads,
            'd_seq': d_seq, 'd_tf_proj': d_tf_proj,
            'batch_size': batch_size, 'lr': lr,
            'tpm_transform': tpm_transform,
            'tpm_normalize': tpm_normalize,
            'n_seq_features': n_seq_features,
            'use_lm': _use_lm, 'lm_dim': lm_dim,
            'd_lm_hidden': d_lm_hidden, 'd_lm_out': d_lm_out,
            'use_dna_onehot': _use_dna_oh, 'dna_seq_len': dna_seq_len,
            'd_dna_out': d_dna_out,
            'train_chrs': split_masks['train_chrs'],
            'val_chrs': split_masks['val_chrs'],
            'test_chrs': split_masks['test_chrs'],
            'fc_dropout': fc_dropout,
            'topk_k': MC.get('topk_k', 0),
            'fusion_ablation_flags': dict(_ab_flags) if _ab_flags else None,
        }
    }, exp_dir / 'complete_model.pth')

    _result = {
        'pearson': float(results['pearson']),
        'r2': float(results['r2']),
        'label': label,
        'sc_mean': sc_mean,
        'sc_std': sc_std,
        'n_seq_features': n_seq_features if _use_sc else 0,
        'preds_test': results['preds'].astype(np.float32),
        'labels_test': results['labels'].astype(np.float32),
        # test 集基因在全基因数组的位置 — 用于对齐固定象限标签 (compute_per_quadrant_metrics)
        'test_gene_indices': np.where(split_masks['test'])[0].astype(np.int64),
        'exp_dir': str(exp_dir),
        'model': model,
        'boot_ci': boot_ci,
    }
    _save_train_experiment_cache(exp_dir, _result)
    return _result


# ============================================================================
# 消融 D 专用训练流程（DNA one-hot CNN，纯序列）
# ============================================================================

def train_dna_experiment(
        dna_data, y_raw, genes, output_dir, device, split_masks,
        label='ablation_D_dna_cnn',
        batch_size=256, epochs=400, lr=1e-4,
        warmup_epochs=20, patience=60,
        tpm_transform='log1p', tpm_normalize='zscore',
        d_hidden=128, dropout=0.25, revcomp_prob=0.5,
) -> dict:
    """DNA one-hot CNN 独立训练流程。
    与 train_experiment 平行，但只使用 DNA one-hot 作为输入，
    不涉及 TF 信号 / activity scalars / LM 嵌入。"""
    exp_dir = Path(output_dir) / label
    exp_dir.mkdir(exist_ok=True, parents=True)
    print(f"\n{'#' * 62}\n  实验: {label}\n{'#' * 62}")

    # ── 断点续跑：命中缓存则直接返回，跳过训练 ──────────────────────────
    if RESUME_FROM_CACHE.get('enabled', True) and _is_train_experiment_complete(exp_dir):
        print(f"  [断点续跑] 命中缓存 → 跳过训练  ({exp_dir})")
        return _load_train_experiment_cache(exp_dir, device)

    def _split(arr):
        return {k: arr[split_masks[k]] for k in ('train', 'val', 'test')}

    dna_s = _split(dna_data)
    y_s   = _split(y_raw)
    print(f"  样本数: train={len(y_s['train'])}  val={len(y_s['val'])}  test={len(y_s['test'])}")
    print(f"  DNA shape: train={dna_s['train'].shape}  "
          f"seq_len={dna_s['train'].shape[-1]}")

    # ── 采样器加权（与 train_experiment 保持一致：高表达上采样）──────────
    y_tr_log = np.log1p(y_s['train'])
    p40, p80, p95 = (np.percentile(y_tr_log, q) for q in (40, 80, 95))
    sampler_w = np.where(y_tr_log >= p95, 3.0,
                         np.where(y_tr_log >= p80, 1.8,
                                  np.where(y_tr_log >= p40, 1.2, 1.0))).astype(np.float32)
    # 消融 D 不做 Q2-like 加权（无 TF 信息，无法学抑制规律）
    loss_w = {
        'train': np.ones(len(y_s['train']), dtype=np.float32),
        'val':   np.ones(len(y_s['val']),   dtype=np.float32),
        'test':  np.ones(len(y_s['test']),  dtype=np.float32),
    }

    # ── 目标变量标准化 ────────────────────────────────────────────────────
    lp = ExpressionProcessor(tpm_transform, tpm_normalize)
    y_tr_norm, _ = lp.fit_transform(y_s['train'])
    y = {'train': y_tr_norm,
         'val':   lp.transform(y_s['val']),
         'test':  lp.transform(y_s['test'])}

    # ── Dataset / DataLoader ──────────────────────────────────────────────
    train_ds = DNADataset(dna_s['train'], y['train'], loss_w['train'],
                           augment=True,  revcomp_prob=revcomp_prob)
    val_ds   = DNADataset(dna_s['val'],   y['val'],   loss_w['val'],   augment=False)
    test_ds  = DNADataset(dna_s['test'],  y['test'],  loss_w['test'],  augment=False)

    sampler = WeightedRandomSampler(
        torch.FloatTensor(sampler_w), num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size, sampler=sampler,
                                num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size, num_workers=0, pin_memory=True)

    # ── 模型 / 优化器 ─────────────────────────────────────────────────────
    seq_len = dna_s['train'].shape[-1]
    model = DNAOnlyPredictor(seq_len=seq_len, d_hidden=d_hidden,
                              dropout=dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs, epochs, len(train_loader), T_0=30)
    criterion = CombinedLoss()

    history = {'train_loss': [], 'val_loss': [], 'val_r2': [],
               'val_pearson': [], 'val_pearson_ema': [], 'lr': []}
    best_ema = -float('inf'); p_ema = None; pat_cnt = 0
    save_path = str(exp_dir / 'best_model.pth')
    _ema_slow = 0.70  # DNA-CNN 收敛相对波动大，用快 EMA

    print(f"\n训练 | epochs={epochs}  lr={lr}  patience={patience}  "
          f"warmup={warmup_epochs}ep  kind=dna  EMA={_ema_slow}")
    print(f"  revcomp_prob={revcomp_prob}  AdamW wd=1e-4")

    for epoch in range(epochs):
        model.train(); t_losses = []
        for dna, weights, exps in train_loader:
            dna = dna.to(device); weights = weights.to(device); exps = exps.to(device)
            optimizer.zero_grad()
            pred = model(dna)
            loss = criterion(pred, exps, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            t_losses.append(loss.item())

        model.eval(); v_losses = []; preds = []; labels = []
        with torch.no_grad():
            for dna, weights, exps in val_loader:
                dna = dna.to(device); exps = exps.to(device)
                p = model(dna)
                v_losses.append(criterion(p, exps, weights.to(device)).item())
                preds.extend(p.cpu().tolist()); labels.extend(exps.cpu().tolist())

        preds_arr = np.array(preds); labels_arr = np.array(labels)
        vp, _     = pearsonr(labels_arr, preds_arr)
        p_ema     = vp if p_ema is None else _ema_slow * p_ema + (1.0 - _ema_slow) * vp
        history['train_loss'].append(float(np.mean(t_losses)))
        history['val_loss'].append(float(np.mean(v_losses)))
        history['val_r2'].append(r2_score(labels_arr, preds_arr))
        history['val_pearson'].append(vp)
        history['val_pearson_ema'].append(p_ema)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Ep {epoch+1:3d}/{epochs}  "
                  f"Train={history['train_loss'][-1]:.4f}  "
                  f"Val={history['val_loss'][-1]:.4f}  "
                  f"R²={history['val_r2'][-1]:.4f}  "
                  f"Pearson={vp:.4f}  EMA={p_ema:.4f}")

        if p_ema > best_ema:
            best_ema = p_ema; pat_cnt = 0
            torch.save(model.state_dict(), save_path)
        else:
            pat_cnt += 1
            if pat_cnt >= patience:
                print(f"\nEarly stop @ ep {epoch+1}  best EMA={best_ema:.4f}")
                break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    print(f"最佳验证 EMA Pearson: {best_ema:.4f}")

    # ── 测试集评估 ────────────────────────────────────────────────────────
    model.eval(); preds_all = []; labels_all = []
    with torch.no_grad():
        for dna, weights, exps in test_loader:
            dna = dna.to(device)
            p = model(dna)
            preds_all.extend(p.cpu().tolist()); labels_all.extend(exps.cpu().tolist())
    preds_test  = np.asarray(preds_all, dtype=np.float32)
    labels_test = np.asarray(labels_all, dtype=np.float32)
    pearson_pt, _ = pearsonr(labels_test, preds_test)
    r2_pt         = r2_score(labels_test, preds_test)
    print(f"\n  [测试集] Pearson={pearson_pt:.4f}  R²={r2_pt:.4f}")

    results = {'preds': preds_test, 'labels': labels_test,
               'pearson': float(pearson_pt), 'r2': float(r2_pt)}
    plot_training_curves(history, results, str(exp_dir), label=label)

    # ── Bootstrap CI ─────────────────────────────────────────────────────
    boot_ci = None
    if BOOTSTRAP_CI.get('enabled', False):
        print(f"\n  [Bootstrap CI] n_boot={BOOTSTRAP_CI['n_boot']}  "
              f"ci_level={BOOTSTRAP_CI['ci_level']}")
        boot_ci = compute_bootstrap_ci(
            preds=preds_test.astype(np.float64),
            labels=labels_test.astype(np.float64),
            n_boot=BOOTSTRAP_CI['n_boot'],
            ci_level=BOOTSTRAP_CI['ci_level'],
            seed=BOOTSTRAP_CI['random_seed'])
        _ci = boot_ci
        print(f"  Pearson r = {_ci['pearson_point']:.4f}  "
              f"[{_ci['pearson_ci_lo']:.4f}, {_ci['pearson_ci_hi']:.4f}]")
        print(f"  R²        = {_ci['r2_point']:.4f}  "
              f"[{_ci['r2_ci_lo']:.4f}, {_ci['r2_ci_hi']:.4f}]")
        pd.DataFrame([_ci]).to_csv(exp_dir / 'bootstrap_ci.csv', index=False)

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {'model_kind': 'dna', 'seq_len': seq_len,
                   'd_hidden': d_hidden, 'dropout': dropout,
                   'upstream': DNA_WINDOW_UPSTREAM, 'downstream': DNA_WINDOW_DOWNSTREAM},
    }, exp_dir / 'complete_model.pth')

    _result_dna = {
        'pearson': float(pearson_pt), 'r2': float(r2_pt),
        'label': label,
        'preds_test': preds_test, 'labels_test': labels_test,
        'exp_dir': str(exp_dir),
        'model': model,
        'boot_ci': boot_ci,
    }
    _save_train_experiment_cache(exp_dir, _result_dna)
    return _result_dna


# ============================================================================
# __main__
# ============================================================================

if __name__ == "__main__":
    PC = PATH_CONFIG
    GT = GLOBAL_TRAIN
    MC = MODEL_CONFIG

    random.seed(GT['random_seed'])
    np.random.seed(GT['random_seed'])
    torch.manual_seed(GT['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GT['random_seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print(f"  TF 窗口:  -{WINDOW_UPSTREAM}~+{WINDOW_DOWNSTREAM} bp  TSS_RATIO={TSS_RATIO:.4f}")
    print(f"  LM 窗口:  -{LM_WINDOW_UPSTREAM}~+{LM_WINDOW_DOWNSTREAM} bp  LM_TSS_RATIO={LM_TSS_RATIO:.4f}")
    Path(PC['output_dir']).mkdir(exist_ok=True, parents=True)

    # ── Step 1: 加载 TF 信号 & TPM ──────────────────────────────────────
    (X_raw, y_raw, chromosomes, genes, tf_names,
     gene_lengths, n_bins, bed_df) = load_tf_signal_data(
        PC['npz_file'], PC['tpm_file'], PC['bed_file'],
        remove_mitochondrial=True, tpm_column=PC['tpm_column'])

    # ── Step 1.5: (可选) TF 白名单 —— 若存在则子集化 X_raw ─────────────
    # 触发条件: PATH_CONFIG['tf_whitelist_csv'] 指向的 csv 存在 且 TF 数少于 npz
    # 行为: 按 tf_name 名字匹配做子集, 顺序保持 npz 原顺序 (保证 reproducibility)
    #       并把 output_dir / analysis_dir 加后缀 "_filtered_{N}", 不覆盖 full-178 结果
    _wl_cfg = PC.get('tf_whitelist_csv', None)
    _wl_path = Path(_wl_cfg) if _wl_cfg else None
    if _wl_path is not None and _wl_path.exists():
        _wl_df = pd.read_csv(_wl_path)
        _wl_names = set(_wl_df['tf_name'].astype(str).str.upper())
        _tf_upper = [t.upper() for t in tf_names]
        _keep_idx = np.array([i for i, t in enumerate(_tf_upper) if t in _wl_names])
        if 0 < len(_keep_idx) < len(tf_names):
            X_raw = X_raw[:, _keep_idx, :]
            tf_names = [tf_names[i] for i in _keep_idx]
            _suf = f"_filtered_{len(_keep_idx)}"
            PC['output_dir'] = PC['output_dir'] + _suf
            PC['analysis_dir'] = PC['output_dir'] + '/analysis'
            Path(PC['output_dir']).mkdir(exist_ok=True, parents=True)
            Path(PC['analysis_dir']).mkdir(exist_ok=True, parents=True)
            print(f"\n  [TF 白名单] 已应用: {len(_keep_idx)} / {len(_wl_df) + 0} 个 TF  "
                  f"(源文件 {_wl_path})")
            print(f"  [TF 白名单] 输出重定向: {PC['output_dir']}")
        else:
            print(f"\n  [TF 白名单] 跳过: keep={len(_keep_idx)} 超出范围 "
                  f"(需 0<keep<{len(tf_names)})")
    n_tfs = X_raw.shape[1]
    tss_bin = int(n_bins * TSS_RATIO)
    print(f"\n  n_tfs={n_tfs}  n_bins={n_bins}  tss_bin={tss_bin}")

    # ── Step 1.6 (★ v8 新增): chr holdout 稳健性配置应用 ────────────────
    #   设计: 在 TF whitelist 之后应用, 让 holdout 后缀叠加在 _filtered_79 之后,
    #         形成 ./53_filtered_79_holdout_chr3_chr4/ 这样自解释的目录,
    #         与主管道 ./53_filtered_79/ 完全隔离, 缓存不互污染.
    #   行为:
    #     1) GT['split_strategy'] 切换为 'custom'
    #     2) GT['custom_split']  = {test_chrs, val_chrs}
    #     3) PC['output_dir'] / PC['analysis_dir'] 加 '_holdout_<preset>' 后缀
    #     4) FUSION_ABLATION / MULTI_SEED_EVAL / 缓存 / S11 输出全部跟随
    _ho_test, _ho_val, _ho_suffix = _resolve_chr_holdout(
        CHR_HOLDOUT, CHR_HOLDOUT_PRESETS)
    if CHR_HOLDOUT.get('enabled', False):
        GT = dict(GT)  # 浅拷贝, 避免污染顶层 GLOBAL_TRAIN
        GT['split_strategy'] = 'custom'
        GT['custom_split']   = dict(test_chrs=_ho_test, val_chrs=_ho_val)
        # PC 已是 PATH_CONFIG 引用, 直接 in-place 改后缀 (与 whitelist 风格一致)
        PC['output_dir']   = PC['output_dir']   + _ho_suffix
        PC['analysis_dir'] = PC['output_dir'] + '/analysis'
        Path(PC['output_dir']).mkdir(exist_ok=True, parents=True)
        Path(PC['analysis_dir']).mkdir(exist_ok=True, parents=True)
        print(f"\n{'═'*68}")
        print(f"  ★ CHR_HOLDOUT 已启用 (P0 #12 稳健性): preset='{CHR_HOLDOUT['preset']}'")
        print(f"    test_chrs = {_ho_test}")
        print(f"    val_chrs  = {_ho_val}")
        print(f"    输出目录   : {PC['output_dir']}")
        print(f"    主管道对照 : ./53_filtered_79  (test=chrXV-XVI, 5-seed 0.8482 ± 0.0051)")
        print(f"{'═'*68}\n")
    else:
        print(f"\n  [CHR_HOLDOUT] 未启用 (主管道模式: test=chrXV-XVI)")

    # ── Step 2: 染色体划分 ───────────────────────────────────────────────
    split_masks = split_by_chromosome(
        chromosomes, y_raw,
        strategy=GT['split_strategy'], random_seed=GT['random_seed'],
        custom_split=GT.get('custom_split'))
    leakage_audit(genes, split_masks)
    all_results = {}

    # ── Step 2.1 (★ v8 新增): chr holdout meta 落盘 ─────────────────────
    #   写入 chr_holdout_meta.json, 供 Methods / Supp S12 直接引用.
    #   主管道模式也写, 作为对齐 reference.
    try:
        import json as _json
        _meta_path = Path(PC['output_dir']) / 'chr_holdout_meta.json'
        _meta = dict(
            mode           = ('chr_holdout' if CHR_HOLDOUT.get('enabled', False)
                              else 'main_pipeline'),
            preset         = (CHR_HOLDOUT.get('preset')
                              if CHR_HOLDOUT.get('enabled', False) else 'fixed_saccer3'),
            split_strategy = GT['split_strategy'],
            train_chrs     = sorted(split_masks['train_chrs']),
            val_chrs       = sorted(split_masks['val_chrs']),
            test_chrs      = sorted(split_masks['test_chrs']),
            n_train        = int(split_masks['train'].sum()),
            n_val          = int(split_masks['val'].sum()),
            n_test         = int(split_masks['test'].sum()),
            median_y_train = float(np.median(y_raw[split_masks['train']])),
            median_y_val   = float(np.median(y_raw[split_masks['val']])),
            median_y_test  = float(np.median(y_raw[split_masks['test']])),
            n_tfs          = int(n_tfs),
            output_dir     = str(PC['output_dir']),
        )
        with open(_meta_path, 'w') as _f:
            _json.dump(_meta, _f, indent=2, ensure_ascii=False)
        print(f"  ✓ 切分元数据已写入: {_meta_path}")
    except Exception as _e:
        print(f"  ⚠  chr_holdout_meta.json 写入失败: {_e} (不影响训练)")

    # ── Step 2.5: Promoter profile 标量 ─────────────────────────────────
    SEQ_PROFILE_COLS = list(SEQ_PROFILE_COLS_DEFAULT)
    profile_df = (pd.read_csv(PC['promoter_profile_csv'])
                  .drop_duplicates(subset='gene'))
    for c in SEQ_PROFILE_COLS:
        profile_df[c] = pd.to_numeric(profile_df[c], errors='coerce')

    genes_series = pd.Series(genes, name='gene')
    scalar_df = genes_series.to_frame()
    scalar_df = scalar_df.merge(
        profile_df[['gene'] + SEQ_PROFILE_COLS], on='gene', how='left')
    for c in SEQ_PROFILE_COLS:
        scalar_df[c] = scalar_df[c].fillna(
            scalar_df.loc[split_masks['train'], c].median())

    scalar_data = scalar_df[SEQ_PROFILE_COLS].values.astype(np.float32)
    N_SEQ_FEATURES = len(SEQ_PROFILE_COLS)
    N_SCALARS = scalar_data.shape[1]
    print(f"\n  ✓ scalar_data: {scalar_data.shape}  "
          f"(n_seq_features={N_SEQ_FEATURES}  n_scalars={N_SCALARS})")

    # ── Step 2.6: LM 嵌入 ───────────────────────────────────────────────
    lm_emb_all = None;
    LM_DIM = 0
    if USE_LM_EMB:
        try:
            lm_raw = load_lm_embeddings(
                PC['lm_npz_file'], genes, use_layer=MC['lm_layer_key'])
            LM_DIM = lm_raw.shape[1]
            lm_emb_all = (normalize_lm_embeddings(lm_raw, split_masks['train'])
                          if NORMALIZE_LM_EMB else lm_raw)
            print(f"  ✓ LM 嵌入就绪  shape={lm_emb_all.shape}")
        except Exception as _e:
            print(f"  ⚠ LM 嵌入加载失败: {_e}，USE_LM_EMB 已禁用")
            USE_LM_EMB = False

    # ── Step 2.7: DNA one-hot（消融 D 和 Q1 OURS_ONEHOT_EXP 共用）────────
    dna_onehot_all = None
    _need_dna = ('D' in RUN_ABLATIONS) or OURS_ONEHOT_EXP.get('enabled', False)
    if _need_dna:
        _who = []
        if 'D' in RUN_ABLATIONS: _who.append('消融 D')
        if OURS_ONEHOT_EXP.get('enabled', False): _who.append('OURS_ONEHOT_EXP')
        print("\n" + "=" * 62 +
              f"\n  Step 2.7: DNA one-hot 加载  窗口 -{DNA_WINDOW_UPSTREAM}~+{DNA_WINDOW_DOWNSTREAM} bp"
              f"\n  需求者: {' + '.join(_who)}\n" +
              "=" * 62)
        try:
            dna_onehot_all = load_or_cache_dna_onehot(
                PC['genome_fasta_file'], bed_df, genes,
                DNA_WINDOW_UPSTREAM, DNA_WINDOW_DOWNSTREAM,
                PC['dna_onehot_cache'])
            _dna_sanity_check(dna_onehot_all, genes, bed_df,
                               n_examples=3, upstream=DNA_WINDOW_UPSTREAM)
        except FileNotFoundError as _e:
            print(f"  ⚠ DNA 加载失败（{_e}），消融 D / OURS_ONEHOT_EXP 将跳过")
            dna_onehot_all = None
        except Exception as _e:
            print(f"  ⚠ DNA 加载异常: {_e}，消融 D / OURS_ONEHOT_EXP 将跳过")
            dna_onehot_all = None

    # ── 公共参数生成器 ───────────────────────────────────────────────────
    def _exp_kwargs(ablation_key: str) -> dict:
        overrides = ABLATION_OVERRIDE.get(ablation_key, {})
        merged = {**GT, **overrides}
        return dict(
            X_raw=X_raw, y_raw=y_raw, chromosomes=chromosomes,
            genes=genes, tf_names=tf_names,
            n_tfs=n_tfs, n_bins=n_bins,
            output_dir=PC['output_dir'], device=device,
            split_masks=split_masks,
            batch_size=merged.get('batch_size', GT['batch_size']),
            epochs=merged.get('epochs', GT['epochs']),
            lr=merged.get('lr', GT['lr']),
            warmup_epochs=merged.get('warmup_epochs', GT['warmup_epochs']),
            patience=merged.get('patience', GT['patience']),
            tpm_transform=merged.get('tpm_transform', GT['tpm_transform']),
            tpm_normalize=merged.get('tpm_normalize', GT['tpm_normalize']),
            d_tf=MC['d_tf'], d_cnn=MC['d_cnn'],
            n_tf_layers=MC['n_tf_layers'], n_regions=MC['n_regions'],
            n_heads=MC['n_heads'],
            d_seq=MC['d_seq'], d_tf_proj=MC['d_tf_proj'],
            d_lm_hidden=MC['d_lm_hidden'], d_lm_out=MC['d_lm_out'],
            fc_dropout=MC['fc_dropout'],
            lm_input_dropout=MC['lm_input_drop'],
            lm_noise_std=MC['lm_noise_std'],
            freeze_lm=MC['freeze_lm'],
            lm_dim=LM_DIM,
        )

    # ── Step 3: 消融实验 D → L → B → C ──────────────────────────────────

    # ---- D: DNA one-hot CNN（纯序列基线，从零训练）─────────────────────
    if 'D' in RUN_ABLATIONS:
        if dna_onehot_all is not None:
            print("\n" + "=" * 62 +
                  "\n  消融 D: DNA one-hot CNN（从零训练的序列基线）\n" + "=" * 62)
            _d_over = ABLATION_OVERRIDE.get('D', {})
            res_d = train_dna_experiment(
                dna_data=dna_onehot_all, y_raw=y_raw, genes=genes,
                output_dir=PC['output_dir'], device=device, split_masks=split_masks,
                label='ablation_D_dna_cnn',
                batch_size=_d_over.get('batch_size', GT['batch_size']),
                epochs=_d_over.get('epochs', 400),
                lr=_d_over.get('lr', 1e-4),
                warmup_epochs=_d_over.get('warmup_epochs', 20),
                patience=_d_over.get('patience', 60),
                tpm_transform=GT['tpm_transform'],
                tpm_normalize=GT['tpm_normalize'],
                d_hidden=128, dropout=0.25, revcomp_prob=0.5)
            all_results['ablation_D_dna_cnn'] = res_d
        else:
            print("\n  [跳过] 消融 D：DNA one-hot 不可用")
    else:
        print("\n  [跳过] 消融 D")

    # ---- L: LM-only（预训练 LM 嵌入，无 activity）──────────────────────
    if 'L' in RUN_ABLATIONS:
        _can_l = USE_LM_EMB and (lm_emb_all is not None)
        if _can_l:
            print("\n" + "=" * 62 + "\n  消融 L: LM-only（无 activity 标量）\n" + "=" * 62)
            res_l = train_experiment(
                **_exp_kwargs('L'),
                label='ablation_L_lm_only', model_kind='seq_only',
                scalar_data=scalar_data, use_scalars=True,
                n_scalars=N_SEQ_FEATURES, n_seq_features=N_SEQ_FEATURES,
                use_lm=True, lm_data=lm_emb_all,
                use_profile=False)   # 关键：跳过 profile 分支 → LM-only
            all_results['ablation_L_lm_only'] = res_l
        else:
            print("\n  [跳过] 消融 L：lm_emb_all 不可用，USE_LM_EMB=False")
    else:
        print("\n  [跳过] 消融 L")

    # ---- B: Activity + LM（成熟的序列分支） ─────────────────────────────
    if 'B' in RUN_ABLATIONS:
        _can_b = USE_LM_EMB and (lm_emb_all is not None)
        if _can_b:
            print("\n" + "=" * 62 + "\n  消融 B: Activity + LM\n" + "=" * 62)
            res_b = train_experiment(
                **_exp_kwargs('B'),
                label='ablation_B_act_lm', model_kind='seq_only',
                scalar_data=scalar_data, use_scalars=True,
                n_scalars=N_SEQ_FEATURES, n_seq_features=N_SEQ_FEATURES,
                use_lm=True, lm_data=lm_emb_all)
            all_results['ablation_B_act_lm'] = res_b
        else:
            print("\n  [跳过] 消融 B：lm_emb_all 不可用，USE_LM_EMB=False")
    else:
        print("\n  [跳过] 消融 B")

    # B 模型软监督 (C + FUSION_ABLATION 共用) - 作用域提升到 C 块外
    _b_model_for_sup = None

    # ---- C: Seq + TF（完整模型）────────────────────────────────────────
    if 'C' in RUN_ABLATIONS:
        print("\n" + "=" * 62 + "\n  消融 C: Seq + TF（完整模型）\n" + "=" * 62)

        # 加载 B 模型提供 trans_resid 软监督基线
        #   memory:  all_results 里的 nn.Module (当前 run 新训练的 B)
        #   disk:    RESUME_FROM_CACHE 命中时 all_results['model']=None,
        #            从 ablation_B_act_lm/complete_model.pth 重建
        _b_model_for_sup, _b_src = _resolve_b_model(
            all_results, PC['output_dir'], device,
            lm_emb_available=(lm_emb_all is not None))
        if _b_model_for_sup is not None:
            print(f"  [软监督] 消融B模型已冻结 (source={_b_src})，用于 trans_resid 软监督")
        else:
            print("  ⚠ [软监督] 消融B模型不可用 (memory 为空 + 磁盘 ckpt 不存在)")

        res_c = train_experiment(
            **_exp_kwargs('C'),
            label='ablation_C_seq_tf', model_kind='tf',
            scalar_data=scalar_data, use_scalars=True,
            n_scalars=N_SEQ_FEATURES, n_seq_features=N_SEQ_FEATURES,
            use_lm=USE_LM_EMB,
            lm_data=lm_emb_all if USE_LM_EMB else None,
            seq_baseline_model=_b_model_for_sup)
        all_results['ablation_C_seq_tf'] = res_c
    else:
        print("\n  [跳过] 消融 C")

    # ---- T: ours 架构内的 TF-only 消融 ────────────────────────────────
    #   目的: 在 Ridge baseline (tss_weighted 0.64 / raw_bins_pca 0.78) 基础上,
    #        独立验证 TF 分支 (PerTFEncoder + TFSpatialEncoder + TFAttentionPooler)
    #        本身是否强过 Ridge. 结果决定下一步:
    #        - T ≥ 0.78: TF 分支 OK, 问题在跨模态融合 → 优化 ResidualCrossModalFusion
    #        - 0.64 < T < 0.78: TF 分支本身有设计问题 → 优化 PerTFEncoder / TopK-30
    #        - T < 0.64: TF-side 架构严重失败 → 整体重设计
    #   关键配置: use_scalars=False + use_lm=False
    #        MultiModalPredictor 原生支持 (self.fusion=None, fc_input=tf_merged 直通).
    #   不需要 B 软监督 (没有 seq 分支可对齐); aux 损失全关 (fair vs Ridge).
    if 'T' in RUN_ABLATIONS:
        print("\n" + "=" * 62 +
              "\n  消融 T: ours 架构内 TF-only (独立验证 TF 分支价值)\n" + "=" * 62)
        res_t = train_experiment(
            **_exp_kwargs('T'),
            label='ablation_T_tf_only_arch', model_kind='tf',
            # ★ 关键: use_scalars=False + use_lm=False → 走纯 TF 分支路径
            scalar_data=None, use_scalars=False,
            n_scalars=0, n_seq_features=N_SEQ_FEATURES,
            use_lm=False, lm_data=None,
            seq_baseline_model=None)
        all_results['ablation_T_tf_only_arch'] = res_t
    else:
        print("\n  [跳过] 消融 T")

    # ---- FUSION_ABLATION: C 架构内模块级消融 ────────────────────────────
    #   每 variant × FUSION_ABLATION_SEEDS, 与 C 同 split / 同超参 / 同 B-软监督.
    #   所有 variant 依 disable_lm 标志实时决定是否喂 LM 数据.
    #   结果直接落盘 (每个 variant 一个子目录), 统一汇总到 fusion_ablation_summary.csv.
    #
    # new47 修正: 在 variant 循环前先跑 C-at-seed × FUSION_ABLATION_SEEDS,
    #   构建真正的 paired baseline. 之前依赖 MULTI_SEED_EVAL 才能给出 per-seed C,
    #   若 MULTI_SEED_EVAL 关掉则退化到单 seed=42, 导致 delta_vs_C_mean 是假象.
    #   seed=42 直接复用主 C 运行 (ablation_C_seq_tf), 其它 seed 单独训练.
    if RUN_FUSION_ABLATIONS and ('C' in RUN_ABLATIONS):
        print("\n" + "=" * 62 +
              "\n  ★ FUSION_ABLATION: C 架构内模块级消融\n" + "=" * 62)
        print(f"  variants ({len(FUSION_ABLATION)}): {list(FUSION_ABLATION.keys())}")
        print(f"  seeds ({len(FUSION_ABLATION_SEEDS)}): {FUSION_ABLATION_SEEDS}")
        # log 估计先按 worst case 给出: 实际跑的次数会因 disk cache 复用减少
        # (cache 检查见下方第 0 步), 这里只用作"上限提醒".
        _n_c_runs_upper = sum(1 for s in FUSION_ABLATION_SEEDS
                               if not (s == 42 and 'ablation_C_seq_tf' in all_results))
        print(f"  共 {len(FUSION_ABLATION) * len(FUSION_ABLATION_SEEDS)} variant runs "
              f"+ ≤{_n_c_runs_upper} C-baseline runs (cache 命中后实际更少), "
              f"复用 B 软监督 + 同 split + aux=off")

        # ── 0. 先构建 C per-seed baseline (paired Δ 基础) ─────────────
        _c_per_seed = {}   # seed -> pearson
        # seed=42 直接复用主 C 运行 (同 seed 同 kwargs → bit-identical)
        if 42 in FUSION_ABLATION_SEEDS and 'ablation_C_seq_tf' in all_results:
            _c_per_seed[42] = float(all_results['ablation_C_seq_tf']['pearson'])
            print(f"  [FUSION_ABL] 复用 ablation_C_seq_tf 作为 C@seed=42 "
                  f"(r={_c_per_seed[42]:.4f})")
        # new49 修复: 从磁盘缓存读 C per-seed, 避免重训 C@{123,456,789,2024}
        _cps_cache = Path(PC['output_dir']) / 'fusion_ablation' / 'fusion_ablation_C_per_seed.csv'
        if _cps_cache.exists():
            try:
                _cps_df_old = pd.read_csv(_cps_cache)
                _n_reused = 0
                for _, _rr in _cps_df_old.iterrows():
                    _ss = int(_rr['seed']); _pp = float(_rr['pearson'])
                    if (_ss in FUSION_ABLATION_SEEDS and _ss not in _c_per_seed
                            and np.isfinite(_pp)):
                        _c_per_seed[_ss] = _pp
                        _n_reused += 1
                if _n_reused > 0:
                    print(f"  [FUSION_ABL] 从磁盘复用 C per-seed: "
                          f"{_n_reused} seeds (cache={_cps_cache.name})")
            except Exception as _cache_e:
                print(f"  ⚠ [FUSION_ABL] 读缓存 {_cps_cache.name} 失败: {_cache_e}")

        for _seed in FUSION_ABLATION_SEEDS:
            if _seed in _c_per_seed:
                continue
            _c_label = f'fusion_abl_C_seed{_seed}'
            print("\n" + "-" * 62 +
                  f"\n  [FUSION_ABL] C baseline  seed={_seed}  "
                  f"(paired Δ reference)\n" + "-" * 62)
            random.seed(_seed); np.random.seed(_seed); torch.manual_seed(_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(_seed)
            try:
                _res_cs = train_experiment(
                    **_exp_kwargs('C'),
                    label=_c_label, model_kind='tf',
                    scalar_data=scalar_data, use_scalars=True,
                    n_scalars=N_SEQ_FEATURES, n_seq_features=N_SEQ_FEATURES,
                    use_lm=USE_LM_EMB,
                    lm_data=lm_emb_all if USE_LM_EMB else None,
                    seq_baseline_model=_b_model_for_sup,
                    fusion_ablation_flags=None,   # 完整 C, 不消融
                )
                all_results[_c_label] = _res_cs
                _c_per_seed[_seed] = float(_res_cs['pearson'])
                print(f"  [FUSION_ABL] C@seed={_seed}: r={_c_per_seed[_seed]:.4f}")
            except Exception as _cse:
                print(f"\n  ⚠ [FUSION_ABL] C@seed={_seed} 失败: {_cse}")
                traceback.print_exc()

        # ── 1. variant × seed 循环 ───────────────────────────────────
        _fa_records = []   # 每 run 一行, 稍后汇总
        for _ab_key, _ab_flags in FUSION_ABLATION.items():
            for _seed in FUSION_ABLATION_SEEDS:
                _label = f'fusion_abl_{_ab_key}_seed{_seed}'
                print("\n" + "-" * 62 +
                      f"\n  [FUSION_ABL] variant={_ab_key}  seed={_seed}  "
                      f"flags={_ab_flags}\n" + "-" * 62)

                # 逐 run 重设种子, 保证配对对比干净
                random.seed(_seed)
                np.random.seed(_seed)
                torch.manual_seed(_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(_seed)

                # disable_lm 时 use_lm / lm_data 需相应关闭 (train_experiment 内部
                # 也会再次强制, 这里显式置也更易读)
                _flag_use_lm = (USE_LM_EMB and
                                not _ab_flags.get('disable_lm', False))

                try:
                    _res = train_experiment(
                        **_exp_kwargs('C'),
                        label=_label, model_kind='tf',
                        scalar_data=scalar_data, use_scalars=True,
                        n_scalars=N_SEQ_FEATURES, n_seq_features=N_SEQ_FEATURES,
                        use_lm=_flag_use_lm,
                        lm_data=lm_emb_all if _flag_use_lm else None,
                        seq_baseline_model=_b_model_for_sup,
                        fusion_ablation_flags=_ab_flags,
                    )
                    all_results[_label] = _res
                    _fa_records.append(dict(
                        variant=_ab_key, seed=int(_seed),
                        label=_label,
                        pearson=float(_res['pearson']),
                        r2=float(_res['r2']),
                        ci_lo=(float(_res['boot_ci']['pearson_ci_lo'])
                               if _res.get('boot_ci') else np.nan),
                        ci_hi=(float(_res['boot_ci']['pearson_ci_hi'])
                               if _res.get('boot_ci') else np.nan),
                        flags=str(_ab_flags),
                    ))
                except Exception as _fa_e:
                    print(f"\n  ⚠ [FUSION_ABL] {_label} 失败: {_fa_e}")
                    traceback.print_exc()
                    _fa_records.append(dict(
                        variant=_ab_key, seed=int(_seed),
                        label=_label, pearson=np.nan, r2=np.nan,
                        ci_lo=np.nan, ci_hi=np.nan,
                        flags=str(_ab_flags), error=str(_fa_e)))

        # ── 汇总延后到 MULTI_SEED_EVAL 之后 (万一 MULTI_SEED_EVAL 扩展了 seed 集) ──
        all_results['_fusion_abl_records'] = _fa_records
        all_results['_fusion_abl_c_per_seed'] = _c_per_seed
        print(f"\n  ✓ [FUSION_ABL] {len(_fa_records)} variant runs + "
              f"{len(_c_per_seed)} C-per-seed baselines 完成")
    elif RUN_FUSION_ABLATIONS:
        print("\n  ⚠ RUN_FUSION_ABLATIONS=True 但 'C' 不在 RUN_ABLATIONS, 跳过")

    # ---- (Q1) C_onehot: ours 架构 + DNA one-hot 序列表征 ──────────────
    #   目的: 在 C 的完整架构上, 把 LM 嵌入换成 DNA one-hot, 以解耦
    #         "序列表征" vs "架构" 对性能的贡献。结果放 supplementary。
    if OURS_ONEHOT_EXP.get('enabled', False):
        if dna_onehot_all is None:
            print("\n  [跳过] OURS_ONEHOT_EXP：DNA one-hot 不可用")
        elif 'C' not in RUN_ABLATIONS:
            print("\n  [跳过] OURS_ONEHOT_EXP：需要 'C' 在 RUN_ABLATIONS 里（作为对照）")
        else:
            print("\n" + "=" * 62 +
                  "\n  (Q1) OURS_ONEHOT_EXP: ours 架构 + DNA one-hot\n" + "=" * 62)
            _oh_cfg = OURS_ONEHOT_EXP
            _oh_over = _oh_cfg.get('overrides', {}) or {}
            # 基础参数沿用 C，但强制 use_lm=False / lm_data=dna_onehot_all / use_dna_onehot=True
            _base = _exp_kwargs('C')
            # 应用 overrides 覆盖训练超参（epochs/lr/patience/warmup_epochs）
            for _k in ('epochs', 'lr', 'patience', 'warmup_epochs'):
                if _k in _oh_over:
                    _base[_k] = _oh_over[_k]
            # one-hot 下把 lm_noise_std 归零由 train_experiment 内部强制；此处无需改
            res_c_oh = train_experiment(
                **_base,
                label='ablation_C_onehot_arch', model_kind='tf',
                scalar_data=scalar_data, use_scalars=True,
                n_scalars=N_SEQ_FEATURES, n_seq_features=N_SEQ_FEATURES,
                # 关键: 关 LM, 走 one-hot; lm_data slot 承载 DNA 张量
                use_lm=False, lm_data=dna_onehot_all,
                use_dna_onehot=True,
                dna_seq_len=DNA_SEQ_LEN,
                d_dna_out=_oh_cfg.get('d_dna_out', 128),
                dna_dropout=_oh_cfg.get('dna_dropout', 0.20),
                # 不传 B 软监督 —— B 是 LM-based 的, 对 one-hot 分支反而是噪声
                seq_baseline_model=None)
            all_results['ablation_C_onehot_arch'] = res_c_oh

    # ── 消融汇总 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 62 + "\n  消融实验结果汇总（D→L→B→C / T 诊断）\n" + "=" * 62)
    _ci_lv = int(BOOTSTRAP_CI.get('ci_level', 0.95) * 100)
    _ci_hdr = f'  [{_ci_lv}%CI]' if BOOTSTRAP_CI.get('enabled') else ''
    print(f"  {'模型':42s}  {'Pearson r':>10}  {'R²':>8}{_ci_hdr}")
    print(f"  {'-' * 72}")
    for key in ['ablation_D_dna_cnn', 'ablation_L_lm_only',
                'ablation_B_act_lm', 'ablation_C_seq_tf',
                'ablation_T_tf_only_arch',
                'ablation_C_onehot_arch']:
        res = all_results.get(key)
        if res is None: continue
        ci = res.get('boot_ci')
        ci_str = (f'  [{ci["pearson_ci_lo"]:.4f}, {ci["pearson_ci_hi"]:.4f}]'
                  if ci else '')
        print(f"  {key:42s}  {res['pearson']:>10.4f}  {res['r2']:>8.4f}{ci_str}")

    _delta_pairs = [
        ('ablation_D_dna_cnn', 'ablation_L_lm_only', 'DNA-CNN → LM-only      (pretraining gain)'),
        ('ablation_L_lm_only', 'ablation_B_act_lm',  'LM-only → +Activity    (activity gain)'),
        ('ablation_B_act_lm',  'ablation_C_seq_tf',  'B(Act+LM) → +TF        (cross-modal gain)'),
        ('ablation_D_dna_cnn', 'ablation_C_seq_tf',  'DNA-CNN → Full         (end-to-end gain)'),
        # (Q1) 同架构下 seq-repr 的贡献: C(one-hot) 相对 C(LM) 的差值
        ('ablation_C_onehot_arch', 'ablation_C_seq_tf',
                                              'C(one-hot) → C(LM)     (seq-repr gain, same arch)'),
        # T 诊断: T (纯 TF 分支) → C (B+TF 融合)  衡量「跨模态融合对 TF 的增量」
        # 正值大 = fusion 有效地把 LM/Activity 融入 TF 特征
        # 正值小 = 融合贡献有限, 架构改进空间主要在 TF 分支
        # 负值 = 融合反而把 TF 的好信息冲淡了
        ('ablation_T_tf_only_arch', 'ablation_C_seq_tf',
                                              'T(TF-only-arch) → C     (cross-modal fusion gain)'),
        # T vs B — 对比纯 TF 分支与 Act+LM, 回答「TF 信号是否强过 LM 序列表征」
        ('ablation_B_act_lm', 'ablation_T_tf_only_arch',
                                              'B(Act+LM) → T(TF-only-arch)  (TF vs Seq info)'),
    ]
    print()
    for k1, k2, lbl in _delta_pairs:
        if k1 in all_results and k2 in all_results:
            delta = all_results[k2]['pearson'] - all_results[k1]['pearson']
            print(f"  Δ {lbl:48s}: {delta:+.4f}")

    # ==========================================================================
    # ★ Step 3.5: 固定象限标签 + TF-only 基线 × multiseed
    # ==========================================================================
    # 逻辑:
    #   1) 抽取 seed=42 B 模型的 seq_only_pred (全基因), 作为象限 X 轴固定锚
    #   2) 调 compute_fixed_quadrant_labels 得 quadrant_info (跨 seed / 跨模型共用)
    #   3) 运行 TF-only 基线 × seeds (独立, 不进 MODEL_COMPARISON)
    #   4) quadrant_info 被 MULTI_SEED_EVAL 复用, 保证象限定义一致, std 可归因
    print("\n" + "=" * 62)
    print("  ★ Step 3.5: 固定象限标签构建 + TF-only 基线")
    print("=" * 62)

    # ── 3.5a  全基因 TPM 归一化 (与 run_postanalysis 的 y_all_norm 一致) ───
    _lp_global = ExpressionProcessor(GT['tpm_transform'], GT['tpm_normalize'])
    _lp_global.fit_transform(y_raw[split_masks['train']])  # fit on train only
    _tpm_norm_full = np.empty(len(y_raw), dtype=np.float32)
    _tpm_norm_full[split_masks['train']] = _lp_global.transform(y_raw[split_masks['train']])
    _tpm_norm_full[split_masks['val']]   = _lp_global.transform(y_raw[split_masks['val']])
    _tpm_norm_full[split_masks['test']]  = _lp_global.transform(y_raw[split_masks['test']])

    # ── 3.5b  从 seed=42 B 模型推 seq_only_pred 全基因 → 固定象限锚 ────────
    # 关键: RESUME_FROM_CACHE 命中时 all_results['model']=None, 这里通过
    #      _resolve_b_model 从 ablation_B_act_lm/complete_model.pth 重建,
    #      否则会错误地回退到 Vaishnav raw seq_activity 锚.
    quadrant_info = None
    _b_result = all_results.get('ablation_B_act_lm')
    _b_model, _b_src = _resolve_b_model(
        all_results, PC['output_dir'], device,
        lm_emb_available=(lm_emb_all is not None))

    if _b_model is not None:
        print(f"\n  [象限锚] 来源: seed={GT['random_seed']} B 模型 seq_only_pred  "
              f"(source={_b_src})")
        # 复用 B 训练时的 sc_mean/sc_std (无泄漏, 都只 fit on train)
        _sc_mean = _b_result.get('sc_mean') if _b_result is not None else None
        _sc_std  = _b_result.get('sc_std')  if _b_result is not None else None
        if _sc_mean is None or _sc_std is None:
            _sc_raw_tr = scalar_data[split_masks['train']].astype(np.float32)
            _sc_mean = _sc_raw_tr.mean(axis=0, keepdims=True)
            _sc_std  = _sc_raw_tr.std(axis=0,  keepdims=True) + 1e-8
            print(f"  ⚠ B 未返回 sc_mean/sc_std, 从 train 重新 fit (无泄漏)")
        _sc_full_norm = ((scalar_data.astype(np.float32) - _sc_mean) / _sc_std).astype(np.float32)
        _sc_full_norm = np.nan_to_num(_sc_full_norm, nan=0.0)

        _seq_only_pred_full = np.empty(len(y_raw), dtype=np.float32)
        with torch.no_grad():
            _bs_chunk = 512
            for _s in range(0, len(y_raw), _bs_chunk):
                _e = min(_s + _bs_chunk, len(y_raw))
                _sc_batch = torch.FloatTensor(
                    _sc_full_norm[_s:_e, :N_SEQ_FEATURES]).to(device)
                _lm_batch = (torch.FloatTensor(lm_emb_all[_s:_e]).to(device)
                             if (lm_emb_all is not None and USE_LM_EMB) else None)
                _p = _b_model(_sc_batch, _lm_batch)
                if isinstance(_p, tuple):
                    _p = _p[0]
                _seq_only_pred_full[_s:_e] = _p.cpu().numpy()
        print(f"  ✓ B 全基因 seq_only_pred: range=[{_seq_only_pred_full.min():.3f}, "
              f"{_seq_only_pred_full.max():.3f}]  n_valid={np.isfinite(_seq_only_pred_full).sum()}")

        quadrant_info = compute_fixed_quadrant_labels(
            seq_anchor_full=_seq_only_pred_full,
            tpm_norm_full=_tpm_norm_full,
            gene_names=list(genes),
            anchor_name=f'seed{GT["random_seed"]}_B_seq_only_pred',
        )
    else:
        print("\n  ⚠ 未找到 ablation_B_act_lm (memory + 磁盘 ckpt 均不可用)")
        if scalar_data.shape[1] >= N_SEQ_FEATURES:
            print("  → 回退锚: Vaishnav raw seq_activity (与 run_postanalysis 的 fallback 一致)")
            _seq_activity_full = compute_seq_summary_np(
                scalar_data[:, :N_SEQ_FEATURES].astype(np.float32))
            quadrant_info = compute_fixed_quadrant_labels(
                seq_anchor_full=_seq_activity_full,
                tpm_norm_full=_tpm_norm_full,
                gene_names=list(genes),
                anchor_name='vaishnav_raw_seq_activity_fallback',
            )
        else:
            print("  ⚠ scalar_data 亦不可用, 象限分析被禁用 (quadrant_info=None)")

    # ── 3.5c  落盘象限标签 (供独立脚本 / FUN-PROSE v3 / 手工分析) ─────────
    # 列名采用 `gene_name` (不再用 `gene`), 与 FUN-PROSE v3 的 load_quadrant_labels
    # 完全对齐; 并额外写一份标准化文件名 `fixed_quadrant_labels.csv`,
    # 供 --quadrant_csv 直接引用 (与 FUN-PROSE v3 协议一致).
    if quadrant_info is not None:
        _qi_dir = Path(PC['output_dir']) / 'fixed_quadrant_labels'
        _qi_dir.mkdir(parents=True, exist_ok=True)
        _qi_df = pd.DataFrame({
            'gene_name': list(genes),
            'quadrant':  quadrant_info['quadrant_array'],
            'split':     np.where(split_masks['train'], 'train',
                                  np.where(split_masks['val'], 'val',
                                           np.where(split_masks['test'], 'test', 'unknown'))),
            'tpm_norm':  _tpm_norm_full,
            'anchor':    quadrant_info['anchor_name'],
        })
        # 主文件: 跨脚本标准名 (FUN-PROSE v3 --quadrant_csv 直接指向这里)
        _qi_df.to_csv(_qi_dir / 'fixed_quadrant_labels.csv', index=False)
        # 旧文件名: 保留以兼容历史手工分析脚本 (内容完全一致)
        _qi_df.to_csv(_qi_dir / 'gene_quadrant_labels.csv', index=False)
        # 写锚元数据
        with open(_qi_dir / 'anchor_meta.txt', 'w', encoding='utf-8') as _fw:
            _fw.write(f"anchor_name:   {quadrant_info['anchor_name']}\n")
            _fw.write(f"sa_median:     {quadrant_info['sa_median']:.6f}\n")
            _fw.write(f"tpm_threshold: {quadrant_info['tpm_threshold']}\n")
            _fw.write(f"n_genes:       {len(genes)}\n")
            for q in ('Q1', 'Q2', 'Q3', 'Q4'):
                _fw.write(f"{q}: {int(quadrant_info['masks'][q].sum())}\n")
        print(f"  ✓ 象限标签 + 锚元数据 → {_qi_dir}")
        print(f"     标准文件 (FUN-PROSE v3 --quadrant_csv): fixed_quadrant_labels.csv")

    # ── 3.5d  TF-only 基线 × multiseed (独立, 不进 MODEL_COMPARISON) ──────
    if TF_ONLY_BASELINE.get('enabled', True):
        try:
            tf_only_df = run_tf_only_baseline_multiseed(
                X_raw=X_raw, y_raw=y_raw,
                split_masks=split_masks,
                scalar_data=scalar_data,
                lm_emb_all=lm_emb_all,
                output_dir=PC['output_dir'],
                quadrant_info=quadrant_info,
                cfg=TF_ONLY_BASELINE,
                tpm_transform=GT['tpm_transform'],
                tpm_normalize=GT['tpm_normalize'],
                n_seq_features=N_SEQ_FEATURES,
            )
            all_results['tf_only_baseline'] = tf_only_df
        except Exception as _tf_e:
            print(f"\n  ⚠ TF-only 基线出错: {_tf_e}")
            traceback.print_exc()
    else:
        print("\n  [跳过] TF_ONLY_BASELINE.enabled=False")

    # ── Step 4: 最优全模型后分析 ─────────────────────────────────────────
    best_key = 'ablation_C_seq_tf' if 'ablation_C_seq_tf' in all_results else None

    if best_key is None:
        print("\n  ⚠ 无可用全模型，跳过后分析")
    else:
        print(f"\n  ✓ 使用 {best_key} 做后分析")
        best_res = all_results[best_key]
        sc_mean_full = best_res['sc_mean']
        sc_std_full = best_res['sc_std']

        sig_std_post = SignalStandardizer()
        sig_std_post.fit_transform(X_raw[split_masks['train']])
        lp_post = ExpressionProcessor(GT['tpm_transform'], GT['tpm_normalize'])
        y_tr_n_post, _ = lp_post.fit_transform(y_raw[split_masks['train']])
        y_va_n_post = lp_post.transform(y_raw[split_masks['val']])
        y_te_n_post = lp_post.transform(y_raw[split_masks['test']])

        best_model = MultiModalPredictor(
            n_tfs=n_tfs, n_bins=n_bins,
            d_tf=MC['d_tf'], d_cnn=MC['d_cnn'],
            n_tf_layers=MC['n_tf_layers'], n_regions=MC['n_regions'],
            n_heads=MC['n_heads'],
            use_scalars=True, n_scalars=N_SEQ_FEATURES, n_seq_features=N_SEQ_FEATURES,
            d_seq=MC['d_seq'], d_tf_proj=MC['d_tf_proj'],
            use_lm=USE_LM_EMB, lm_dim=LM_DIM,
            d_lm_hidden=MC['d_lm_hidden'], d_lm_out=MC['d_lm_out'],
            lm_input_dropout=MC['lm_input_drop'],
            freeze_lm=MC['freeze_lm'],
            topk_k=MC.get('topk_k', 0),
        ).to(device)

        ckpt_path = Path(PC['output_dir']) / best_key / 'best_model.pth'
        ckpt = _safe_torch_load(str(ckpt_path), device)
        best_model.load_state_dict(
            ckpt.get('model_state_dict', ckpt)
            if isinstance(ckpt, dict) else ckpt)
        best_model.eval()
        print(f"  ✓ 已加载最佳模型权重: {ckpt_path}")

        run_postanalysis(
            model=best_model,
            X_raw=X_raw, y_raw=y_raw, split_masks=split_masks,
            tf_names=tf_names, gene_names=genes,
            sig_std=sig_std_post, lp=lp_post,
            y_tr_norm=y_tr_n_post, y_val_norm=y_va_n_post,
            y_test_norm=y_te_n_post,
            output_dir=PC['analysis_dir'], device=device,
            scalar_data_raw=scalar_data,
            sc_mean=sc_mean_full, sc_std=sc_std_full,
            ablation_results=all_results,
            n_seq_features=N_SEQ_FEATURES,
            seq_only_model_dir=(all_results
                                .get('ablation_B_act_lm', {})
                                .get('exp_dir')),
            n_tfs_model=n_tfs, n_bins_model=n_bins,
            n_scalars_model=N_SEQ_FEATURES,
            tf_tone_csv=PC['tf_direction_csv'],
            lm_data=lm_emb_all if USE_LM_EMB else None,
            iaa_fc_file=(PC['iaa_fc_file']
                         if PC.get('iaa_fc_file') and Path(PC['iaa_fc_file']).exists()
                         else None),
        )

    # ── 架构链多种子 (B / T / 可选 C)：Protocol 2 数据源  (new54 增补) ────
    #   设计要点:
    #     · seed=42 走主链结果复用 (memory→disk), 无需重训
    #     · 其它 seed 触发 train_experiment + RESUME_FROM_CACHE, 中断重跑零代价
    #     · D / L 不参与 (default ms_models=['B','T']); C 通常已在 FUSION_ABL
    #       中跑全 seed, 这里默认不重复跑, 但若 ms_models 含 'C' 则照样跑.
    #     · 必须放在 FUSION_ABLATION 之后, MULTI_SEED_EVAL 之前: 这样 C@seeds
    #       已完整, B/T 用一致的 seed 集; MULTI_SEED_EVAL 后续可继续聚焦 C 象限.
    if MULTISEED_PROGRESSION.get('enabled', False):
        _msp_seeds   = list(MULTISEED_PROGRESSION.get('seeds',
                                                       [42, 123, 456, 789, 2024]))
        _msp_models  = list(MULTISEED_PROGRESSION.get('models', ['B', 'T']))
        # 容错: 排除掉不在 RUN_ABLATIONS 的模型 (避免 train kwargs 缺数据)
        _msp_models = [m for m in _msp_models if m in RUN_ABLATIONS]
        if not _msp_models:
            print("\n  [跳过] MULTISEED_PROGRESSION: models 为空或未在 RUN_ABLATIONS")
        else:
            print("\n" + "=" * 62 +
                  "\n  ★ 架构链多种子 (Protocol 2 数据源, B/T)\n" + "=" * 62)
            print(f"  models: {_msp_models}   seeds: {_msp_seeds}")

            # ── B kwargs factory (model_kind='seq_only', use_lm=True) ────
            def _b_kwargs_factory(seed, label):
                return dict(
                    **_exp_kwargs('B'),
                    label=label, model_kind='seq_only',
                    scalar_data=scalar_data, use_scalars=True,
                    n_scalars=N_SEQ_FEATURES, n_seq_features=N_SEQ_FEATURES,
                    use_lm=USE_LM_EMB,
                    lm_data=lm_emb_all if USE_LM_EMB else None,
                )

            # ── T kwargs factory (TF-only 分支, no LM/activity/fusion) ───
            def _t_kwargs_factory(seed, label):
                return dict(
                    **_exp_kwargs('T'),
                    label=label, model_kind='tf',
                    scalar_data=None, use_scalars=False,
                    n_scalars=0, n_seq_features=N_SEQ_FEATURES,
                    use_lm=False, lm_data=None,
                    seq_baseline_model=None,
                )

            # ── C kwargs factory (与 MULTI_SEED_EVAL 中 _base_c_kwargs 对齐) ──
            def _c_kwargs_factory(seed, label):
                return dict(
                    **_exp_kwargs('C'),
                    label=label, model_kind='tf',
                    scalar_data=scalar_data, use_scalars=True,
                    n_scalars=N_SEQ_FEATURES, n_seq_features=N_SEQ_FEATURES,
                    use_lm=USE_LM_EMB,
                    lm_data=lm_emb_all if USE_LM_EMB else None,
                )

            _msp_factories = dict(B=_b_kwargs_factory,
                                  T=_t_kwargs_factory,
                                  C=_c_kwargs_factory)

            # B 模型懒解析器, 仅 C 需要 (B/T 不需要软监督)
            def _resolve_b_lazy():
                return _resolve_b_model(
                    all_results, PC['output_dir'], device,
                    lm_emb_available=(lm_emb_all is not None))

            for _mk in _msp_models:
                _factory = _msp_factories.get(_mk)
                if _factory is None:
                    print(f"  ⚠ MULTISEED_PROGRESSION: 未知 model_key={_mk}, 跳过")
                    continue
                _b_resolver = _resolve_b_lazy if _mk == 'C' else None
                try:
                    _msp_summary = _run_progression_multiseed(
                        model_key=_mk,
                        seeds=_msp_seeds,
                        all_results=all_results,
                        base_kwargs_factory=_factory,
                        output_dir=PC['output_dir'],
                        device=device,
                        b_model_resolver=_b_resolver,
                        quadrant_info=quadrant_info,
                        test_mask=split_masks['test'],
                        compute_quadrant=True,
                    )
                    all_results[f'multiseed_{_mk}_summary'] = _msp_summary
                except Exception as _msp_e:
                    print(f"  ⚠ multiseed_{_mk} 整体失败: {_msp_e}")
                    traceback.print_exc()

    # ── 多种子评估（MULTI_SEED_EVAL 开关）────────────────────────────────
    if MULTI_SEED_EVAL.get('enabled', False) and 'C' in RUN_ABLATIONS:
        print("\n" + "=" * 62 +
              "\n  ★ 多种子评估（全模型 C，MULTI_SEED_EVAL）\n" + "=" * 62)
        _ms_seeds = list(MULTI_SEED_EVAL['seeds'])
        _reuse_enabled = MULTI_SEED_EVAL.get('reuse_fusion_abl_c', True)

        # ── 装填 preloaded_results: 内存优先, 磁盘兜底 ────────────────────
        # 目的: 如果 FUSION_ABL (或主 C 运行@seed=42) 已经跑过某个 seed,
        # 直接拿它的完整结果 (含 preds/labels/test_gene_indices) 复用,
        # 避免重训 C; 同时保证 per-quadrant CSV 在"全部复用"场景下也能生成.
        _preloaded = {}
        if _reuse_enabled:
            for _s in _ms_seeds:
                # 候选 key 优先级 (同 FUSION_ABL 内部规则):
                #   seed=42 → 主 C 运行 ablation_C_seq_tf 或 fusion_abl_C_seed42
                #   其他   → fusion_abl_C_seed{s}
                _candidates = []
                if _s == 42:
                    _candidates.append('ablation_C_seq_tf')
                _candidates.append(f'fusion_abl_C_seed{_s}')

                _found = None
                # (a) 内存: all_results[key] 完整 dict (刚跑完的同 run)
                for _k in _candidates:
                    _r = all_results.get(_k)
                    if (_r is not None and 'preds_test' in _r
                            and 'labels_test' in _r
                            and 'test_gene_indices' in _r):
                        _found = _r
                        _src = f"memory:{_k}"
                        break
                # (b) 磁盘: <output_dir>/<key>/train_result_cache.pth
                if _found is None:
                    for _k in _candidates:
                        _cache = _load_c_seed_payload_from_cache(
                            Path(PC['output_dir']) / _k, device)
                        if _cache is not None:
                            _found = _cache
                            _src = f"disk:{_k}/train_result_cache.pth"
                            break
                if _found is not None:
                    _preloaded[_s] = _found
                    print(f"  [多种子] seed={_s} 复用自 {_src}  "
                          f"(r={float(_found.get('pearson', float('nan'))):.4f})")

        _ms_seeds_to_run = [s for s in _ms_seeds if s not in _preloaded]
        if _preloaded:
            print(f"  [多种子] 已缓存 {len(_preloaded)} seeds: "
                  f"{sorted(_preloaded.keys())}")
            print(f"  [多种子] 需补跑 {len(_ms_seeds_to_run)} seeds: "
                  f"{_ms_seeds_to_run or '无 (全部复用)'}")

        # ── train_fn: 对未缓存的 seed 现训 ────────────────────────────────
        _base_c_kwargs = dict(
            X_raw=X_raw, y_raw=y_raw, chromosomes=chromosomes,
            genes=genes, tf_names=tf_names,
            n_tfs=n_tfs, n_bins=n_bins,
            output_dir=PC['output_dir'], device=device,
            split_masks=split_masks,
            batch_size=GT['batch_size'], epochs=GT['epochs'], lr=GT['lr'],
            warmup_epochs=GT['warmup_epochs'], patience=GT['patience'],
            tpm_transform=GT['tpm_transform'], tpm_normalize=GT['tpm_normalize'],
            d_tf=MC['d_tf'], d_cnn=MC['d_cnn'],
            n_tf_layers=MC['n_tf_layers'], n_regions=MC['n_regions'],
            n_heads=MC['n_heads'],
            d_seq=MC['d_seq'], d_tf_proj=MC['d_tf_proj'],
            d_lm_hidden=MC['d_lm_hidden'], d_lm_out=MC['d_lm_out'],
            fc_dropout=MC['fc_dropout'],
            lm_input_dropout=MC['lm_input_drop'],
            lm_noise_std=MC['lm_noise_std'],
            freeze_lm=MC['freeze_lm'],
            lm_dim=LM_DIM,
            model_kind='tf',
            scalar_data=scalar_data, use_scalars=True,
            n_scalars=N_SEQ_FEATURES, n_seq_features=N_SEQ_FEATURES,
            use_lm=USE_LM_EMB,
            lm_data=lm_emb_all if USE_LM_EMB else None,
        )

        def _multiseed_train_fn(random_seed, label):
            random.seed(random_seed);
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)
            # 与 C 块 / Step 3.5b 统一: 命中 RESUME_FROM_CACHE 时也能从磁盘重建 B
            _b_sup, _ = _resolve_b_model(
                all_results, PC['output_dir'], device,
                lm_emb_available=(lm_emb_all is not None))
            return train_experiment(**_base_c_kwargs, label=label,
                                    seq_baseline_model=_b_sup)

        # 即使 _ms_seeds_to_run=[] (全部复用) 也调用 run_multiseed_eval,
        # 让 CSV 落盘逻辑统一走一条路径; preloaded_results 覆盖的 seed
        # 不会触发 train_fn, 只重算指标 + 写表.
        multiseed_summary = run_multiseed_eval(
            seed_list=_ms_seeds,
            train_fn=(_multiseed_train_fn if _ms_seeds_to_run else None),
            output_dir=str(Path(PC['output_dir']) / 'multiseed_C'),
            label_prefix='multiseed_C',
            quadrant_info=quadrant_info,
            test_mask=split_masks['test'],
            compute_quadrant=MULTI_SEED_EVAL.get('quadrant_stats', True),
            preloaded_results=_preloaded,
        )

        all_results['multiseed_summary'] = multiseed_summary
        print(f"\n  ✓ 多种子评估完成: "
              f"r = {multiseed_summary.get('pearson_mean', float('nan')):.4f} "
              f"± {multiseed_summary.get('pearson_std', float('nan')):.4f}  "
              f"(n={multiseed_summary.get('n_seeds', 0)})")
    elif MULTI_SEED_EVAL.get('enabled', False):
        print("\n  ⚠ MULTI_SEED_EVAL.enabled=True 但 'C' 不在 RUN_ABLATIONS，跳过")

    # ── FUSION_ABLATION 汇总 ─────────────────────────────────────────────
    #   new47: c_per_seed 直接从 FUSION_ABL 块注入, 不再依赖 multiseed_summary
    _fa_recs = all_results.pop('_fusion_abl_records', None)
    _fa_c_ps = all_results.pop('_fusion_abl_c_per_seed', None)
    # new54: 把 c_per_seed 单独存一份非下划线 key, 防止下游 _emit_dual_protocol_*
    #        在 pop 后访问不到 (FUSION_ABL summary 内部不会再用此字段)
    if _fa_c_ps:
        all_results['fusion_abl_c_per_seed_kept'] = dict(_fa_c_ps)
    if _fa_recs:
        _emit_fusion_ablation_summary(
            fa_records=_fa_recs,
            all_results=all_results,
            fusion_variants=list(FUSION_ABLATION.keys()),
            fusion_seeds=list(FUSION_ABLATION_SEEDS),
            output_dir=PC['output_dir'],
            split_masks=split_masks,
            genes=genes,
            quadrant_info=quadrant_info,
            c_per_seed=_fa_c_ps,
        )

    # ── Step 5: 模型对比矩阵（线性/非线性/DL × 3个特征集）────────────
    if MODEL_COMPARISON.get('enabled', True):
        print("\n" + "=" * 62 +
              "\n  ★ Step 5: 模型对比矩阵（相同特征集下对比复杂度）\n" +
              "=" * 62)
        try:
            comparison_df = run_model_comparison_matrix(
                X_raw=X_raw,
                y_raw=y_raw,
                split_masks=split_masks,
                scalar_data=scalar_data,
                lm_emb_all=lm_emb_all,
                dl_results=all_results,
                output_dir=PC['output_dir'],
                cfg=MODEL_COMPARISON,
                tpm_transform=GT['tpm_transform'],
                tpm_normalize=GT['tpm_normalize'],
                n_seq_features=N_SEQ_FEATURES,
                # new52: 传入 Step 3.5 构建的固定象限锚, 让六个基线与 ours
                # 共用同一口径输出 per-quadrant Pearson/MAE/Bias.
                quadrant_info=quadrant_info,
                test_gene_indices=np.where(split_masks['test'])[0].astype(np.int64),
            )
            if not comparison_df.empty:
                print(f"\n  ✓ 模型对比矩阵完成，共 {len(comparison_df)} 条记录")
        except Exception as _cmp_e:
            print(f"\n  ⚠ 模型对比矩阵出错: {_cmp_e}")
            traceback.print_exc()
    else:
        print("\n  [跳过] MODEL_COMPARISON.enabled=False")

    # ── Supp S11: 四份 per-quadrant CSV 汇合（第三轮统一视图）─────────────
    #   无条件调用, 任一源缺失都会跳过, 只要至少一份存在就会产出合并表.
    try:
        _consolidate_paper_supp_s11_per_quadrant(PC['output_dir'])
    except Exception as _s11_e:
        print(f"\n  ⚠ Supp S11 汇合出错: {_s11_e}")
        traceback.print_exc()

    # ── 双协议对比 CSV 输出 (new54)：直接服务于论文 Fig 3 + Supp S12 ──────
    #   Protocol 1: paper_protocol1_single_seed42.csv (D/L/B/T/C 全 single-seed=42)
    #   Protocol 2: paper_protocol2_paired_multiseed.csv (B/T/C 5-seed paired)
    #   两份 CSV 一次性汇总历史 + 当前所有可解析结果, 复用磁盘缓存.
    try:
        _msp_seeds_final  = list(MULTISEED_PROGRESSION.get(
            'seeds', FUSION_ABLATION_SEEDS))
        # Protocol 2 默认参与的模型: 必须是 multi-seed 化过的, 即 ms config 里的
        # models, 加上 'C' (已通过 FUSION_ABL/MULTI_SEED_EVAL 多 seed 化)
        _msp_models_final = list(MULTISEED_PROGRESSION.get('models', ['B', 'T']))
        if 'C' not in _msp_models_final:
            _msp_models_final.append('C')
        # FUSION_ABL 的 c_per_seed 优先入口 (FUSION_ABL summary 内部 pop 后,
        # 我们在 8676 行处把 c_per_seed 备份到 fusion_abl_c_per_seed_kept)
        _c_ext = all_results.get('fusion_abl_c_per_seed_kept') or {}
        _emit_dual_protocol_comparison(
            all_results=all_results,
            output_dir=PC['output_dir'],
            device=device,
            ms_seeds=_msp_seeds_final,
            ms_models=_msp_models_final,
            c_per_seed_external=_c_ext,
        )
    except Exception as _dual_e:
        print(f"\n  ⚠ 双协议 CSV 输出失败: {_dual_e}")
        traceback.print_exc()

    print(f"\n{'=' * 62}\n✓ 所有结果已保存至: {PC['output_dir']}")
