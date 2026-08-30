"""
GO Enrichment Analysis for cross-attention weights (c values).
================================================================

Narrative framing: the c-axis captures the yeast growth-vs-differentiation
axis (high-c = growth/biosynthesis; low-c = sporulation/meiosis).

Runs two complementary enrichment methods and four biological validations:

  [Enrichment — main figures]
    1. Fisher's Exact at primary cutoff (DEFAULT_PCT=10%)  (Section 3-4)
    2. Mann-Whitney U on the full c distribution            (Section 5-6)
       — threshold-free, produces full-GO and GO-Slim lollipop figures

  [Biological validation — main figures]
    3. Canonical growth-vs-sporulation gene table           (Section 7)
       10 growth-program genes (ribosome/glycolysis) + 10 sporulation/meiosis
       genes; c-values, percentile ranks, iaa_mean_sens.
    4. Brauer 2008 / Gasch 2000 external signature comparison (Section 8)
       MW test of published growth-rate and ESR gene sets vs. c distribution.
       Expected: Brauer-up / Gasch-repressed → high-c;
                 Brauer-down / Gasch-induced  → low-c.

  [Supplementary (outputs to ./result/supp/)]
    5. TF-encoding gene × IAA sensitivity                  (Section 9 supp)
    6. Eisosome deep-dive                                   (Section 10 supp)
    7. Bootstrap stability of top terms                     (Section 11)

Required input files (place in working directory):
    sgd.gaf                              SGD GAF annotation (gene -> GO)
    go-basic.obo                         Gene Ontology DAG
    go_slim_mapping.tab                  SGD GO Slim mapping — download from
                                           https://downloads.yeastgenome.org/curation/literature/go_slim_mapping.tab
    dirC_crossweight_iaa_summary.csv     Per-gene c value (and optionally
                                           iaa_mean_sens); one row per gene.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from datetime import datetime

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu, spearmanr

from goatools.go_enrichment import GOEnrichmentStudy
from goatools.obo_parser import GODag

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
PATH_CONFIG = dict(
    sgd_gaf        = "sgd.gaf",
    go_obo         = "go-basic.obo",
    go_slim_tab    = "go_slim_mapping.tab",
    # Main input: per-gene cross_weight (+ optional iaa_mean_sens)
    c_summary_csv  = "dirC_crossweight_iaa_summary_178.csv",
    # Used as background gene set (same file as c_summary_csv by default)
    background_csv = "dirC_crossweight_iaa_summary_178.csv",
    # Fallback 5%-cut files if c_summary_csv is missing
    high_c_csv     = "dirC_gene_list_high_c_top5pct.csv",
    low_c_csv      = "dirC_gene_list_low_c_top5pct.csv",
    output_dir     = "./result_178",
)

# Statistics
THRESHOLDS      = [5, 10, 15, 20, 25]   # cutoff sweep (%)
DEFAULT_PCT     = 10                    # primary figure cutoff (10% keeps signal focused)
FDR_CUTOFF      = 0.05
RAWP_CUTOFF     = 0.05
MIN_GENES_TERM  = 20                    # minimum pop_count for Fisher
MW_MIN_GENES    = 15                    # min annotated genes for MW (full GO; 15 reduces noise vs 10)
MW_MAX_GENES    = 300                   # max annotated genes for MW (300 removes trivially broad terms)
SLIM_MIN_GENES  = 5                     # MW min for GO Slim (smaller space)
SLIM_MAX_GENES  = 6000                  # MW max for GO Slim

# Group labels (neutral, data-driven — no preset biology)
G_HIGH = "High cross-weight c"
G_LOW  = "Low cross-weight c"

# Namespace rendering
NS_ORDER = ["biological_process", "molecular_function", "cellular_component"]
NS_SHORT = {"biological_process": "BP", "molecular_function": "MF", "cellular_component": "CC"}
NS_COLOR = {"biological_process": "#27ae60", "molecular_function": "#8e44ad", "cellular_component": "#e67e22"}
NS_LIGHT = {"biological_process": "#eafaf1", "molecular_function": "#f5eef8", "cellular_component": "#fef9e7"}
GROUP_COLOR = {G_HIGH: "#c0392b", G_LOW: "#1a6fa8"}

# GO root terms to drop (too generic to be informative)
ROOT_TERMS = {
    "GO:0008150", "GO:0003674", "GO:0005575",
    "GO:0005622", "GO:0044464", "GO:0044424",
    "GO:0110165", "GO:0009987", "GO:0044699", "GO:0065007",
}

# GO terms annotating TF-encoding genes (Section 10)
TF_GO_TERMS = {
    "GO:0000981",  # DNA-binding TF activity, RNA pol II-specific
    "GO:0003700",  # DNA-binding TF activity
    "GO:0001228",  # DNA-binding TF activator, RNA pol II-specific
    "GO:0001227",  # DNA-binding TF repressor, RNA pol II-specific
    "GO:0140110",  # transcription regulator activity
}

PC = PATH_CONFIG
os.makedirs(PC["output_dir"], exist_ok=True)
SUPP_DIR = os.path.join(PC["output_dir"], "supp")
os.makedirs(SUPP_DIR, exist_ok=True)
STABILITY_LOG = os.path.join(PC["output_dir"], "c_stability_log.jsonl")

# ── Canonical gene sets for Section 7 (gene-level signature table) ─────
# Standard names; resolved to ORF IDs at runtime via GAF name_to_orf map.
# Sources: SGD curated annotations; Brauer 2008; Gasch 2000.
CANONICAL_GROWTH_GENES = [
    # Ribosomal proteins (Ribi regulon)
    "RPS3", "RPL28", "RPL3", "RPL5", "RPL6A", "RPL7A", "RPS2", "RPS5",
    # Translation elongation
    "TEF1", "TEF2",
    # Glycolysis / central carbon
    "CDC19", "ENO2", "TDH3", "TDH2", "GPM1", "PGK1", "FBA1", "TPI1", "ADH1",
    # Amino-acid biosynthesis (GCN4 targets)
    "ILV5", "LEU4", "HIS4", "ARG4",
]

CANONICAL_SPORULATION_GENES = [
    # IME/NDT80 meiosis master regulators
    "IME1", "IME2", "NDT80",
    # Synaptonemal complex
    "ZIP1", "ZIP2", "ZIP3", "HOP1", "RED1",
    # Meiotic recombination
    "SPO11", "REC8", "DMC1", "HOP2", "MND1", "MSH4",
    # Spore-wall assembly
    "SPO20", "SPS4", "DIT1", "DIT2",
    # Sporulation signaling
    "SUM1", "SMK1",
]

# ── Brauer 2008 / Gasch 2000 gene sets for Section 8 ───────────────────
# Brauer et al. 2008 Mol Biol Cell — chemostat growth-rate signature.
# "UP with growth rate" = same as high-c expected.
BRAUER_GROWTH_UP = [
    # Ribosomal proteins (top growth-rate correlated in Brauer 2008 Fig 3)
    "RPL3","RPL4A","RPL5","RPL6A","RPL7A","RPL8A","RPL9A","RPL10",
    "RPL11A","RPL13A","RPL14A","RPL15A","RPL16A","RPL17A","RPL18A",
    "RPL19A","RPL20A","RPL22A","RPL23A","RPL24A","RPL25","RPL27A",
    "RPL28","RPL29","RPL30","RPL31A","RPL32","RPL33A","RPL34A",
    "RPS2","RPS3","RPS4A","RPS5","RPS6A","RPS7A","RPS8A","RPS9A",
    "RPS10A","RPS11A","RPS13","RPS14A","RPS15","RPS16A","RPS17A",
    "RPS18A","RPS19A","RPS20","RPS21A","RPS22A","RPS23A","RPS24A",
    "RPS25A","RPS26A","RPS28A","RPS29A","RPS30A",
    # Translation factors
    "TEF1","TEF2","EFB1",
    # Ribi biogenesis
    "SFP1","FHL1","IFH1",
]

# "DOWN with growth rate" = stress/differentiation = low-c expected.
BRAUER_GROWTH_DOWN = [
    # Environmental stress response (ESR) — induced under slow growth / starvation
    "HSP12","HSP26","HSP42","HSP78","HSP82","HSP104",
    "SSA1","SSA2","SSA3","SSA4","SSB1","SSB2",
    "TPS1","TPS2","TSL1",          # trehalose synthesis
    "CTT1","CTA1",                  # catalases
    "SOD1","SOD2",
    "YHB1",                         # nitric oxide response
    "GRE1","GRE2","GRE3",
    "DDR2","SPI1",
    # Nitrogen starvation / autophagy
    "ATG1","ATG7","ATG8","ATG13","ATG14",
    "GLN1","GDH2",
]

# Gasch et al. 2000 Mol Biol Cell — ESR.
# ESR-induced (stress → induced; growth → repressed; low-c expected).
GASCH_ESR_INDUCED = [
    "HSP12","HSP26","HSP42","HSP78","HSP82","HSP104",
    "SSA3","SSA4","STI1","SIS1",
    "TPS1","TPS2","TSL1","NTH1",
    "CTT1","CTA1","SOD1","SOD2",
    "GRE1","GRE2","GRE3","YHB1","DDR2","SPI1",
    "RPN4","UFD1","CDC48",          # proteasome/ubiquitin
    "HAC1","KAR2","PDI1",           # UPR
]

# ESR-repressed (stress → repressed; growth → high; high-c expected).
GASCH_ESR_REPRESSED = [
    # Ribosome biogenesis
    "RRB1","NOP1","NOP2","NOP7","RRP5","BMS1","UTP4","UTP5","UTP8",
    "NSA1","NOC3","ERB1","YTM1","MAK5","PWP2","RIX7","REI1",
    # Ribosomal proteins
    "RPL3","RPL5","RPL7A","RPL28","RPS3","RPS5","RPS13",
    # Glycolysis
    "CDC19","ENO2","TDH3","PGK1","FBA1","TPI1","ADH1",
    # Amino-acid biosynthesis
    "ILV5","LEU4","HIS4",
]


# ══════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════
_ORF_RE = re.compile(r"^Y[A-P][LR]\d{3}[WC]$")


def is_std(gene: str) -> bool:
    """Standard S. cerevisiae ORF ID check (e.g. YAL001C)."""
    return bool(_ORF_RE.match(str(gene).strip()))


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR. Monotonic, no statsmodels dep."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    fdr_sorted = np.minimum(pvals[order] * n / np.arange(1, n + 1), 1.0)
    fdr_sorted = np.minimum.accumulate(fdr_sorted[::-1])[::-1]
    out = np.empty(n)
    out[order] = fdr_sorted
    return out


def log_c_stability(c_values: np.ndarray, run_id: str | None = None) -> dict:
    """Append per-run c-distribution stats to c_stability_log.jsonl."""
    stats = {
        "timestamp":      datetime.now().isoformat(),
        "run_id":         run_id or datetime.now().strftime("%Y%m%d_%H%M"),
        "n":              int(len(c_values)),
        "mean":           float(np.mean(c_values)),
        "std":            float(np.std(c_values)),
        "p5":             float(np.percentile(c_values, 5)),
        "p25":            float(np.percentile(c_values, 25)),
        "p75":            float(np.percentile(c_values, 75)),
        "p95":            float(np.percentile(c_values, 95)),
        "frac_above_0.8": float((c_values > 0.8).mean()),
        "frac_below_0.2": float((c_values < 0.2).mean()),
    }
    with open(STABILITY_LOG, "a") as fout:
        fout.write(json.dumps(stats) + "\n")
    return stats


def save_fig(fig, out_prefix: str, close: bool = True) -> None:
    """Save a figure as both PDF (300 dpi) and PNG (200 dpi)."""
    for ext, dpi in [(".pdf", 300), (".png", 200)]:
        path = os.path.join(PC["output_dir"], out_prefix + ext)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  ✓ {path}")
    if close:
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
#  1. LOAD DATA  (single pass — no re-reads downstream)
# ══════════════════════════════════════════════════════════════════════
print("\n── 1. Load data ────────────────────────────────────────")
USE_SUMMARY = os.path.exists(PC["c_summary_csv"])

if USE_SUMMARY:
    print(f"  [primary] reading {PC['c_summary_csv']}")
    summary_df = pd.read_csv(PC["c_summary_csv"])
    summary_df = summary_df[summary_df["gene"].apply(is_std)].copy()
    summary_df = (summary_df
                  .drop_duplicates("gene")
                  .sort_values("cross_weight", ascending=False)
                  .reset_index(drop=True))
    ALL_GENES_RANKED = summary_df["gene"].tolist()
    CW_VALUES        = summary_df["cross_weight"].values
    N_TOTAL          = len(ALL_GENES_RANKED)
    HAS_IAA          = "iaa_mean_sens" in summary_df.columns
    print(f"  ✓ genes (ranked): {N_TOTAL}  "
          f"c∈[{CW_VALUES.min():.4f}, {CW_VALUES.max():.4f}]  "
          f"mean={CW_VALUES.mean():.4f}")
    print(f"  ✓ iaa_mean_sens column: {'yes' if HAS_IAA else 'no'}")
    _stab = log_c_stability(CW_VALUES)
    print(f"  [stability log] → {STABILITY_LOG}  "
          f"(p25={_stab['p25']:.4f}  p75={_stab['p75']:.4f})")
else:
    print(f"  [fallback] c_summary_csv missing — loading fixed 5% lists")
    def _load_fixed(path):
        df = pd.read_csv(path)
        return [g for g in df["gene"].tolist() if is_std(g)]
    _high5 = _load_fixed(PC["high_c_csv"])
    _low5  = _load_fixed(PC["low_c_csv"])
    ALL_GENES_RANKED, CW_VALUES = [], np.array([])
    N_TOTAL = len(_high5) + len(_low5)
    HAS_IAA = False
    summary_df = None
    print(f"  ✓ high={len(_high5)}  low={len(_low5)}")

# Background gene set
if PC["background_csv"] == PC["c_summary_csv"] and summary_df is not None:
    background_genes = summary_df["gene"].tolist()
else:
    _bg_df = pd.read_csv(PC["background_csv"])
    background_genes = [g for g in _bg_df["gene"].tolist() if is_std(g)]
print(f"  ✓ background genes: {len(background_genes)}")


# ══════════════════════════════════════════════════════════════════════
#  2. PARSE GAF + INITIALIZE goatools
# ══════════════════════════════════════════════════════════════════════
print("\n── 2. Parse GAF + init goatools ────────────────────────")
orf_to_goterms: dict[str, set] = {}
name_to_orf: dict[str, str] = {}       # standard gene name -> ORF ID
with open(PC["sgd_gaf"]) as f:
    for line in f:
        if line.startswith("!"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 11:
            continue
        go_id = parts[4]
        std_name = parts[2].strip()    # column 3: standard gene name
        for syn in parts[10].split("|"):
            syn = syn.strip()
            if is_std(syn):
                orf_to_goterms.setdefault(syn, set()).add(go_id)
                if std_name:
                    name_to_orf[std_name.upper()] = syn
print(f"  ✓ GAF parsed  annotated genes: {len(orf_to_goterms)}")
print(f"  ✓ standard name → ORF mappings: {len(name_to_orf)}")
print(f"  ✓ background × annotated: {len(set(background_genes) & set(orf_to_goterms))} / {len(background_genes)}")

godag   = GODag(PC["go_obo"])
goeaobj = GOEnrichmentStudy(
    background_genes, orf_to_goterms, godag,
    propagate_counts=True, alpha=FDR_CUTOFF, methods=["fdr_bh"],
)


# ══════════════════════════════════════════════════════════════════════
#  Build propagated GO -> genes inverted index (used in Sections 4, 7-11)
# ══════════════════════════════════════════════════════════════════════
def build_propagated_go2genes(
    gene_go_map: dict, godag: GODag, root_terms: set
) -> dict[str, set]:
    """Propagate direct GAF annotations to all ancestors; drop root terms."""
    go2genes: dict[str, set] = {}
    for gene, direct_terms in gene_go_map.items():
        propagated = set(direct_terms)
        for t in list(direct_terms):
            node = godag.get(t)
            if node:
                propagated.update(node.get_all_parents())
        for go_id in propagated:
            if go_id not in root_terms:
                go2genes.setdefault(go_id, set()).add(gene)
    return go2genes


bg_set = set(background_genes)
gene_go_bg  = {g: v for g, v in orf_to_goterms.items() if g in bg_set}
go2genes_bg = build_propagated_go2genes(gene_go_bg, godag, ROOT_TERMS)
print(f"  ✓ propagated GO terms (bg): {len(go2genes_bg)}")

# c_series: gene -> c value, de-duplicated, restricted to background
if USE_SUMMARY:
    c_series = pd.Series(CW_VALUES, index=ALL_GENES_RANKED)
    c_series = c_series[~c_series.index.duplicated(keep="first")]
    c_series = c_series[c_series.index.isin(bg_set)]
else:
    c_series = pd.Series(dtype=float)

# iaa_series: gene -> iaa_mean_sens  (None if column missing)
iaa_series: pd.Series | None = None
if HAS_IAA:
    iaa_series = (pd.Series(summary_df["iaa_mean_sens"].values,
                            index=summary_df["gene"].tolist())
                  .dropna())
    iaa_series = iaa_series[~iaa_series.index.duplicated(keep="first")]


# ══════════════════════════════════════════════════════════════════════
#  3. FISHER ENRICHMENT (fixed percentile cutoff)
# ══════════════════════════════════════════════════════════════════════
def run_fisher(gene_list: list[str]) -> pd.DataFrame:
    """Run goatools enrichment and return full table (both directions)."""
    if not gene_list:
        return pd.DataFrame()
    results = goeaobj.run_study(gene_list)
    rows = []
    for r in results:
        if r.study_count == 0 or r.pop_count == 0:
            continue
        if r.GO in ROOT_TERMS:
            continue
        if r.pop_count < MIN_GENES_TERM:
            continue
        fe = (r.study_count / r.study_n) / (r.pop_count / r.pop_n)
        node = godag.get(r.GO)
        ns   = node.namespace if node else "unknown"
        rows.append(dict(
            GO_id         = r.GO,
            term          = r.name,
            namespace     = ns,
            p_raw         = r.p_uncorrected,
            p_fdr         = r.p_fdr_bh,
            neg_log10_fdr = -np.log10(max(r.p_fdr_bh,      1e-300)),
            neg_log10_raw = -np.log10(max(r.p_uncorrected, 1e-300)),
            study_count   = r.study_count,
            study_n       = r.study_n,
            pop_count     = r.pop_count,
            pop_n         = r.pop_n,
            gene_ratio    = r.study_count / r.study_n,
            fold_enrich   = fe,
            direction     = "enriched" if fe > 1 else "depleted",
        ))
    return pd.DataFrame(rows)


# ─── 3a. Multi-threshold scan (diagnostic) ─────────────────────────────
print("\n── 3. Fisher enrichment (multi-threshold scan) ─────────")
print(f"  cutoffs: {THRESHOLDS}%")
scan_rows = []
all_res: dict[tuple[int, str], pd.DataFrame] = {}

for pct in THRESHOLDS:
    if USE_SUMMARY:
        k          = max(int(N_TOTAL * pct / 100), 15)
        high_genes = ALL_GENES_RANKED[:k]
        low_genes  = ALL_GENES_RANKED[-k:][::-1]
    else:
        high_genes, low_genes = _high5, _low5
    for grp_key, gene_list in [(G_HIGH, high_genes), (G_LOW, low_genes)]:
        df = run_fisher(gene_list)
        all_res[(pct, grp_key)] = df
        n_fdr_total = int((df["p_fdr"] < FDR_CUTOFF).sum())  if not df.empty else 0
        n_fdr_dep   = int(((df["p_fdr"] < FDR_CUTOFF) & (df["fold_enrich"] < 1)).sum()) if not df.empty else 0
        n_fdr_enr   = n_fdr_total - n_fdr_dep
        n_raw       = int((df["p_raw"] < RAWP_CUTOFF).sum()) if not df.empty else 0
        min_fdr     = float(df["p_fdr"].min()) if not df.empty else 1.0
        scan_rows.append(dict(
            pct=pct, group=grp_key, n_genes=len(gene_list),
            n_fdr_enriched=n_fdr_enr, n_fdr_depleted=n_fdr_dep,
            n_fdr_total=n_fdr_total, n_raw_total=n_raw,
            min_p_fdr=min_fdr,
        ))
        print(f"  [{pct:2d}%] {grp_key:24s}  n={len(gene_list):4d}  "
              f"FDR_enr={n_fdr_enr}  FDR_dep={n_fdr_dep}  "
              f"raw={n_raw}  minFDR={min_fdr:.2e}")

scan_df = pd.DataFrame(scan_rows)
scan_csv = os.path.join(PC["output_dir"], "threshold_scan_summary.csv")
scan_df.to_csv(scan_csv, index=False)
print(f"  ✓ scan table → {scan_csv}")


# ─── 3b. Pick primary cutoff ───────────────────────────────────────────
def pick_primary_pct(default: int = DEFAULT_PCT) -> tuple[int, str]:
    """
    Prefer DEFAULT_PCT if it has any FDR-significant terms.
    Otherwise fall back to the cutoff with the most FDR-sig terms (≥10%),
    then raw-p (≥10%), then smallest min_p_fdr.
    """
    def n_sig(pct, col):
        return sum(int((all_res[(pct, g)][col] < FDR_CUTOFF).sum())
                   for g in [G_HIGH, G_LOW]
                   if not all_res[(pct, g)].empty)

    if default in THRESHOLDS and n_sig(default, "p_fdr") > 0:
        return default, "fdr"
    # Best FDR cutoff (≥10%)
    best = max(((pct, n_sig(pct, "p_fdr")) for pct in THRESHOLDS if pct >= 10),
               key=lambda x: x[1], default=(None, 0))
    if best[1] > 0:
        return best[0], "fdr"
    # Any cutoff with FDR
    best = max(((pct, n_sig(pct, "p_fdr")) for pct in THRESHOLDS),
               key=lambda x: x[1], default=(None, 0))
    if best[1] > 0:
        return best[0], "fdr"
    # Raw p (≥10%)
    best = max(((pct, n_sig(pct, "p_raw")) for pct in THRESHOLDS if pct >= 10),
               key=lambda x: x[1], default=(None, 0))
    if best[1] > 0:
        return best[0], "raw"
    # Any raw
    best = max(((pct, n_sig(pct, "p_raw")) for pct in THRESHOLDS),
               key=lambda x: x[1], default=(None, 0))
    if best[1] > 0:
        return best[0], "raw"
    # Nothing significant — pick smallest min_p_fdr
    row = scan_df.nsmallest(1, "min_p_fdr").iloc[0]
    return int(row["pct"]), "none"


BEST_PCT, BEST_MODE = pick_primary_pct()
print(f"  ✓ primary cutoff: {BEST_PCT}%  mode: {BEST_MODE}")

if USE_SUMMARY:
    k_best    = max(int(N_TOTAL * BEST_PCT / 100), 15)
    high_best = ALL_GENES_RANKED[:k_best]
    low_best  = ALL_GENES_RANKED[-k_best:][::-1]
else:
    high_best, low_best = _high5, _low5


# ─── 3c. Filter, save, select display terms ────────────────────────────
def apply_filter(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if df.empty:
        return df
    if mode == "fdr":
        return df[df["p_fdr"] < FDR_CUTOFF].copy()
    if mode == "raw":
        return df[df["p_raw"] < RAWP_CUTOFF].copy()
    return df.nsmallest(20, "p_raw").copy()


df_high = apply_filter(all_res[(BEST_PCT, G_HIGH)], BEST_MODE)
df_low  = apply_filter(all_res[(BEST_PCT, G_LOW)],  BEST_MODE)
df_high["group"] = G_HIGH
df_low["group"]  = G_LOW

results_df = pd.concat([df_high, df_low], ignore_index=True)
out_csv = os.path.join(PC["output_dir"], "GO_enrichment_crossweight_c.csv")
results_df.to_csv(out_csv, index=False)
print(f"  ✓ Fisher main CSV → {out_csv}")


def select_display_terms(gdf: pd.DataFrame, per_ns: int = 5) -> pd.DataFrame:
    """Per (direction × namespace) take top per_ns by p-value."""
    if gdf.empty:
        return pd.DataFrame()
    key = "p_fdr" if BEST_MODE == "fdr" else "p_raw"
    tops = []
    for direction in ["enriched", "depleted"]:
        sub_d = gdf[gdf["direction"] == direction]
        for ns in NS_ORDER:
            sub = sub_d[sub_d["namespace"] == ns].nsmallest(per_ns, key)
            if not sub.empty:
                tops.append(sub)
    if not tops:
        return pd.DataFrame()
    out = pd.concat(tops).drop_duplicates("GO_id").copy()
    out["ns_rank"]  = out["namespace"].map({n: i for i, n in enumerate(NS_ORDER)})
    out["dir_rank"] = out["direction"].map({"enriched": 0, "depleted": 1})
    return (out.sort_values(["dir_rank", "ns_rank", "fold_enrich"],
                             ascending=[True, True, True])
               .reset_index(drop=True))


plot_high = select_display_terms(df_high)
plot_low  = select_display_terms(df_low)


# ══════════════════════════════════════════════════════════════════════
#  4. FISHER PLOTS  (threshold scan heatmap + 20% lollipop)
# ══════════════════════════════════════════════════════════════════════

def plot_threshold_heatmap(scan_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(
        "Threshold sensitivity: n(significant GO terms) per cutoff\n"
        "(shows at what scale enrichment signal exists)",
        fontsize=10.5, fontweight="bold",
    )
    panels = [
        (axes[0], "n_fdr_total", "FDR < 0.05 (BH-corrected, both directions)", "YlOrRd"),
        (axes[1], "n_raw_total", "raw p < 0.05 (exploratory, uncorrected)",    "YlGnBu"),
    ]
    for ax, metric, title, cmap_name in panels:
        pivot = scan_df.pivot(index="pct", columns="group", values=metric).fillna(0)
        vmax_val = max(pivot.values.max(), 1)
        im = ax.imshow(pivot.values.T, aspect="auto",
                       cmap=plt.cm.get_cmap(cmap_name), vmin=0, vmax=vmax_val)
        ax.set_xticks(range(len(pivot.index)))
        ax.set_xticklabels([f"{p}%" for p in pivot.index], fontsize=9)
        ax.set_yticks(range(len(pivot.columns)))
        ax.set_yticklabels(pivot.columns, fontsize=8)
        ax.set_xlabel("Percentile cutoff", fontsize=9)
        ax.set_title(title, fontsize=9, fontweight="bold")
        for i in range(len(pivot.columns)):
            for j in range(len(pivot.index)):
                v = int(pivot.values[j, i])
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if v > vmax_val * 0.6 else "#333333")
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    save_fig(fig, "GO_threshold_scan")


def _make_sizes(ratios, smin=40, smax=280):
    lo, hi = ratios.min(), ratios.max()
    if hi == lo:
        return np.full(len(ratios), (smin + smax) / 2)
    return smin + (ratios - lo) / (hi - lo) * (smax - smin)


def plot_fisher_lollipop(
    plot_high: pd.DataFrame, plot_low: pd.DataFrame,
    best_pct: int, best_mode: str,
    high_best: list, low_best: list,
) -> None:
    """Two-panel lollipop of fold-enrichment at the chosen % cutoff."""
    _pcol = "neg_log10_fdr" if best_mode == "fdr" else "neg_log10_raw"
    watermark = best_mode in ("raw", "none")
    _pvals = [d[_pcol].max() for d in [plot_high, plot_low]
              if not d.empty and _pcol in d.columns]
    vmax = min(max(_pvals) if _pvals else 10, 30)
    cmap = plt.cm.plasma
    n_rows = max(len(plot_high), len(plot_low), 4)

    fig, (ax_high, ax_low) = plt.subplots(
        1, 2, figsize=(21, max(9, n_rows * 0.45 + 2)))
    fig.patch.set_facecolor("white")

    def _draw(ax, pdf, group_key, n_gene):
        if pdf.empty:
            msg = ("No significant terms (FDR<0.05)"
                   if best_mode == "fdr" else "No significant enrichment")
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="#888")
            ax.set_title(f"{group_key}\n(no terms)",
                         fontsize=10.5, fontweight="bold",
                         color=GROUP_COLOR[group_key], pad=8)
            ax.axis("off")
            return

        n     = len(pdf)
        y     = np.arange(n)
        sizes = _make_sizes(pdf["gene_ratio"].values)

        # Namespace background bands
        prev_ns = None
        for idx, (_, row) in enumerate(pdf.iterrows()):
            ns = row["namespace"]
            if ns != prev_ns and prev_ns is not None:
                ax.axhline(idx - 0.5, color="#cccccc", lw=0.8, zorder=2)
            prev_ns = ns
            ax.axhspan(idx - 0.5, idx + 0.5,
                       color=NS_LIGHT[ns], alpha=0.45, zorder=1)

        # enriched/depleted divider
        dir_vals = pdf["direction"].values
        for idx in range(1, len(dir_vals)):
            if dir_vals[idx] != dir_vals[idx - 1]:
                ax.axhline(idx - 0.5, color="#777", lw=1.5, ls="--", zorder=3)

        # Stems
        for idx, (_, row) in enumerate(pdf.iterrows()):
            col = "#bbb" if row["direction"] == "enriched" else "#ddaaaa"
            ax.plot([1, row["fold_enrich"]], [idx, idx],
                    color=col, lw=1.2, zorder=3, solid_capstyle="round")

        # Scatter — enriched solid, depleted hollow
        enr = pdf["direction"].values == "enriched"
        dep = ~enr
        cv  = pdf[_pcol].values
        if enr.any():
            ax.scatter(pdf.loc[enr, "fold_enrich"], y[enr],
                       s=sizes[enr], c=cv[enr], cmap=cmap, vmin=0, vmax=vmax,
                       edgecolors="#555", linewidths=0.5, alpha=0.95, zorder=5)
        if dep.any():
            ax.scatter(pdf.loc[dep, "fold_enrich"], y[dep],
                       s=sizes[dep], c=cv[dep], cmap=cmap, vmin=0, vmax=vmax,
                       edgecolors="#c0392b", linewidths=1.5,
                       facecolors="none", alpha=0.9, zorder=5)

        # Labels
        labels = []
        for _, row in pdf.iterrows():
            pfx  = "▽ " if row["direction"] == "depleted" else "   "
            ns_l = NS_SHORT.get(row["namespace"], "?")
            term = row["term"][:55] + ("…" if len(row["term"]) > 55 else "")
            labels.append(f"{pfx}[{ns_l}] {term}")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.tick_params(axis="y", length=0, pad=4)

        ax.axvline(x=1, color="#999", lw=1, ls="--", zorder=4)
        fe_vals = pdf["fold_enrich"].values
        xpad    = max(abs(fe_vals.max() - 1), abs(1 - fe_vals.min())) * 0.2
        ax.set_xlim(min(fe_vals.min(), 1) - xpad, fe_vals.max() + xpad * 1.5)
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_xlabel("Fold Enrichment  (▽ = depleted, FE<1)", fontsize=9.5)
        ax.grid(axis="x", color="#ccc", lw=0.5, ls=":", zorder=0)
        for sp in ["top", "right"]:  ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]: ax.spines[sp].set_color("#bbb")

        mode_str = {"fdr": "FDR<0.05", "raw": "raw p<0.05 ⚠",
                    "none": "Top-20 raw p ⚠"}[best_mode]
        ax.set_title(
            f"{group_key}  [{best_pct}% cutoff  n={n_gene}]\n"
            f"{mode_str}  ●enriched={int(enr.sum())}  ○depleted={int(dep.sum())}",
            fontsize=9.5, fontweight="bold",
            color=GROUP_COLOR[group_key], pad=8,
        )
        if watermark:
            ax.text(0.97, 0.03,
                    f"⚠ {mode_str}\n(exploratory, not FDR-corrected)",
                    ha="right", va="bottom", transform=ax.transAxes,
                    fontsize=7.5, color="#e74c3c",
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="white", ec="#e74c3c", alpha=0.6))

    _draw(ax_high, plot_high, G_HIGH, len(high_best))
    _draw(ax_low,  plot_low,  G_LOW,  len(low_best))
    plt.subplots_adjust(wspace=0.60, bottom=0.11)

    # Colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.011, 0.65])
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=matplotlib.colors.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("−log₁₀(FDR)" if best_mode == "fdr" else "−log₁₀(raw p)",
                 fontsize=9.5, labelpad=7)
    cb.ax.tick_params(labelsize=8)

    # Legends
    ns_patches = [mpatches.Patch(facecolor=NS_COLOR[ns], alpha=0.75,
                                 edgecolor="white", label=NS_SHORT[ns])
                  for ns in NS_ORDER]
    _all = [d for d in [plot_high, plot_low] if not d.empty]
    ratios = (pd.concat([d["gene_ratio"] for d in _all])
              if _all else pd.Series([0.01, 0.5]))
    r_lo, r_hi = ratios.min(), ratios.max()
    r_mid = (r_lo + r_hi) / 2
    size_hdl = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
               markersize=np.sqrt(40 + (r - r_lo) / (r_hi - r_lo + 1e-10) * 240),
               label=f"{r*100:.1f}%", markeredgecolor="#aaa")
        for r in [r_lo, r_mid, r_hi]
    ]
    dir_hdl = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#555",
               markersize=9, markeredgecolor="#555", label="● enriched (FE>1)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markersize=9, markeredgecolor="#c0392b", lw=1.5,
               label="○ depleted (FE<1, ▽)"),
    ]
    leg1 = fig.legend(handles=ns_patches, title="Namespace",
                      loc="lower left", bbox_to_anchor=(0.03, 0.01),
                      fontsize=8.5, title_fontsize=9, framealpha=0.9, ncol=3)
    leg2 = fig.legend(handles=size_hdl, title="Gene Ratio",
                      loc="lower left", bbox_to_anchor=(0.28, 0.01),
                      fontsize=8.5, title_fontsize=9, framealpha=0.9, ncol=3)
    fig.legend(handles=dir_hdl, title="Direction",
               loc="lower left", bbox_to_anchor=(0.53, 0.01),
               fontsize=8.5, title_fontsize=9, framealpha=0.9, ncol=1)
    fig.add_artist(leg1)
    fig.add_artist(leg2)

    mode_title = {
        "fdr":  "FDR<0.05 (BH-corrected)",
        "raw":  "raw p<0.05 (exploratory, uncorrected)",
        "none": "Top-20 raw p (no significant enrichment found)",
    }[best_mode]
    fig.suptitle(
        f"GO Enrichment: High vs Low cross-weight c — {best_pct}% cutoff\n"
        f"c = σ(Q·KV/√d)  |  {mode_title}  |  S. cerevisiae",
        fontsize=11.5, fontweight="bold", y=1.01, color="#111",
    )
    save_fig(fig, f"GO_enrichment_crossweight_c_{best_pct}pct")


print("\n── 4. Fisher plots ─────────────────────────────────────")
plot_threshold_heatmap(scan_df)
plot_fisher_lollipop(plot_high, plot_low, BEST_PCT, BEST_MODE, high_best, low_best)


# Console summary of Fisher result
print("\n" + "═" * 72)
print(f"Fisher summary  [cutoff={BEST_PCT}%  mode={BEST_MODE}]")
print("═" * 72)
mode_p = "p_fdr" if BEST_MODE == "fdr" else "p_raw"
for gname, gdf in [(G_HIGH, df_high), (G_LOW, df_low)]:
    print(f"\n[{gname}]")
    if gdf.empty:
        print("  (no significant terms)")
        continue
    for direction in ["enriched", "depleted"]:
        sub = gdf[gdf["direction"] == direction].nsmallest(8, mode_p)
        if sub.empty:
            continue
        arrow = "↑" if direction == "enriched" else "↓"
        print(f"  ── {direction.upper()} {arrow} ──")
        for _, row in sub.iterrows():
            ns = NS_SHORT.get(row["namespace"], "?")
            print(f"  [{ns}] {row['term'][:50]:<50}"
                  f"  FE={row['fold_enrich']:.2f}"
                  f"  {mode_p.upper()}={row[mode_p]:.2e}"
                  f"  n={row['study_count']}/{row['study_n']}")


# ══════════════════════════════════════════════════════════════════════
#  5. MANN-WHITNEY ENRICHMENT (continuous, no threshold — MAIN ANALYSIS)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("Section 5: Mann-Whitney continuous enrichment")
print("═" * 72)


def run_mw_enrichment(
    c_series: pd.Series, go2genes_dict: dict[str, set], godag: GODag,
    min_genes: int = MW_MIN_GENES, max_genes: int = MW_MAX_GENES,
) -> pd.DataFrame:
    """
    Per-GO-term two-sided Mann-Whitney U comparing c-values of annotated
    genes vs the rest of the background.

    Effect size: rank biserial  r = 2U / (n1·n2) − 1  ∈ [−1, +1]
      r > 0 → annotated genes have higher c  → high-c enriched
      r < 0 → annotated genes have lower  c  → low-c  enriched

    BH-FDR applied to all terms tested in a single call.
    """
    c_idx_set = set(c_series.index)
    rows = []
    for go_id, genes_bg in go2genes_dict.items():
        genes_with_c = [g for g in genes_bg if g in c_idx_set]
        n_ann = len(genes_with_c)
        if n_ann < min_genes or n_ann > max_genes:
            continue
        c_ann = c_series.loc[genes_with_c].values
        c_non = c_series[~c_series.index.isin(genes_bg)].values
        if len(c_non) < 5:
            continue

        U, p_two = mannwhitneyu(c_ann, c_non, alternative="two-sided")
        n1, n2 = len(c_ann), len(c_non)
        r_rb   = 2 * U / (n1 * n2) - 1

        node = godag.get(go_id)
        if node is None or node.namespace not in NS_SHORT:
            continue

        rows.append(dict(
            GO_id         = go_id,
            term          = node.name,
            namespace     = node.namespace,
            n_annotated   = n_ann,
            n_background  = n1 + n2,
            mean_c_ann    = float(c_ann.mean()),
            mean_c_all    = float(c_series.mean()),
            rank_biserial = r_rb,
            U_stat        = U,
            p_two         = p_two,
            direction     = "high_c" if r_rb >= 0 else "low_c",
        ))

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    fdr = bh_fdr(df["p_two"].values)
    df["p_fdr"]         = fdr
    df["neg_log10_fdr"] = -np.log10(np.maximum(fdr,                   1e-300))
    df["neg_log10_p"]   = -np.log10(np.maximum(df["p_two"].values, 1e-300))
    return df.sort_values("p_fdr").reset_index(drop=True)


# ─── 5a. Full-GO MW ────────────────────────────────────────────────────
if c_series.empty:
    print("  ⚠ No c_series data (summary CSV not used) — skipping MW analysis")
    mw_df      = pd.DataFrame()
    mw_slim_df = pd.DataFrame()
else:
    print(f"  ✓ genes with c values: {len(c_series)}")
    mw_df = run_mw_enrichment(c_series, go2genes_bg, godag)
    if not mw_df.empty:
        n_h = int(((mw_df["p_fdr"] < FDR_CUTOFF) & (mw_df["direction"] == "high_c")).sum())
        n_l = int(((mw_df["p_fdr"] < FDR_CUTOFF) & (mw_df["direction"] == "low_c")).sum())
        print(f"  Full GO MW: n_tested={len(mw_df)}  "
              f"FDR<0.05 high_c={n_h}  low_c={n_l}")
    mw_csv = os.path.join(PC["output_dir"], "GO_mw_continuous_enrichment.csv")
    mw_df.to_csv(mw_csv, index=False)
    print(f"  ✓ MW table → {mw_csv}")


# ─── 5b. GO Slim MW ────────────────────────────────────────────────────
print("\n── GO Slim (SGD go_slim_mapping.tab) ─────────────────")
go_slim_ids: set[str] = set()
slim_tab: pd.DataFrame | None = None

if os.path.exists(PC["go_slim_tab"]):
    try:
        slim_tab = pd.read_csv(
            PC["go_slim_tab"], sep="\t", header=None, comment="!",
            names=["gene", "sgdid", "slim_term", "go_id", "aspect"],
            dtype=str,
        )
        slim_tab = slim_tab[slim_tab["go_id"].notna()].copy()
        slim_tab["go_id"] = slim_tab["go_id"].str.strip()
        has_aspect = (slim_tab["aspect"].notna().any() and
                      slim_tab["aspect"].str.strip().isin(["P", "F", "C"]).any())
        if has_aspect:
            slim_tab["aspect"] = slim_tab["aspect"].str.strip()
        else:
            # Recover aspect from godag
            ns2asp = {"biological_process": "P",
                      "molecular_function": "F",
                      "cellular_component":  "C"}
            slim_tab["aspect"] = slim_tab["go_id"].map(
                lambda gid: ns2asp.get(godag.get(gid).namespace)
                            if godag.get(gid) else None)
        slim_tab = slim_tab[slim_tab["aspect"].isin(["P", "F", "C"])].copy()
        go_slim_ids = set(slim_tab["go_id"].unique())
        print(f"  ✓ GO Slim terms: {len(go_slim_ids)}  "
              f"(genes: {slim_tab['gene'].nunique()})")
    except Exception as e:
        print(f"  ✗ parse error: {e}")
else:
    # Fallback: from obo goslim_yeast subset (often empty)
    print(f"  [fallback] {PC['go_slim_tab']} missing — scanning obo subset")
    for go_id, node in godag.items():
        if "goslim_yeast" in (getattr(node, "subset", None) or set()):
            go_slim_ids.add(go_id)
    print(f"  ✓ goslim_yeast from OBO: {len(go_slim_ids)}")
    print(f"  → download slim file from: "
          f"https://downloads.yeastgenome.org/curation/literature/go_slim_mapping.tab")

go2genes_slim = {gid: g for gid, g in go2genes_bg.items() if gid in go_slim_ids}
print(f"  ✓ GO Slim terms w/ bg annotations: {len(go2genes_slim)}")

# Diagnostic: eisosome terms in GO Slim
_found = False
for gid in go2genes_slim:
    node = godag.get(gid)
    if node and "eisosome" in node.name.lower():
        g_set = go2genes_slim[gid] & set(c_series.index)
        mc = c_series[list(g_set)].mean() if len(g_set) > 0 else float("nan")
        print(f"  [diag] eisosome: {gid}  {node.name}  n={len(g_set)}  mean_c={mc:.4f}")
        _found = True
if not _found:
    print("  [diag] no eisosome GO Slim term found")

if go2genes_slim and not c_series.empty:
    mw_slim_df = run_mw_enrichment(
        c_series, go2genes_slim, godag,
        min_genes=SLIM_MIN_GENES, max_genes=SLIM_MAX_GENES,
    )
    if not mw_slim_df.empty:
        n_h = int(((mw_slim_df["p_fdr"] < FDR_CUTOFF) & (mw_slim_df["direction"] == "high_c")).sum())
        n_l = int(((mw_slim_df["p_fdr"] < FDR_CUTOFF) & (mw_slim_df["direction"] == "low_c")).sum())
        print(f"  GO Slim MW: n_tested={len(mw_slim_df)}  "
              f"FDR<0.05 high_c={n_h}  low_c={n_l}")
else:
    mw_slim_df = pd.DataFrame()
    print("  (GO Slim MW skipped: no terms or no c_series)")

slim_csv = os.path.join(PC["output_dir"], "GO_slim_mw_enrichment.csv")
mw_slim_df.to_csv(slim_csv, index=False)
print(f"  ✓ GO Slim MW table → {slim_csv}")


# ══════════════════════════════════════════════════════════════════════
#  6. MW VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════

def _mw_select_top(
    mw_result_df: pd.DataFrame, top_n: int, fdr_cutoff: float,
) -> tuple[pd.DataFrame, bool, str]:
    """Apply FDR filter; fall back to top raw-p with watermark."""
    sig = mw_result_df[mw_result_df["p_fdr"] < fdr_cutoff].copy()
    is_exploratory = sig.empty
    if is_exploratory:
        sig = mw_result_df.nsmallest(min(top_n * 2, len(mw_result_df)), "p_two").copy()
        mode_str = f"Top-{len(sig)} raw p (no FDR<{fdr_cutoff} terms; exploratory)"
    else:
        mode_str = f"FDR<{fdr_cutoff} (BH-corrected, n={len(sig)})"
    return sig, is_exploratory, mode_str


def plot_mw_lollipop(
    mw_result_df: pd.DataFrame, title_suffix: str, out_prefix: str,
    top_n: int = 15, fdr_cutoff: float = FDR_CUTOFF,
) -> None:
    """Single-axis two-sided lollipop (rank-biserial on x-axis)."""
    if mw_result_df.empty:
        print(f"  [MW lollipop] {title_suffix}: no data, skipped")
        return
    sig, is_exploratory, mode_str = _mw_select_top(mw_result_df, top_n, fdr_cutoff)

    high_top = (sig[sig["direction"] == "high_c"]
                .nlargest(top_n, "neg_log10_fdr")
                .sort_values("rank_biserial", ascending=True))
    low_top  = (sig[sig["direction"] == "low_c"]
                .nsmallest(top_n, "rank_biserial")
                .sort_values("rank_biserial", ascending=True))
    plot_df = pd.concat([low_top, high_top], ignore_index=True)
    if plot_df.empty:
        print(f"  [MW lollipop] {title_suffix}: no plottable terms, skipped")
        return

    n = len(plot_df)
    fig, ax = plt.subplots(figsize=(15, max(8, n * 0.44 + 3.0)))
    fig.patch.set_facecolor("white")

    cmap_mw = plt.cm.plasma
    vmax_c  = min(plot_df["neg_log10_fdr"].max() + 0.5, 30)
    norm_mw = matplotlib.colors.Normalize(vmin=0, vmax=vmax_c)
    y = np.arange(n)

    for idx, (_, row) in enumerate(plot_df.iterrows()):
        ns = row["namespace"]
        ax.axhspan(idx - 0.5, idx + 0.5,
                   color=NS_LIGHT.get(ns, "#f0f0f0"), alpha=0.45, zorder=1)
        col = GROUP_COLOR[G_HIGH] if row["direction"] == "high_c" else GROUP_COLOR[G_LOW]
        ax.plot([0, row["rank_biserial"]], [idx, idx],
                color=col, lw=1.6, zorder=3, alpha=0.55, solid_capstyle="round")

    log_n = np.log1p(plot_df["n_annotated"].values)
    sizes = 40 + (log_n - log_n.min()) / (log_n.max() - log_n.min() + 1e-9) * 220

    ax.scatter(
        plot_df["rank_biserial"].values, y, s=sizes,
        c=plot_df["neg_log10_fdr"].values, cmap=cmap_mw, norm=norm_mw,
        edgecolors="#444", linewidths=0.5, alpha=0.95, zorder=5,
    )

    labels = []
    for _, row in plot_df.iterrows():
        ns_l  = NS_SHORT.get(row["namespace"], "?")
        arrow = "▶" if row["direction"] == "high_c" else "◀"
        term  = row["term"][:55] + ("…" if len(row["term"]) > 55 else "")
        labels.append(f"{arrow} [{ns_l}] {term}  r={row['rank_biserial']:+.3f}")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.tick_params(axis="y", length=0, pad=4)

    ax.axvline(x=0, color="#777", lw=1.4, ls="--", zorder=4)
    rb_abs = max(abs(plot_df["rank_biserial"].values).max() * 1.25, 0.05)
    ax.set_xlim(-rb_abs, rb_abs)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel(
        "Rank Biserial r  (◀ negative: Low-c enriched  |  positive: High-c enriched ▶)",
        fontsize=9.5,
    )
    ax.grid(axis="x", color="#ccc", lw=0.5, ls=":", zorder=0)
    for sp in ["top", "right"]:   ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]: ax.spines[sp].set_color("#bbb")

    sm = plt.cm.ScalarMappable(cmap=cmap_mw, norm=norm_mw); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.01)
    cb.set_label("−log₁₀(FDR)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # Combined legend
    ns_patches = [mpatches.Patch(facecolor=NS_COLOR[ns], alpha=0.75,
                                 edgecolor="white", label=NS_SHORT[ns])
                  for ns in NS_ORDER]
    dir_hdl = [
        mpatches.Patch(facecolor=GROUP_COLOR[G_HIGH], alpha=0.7,
                       label="▶ High-c (r > 0)"),
        mpatches.Patch(facecolor=GROUP_COLOR[G_LOW], alpha=0.7,
                       label="◀ Low-c  (r < 0)"),
    ]
    n_vals = plot_df["n_annotated"].values
    n_lo, n_hi = int(n_vals.min()), int(n_vals.max())
    size_hdl = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
               markeredgecolor="#aaa",
               markersize=np.sqrt(40 + (v - n_lo) / max(n_hi - n_lo, 1) * 220),
               label=f"n={v}")
        for v in [n_lo, (n_lo + n_hi) // 2, n_hi]
    ]
    ax.legend(
        handles=ns_patches + dir_hdl + size_hdl,
        loc="lower right", fontsize=7.8, framealpha=0.9,
        title="Namespace / Direction / n(genes)", title_fontsize=8.5, ncol=1,
    )

    if is_exploratory:
        ax.text(0.97, 0.97, "⚠ Exploratory: no FDR<0.05 terms",
                ha="right", va="top", transform=ax.transAxes,
                fontsize=8, color="#e74c3c",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="white", ec="#e74c3c", alpha=0.7))

    ax.set_title(
        f"GO Enrichment — Mann-Whitney Continuous Ranking  [{title_suffix}]\n"
        f"▶ High cross-weight c  vs  ◀ Low cross-weight c  |  {mode_str}\n"
        f"Effect size: rank biserial r  |  n(terms tested)={len(mw_result_df)}  |  S. cerevisiae",
        fontsize=10.5, fontweight="bold", pad=10,
    )
    plt.tight_layout()
    save_fig(fig, out_prefix)


def plot_mw_lollipop_split(
    mw_result_df: pd.DataFrame, title_suffix: str, out_prefix: str,
    top_n: int = 20, fdr_cutoff: float = FDR_CUTOFF,
) -> None:
    """Two-panel lollipop (left=Low-c, right=High-c) for GO Slim."""
    if mw_result_df.empty:
        print(f"  [MW split] {title_suffix}: no data, skipped")
        return
    sig, is_exploratory, mode_str = _mw_select_top(mw_result_df, top_n, fdr_cutoff)

    high_top = (sig[sig["direction"] == "high_c"]
                .nlargest(top_n, "neg_log10_fdr")
                .sort_values("rank_biserial", ascending=False))
    low_top  = (sig[sig["direction"] == "low_c"]
                .nsmallest(top_n, "rank_biserial")
                .sort_values("rank_biserial", ascending=False))
    n_h, n_l = len(high_top), len(low_top)
    if n_h == 0 and n_l == 0:
        print(f"  [MW split] {title_suffix}: no plottable terms, skipped")
        return

    total_n = max(n_h, n_l, 4)
    wl = max(n_l, 2); wh = max(n_h, 2)
    fig_w = 7 + wl * 0.35 + wh * 0.45
    fig_h = max(8, total_n * 0.46 + 4.0)

    cmap_mw = plt.cm.plasma
    vmax_c  = min(sig["neg_log10_fdr"].max() + 0.5, 30)
    norm_mw = matplotlib.colors.Normalize(vmin=0, vmax=vmax_c)

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        1, 2, width_ratios=[wl, wh], wspace=0.55,
        left=0.01, right=0.88, top=0.88, bottom=0.10, figure=fig,
    )
    ax_l = fig.add_subplot(gs[0])
    ax_h = fig.add_subplot(gs[1])

    def _draw_half(ax, df, direction):
        color = GROUP_COLOR[G_HIGH] if direction == "high_c" else GROUP_COLOR[G_LOW]
        if df.empty:
            ax.text(0.5, 0.5, "No FDR<0.05 terms", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="#888", style="italic")
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.set_xticks([]); ax.set_yticks([])
            return
        n = len(df)
        y_offset = (total_n - n) / 2
        y = np.arange(n) + y_offset

        log_n = np.log1p(df["n_annotated"].values)
        denom = log_n.max() - log_n.min() + 1e-9
        sizes = 50 + (log_n - log_n.min()) / denom * 230

        for idx, (_, row) in enumerate(df.iterrows()):
            ns = row["namespace"]
            yi = y[idx]
            ax.axhspan(yi - 0.45, yi + 0.45,
                       color=NS_LIGHT.get(ns, "#f0f0f0"), alpha=0.45, zorder=1)
            ax.plot([0, row["rank_biserial"]], [yi, yi],
                    color=color, lw=1.8, zorder=3, alpha=0.55, solid_capstyle="round")
        ax.scatter(df["rank_biserial"].values, y, s=sizes,
                   c=df["neg_log10_fdr"].values, cmap=cmap_mw, norm=norm_mw,
                   edgecolors="#444", linewidths=0.5, alpha=0.95, zorder=5)

        labels = []
        for _, row in df.iterrows():
            ns_l = NS_SHORT.get(row["namespace"], "?")
            term = row["term"][:52] + ("…" if len(row["term"]) > 52 else "")
            labels.append(f"[{ns_l}] {term}  r={row['rank_biserial']:+.3f}")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.0)
        ax.tick_params(axis="y", length=0, pad=3)

        ax.axvline(x=0, color="#777", lw=1.2, ls="--", zorder=4)
        rb_abs = max(abs(df["rank_biserial"].values).max() * 1.35, 0.08)
        if direction == "high_c":
            ax.set_xlim(-rb_abs * 0.12, rb_abs * 1.15)
        else:
            ax.set_xlim(-rb_abs * 1.15, rb_abs * 0.12)
        ax.set_ylim(-0.7, total_n - 0.3)
        ax.grid(axis="x", color="#ccc", lw=0.5, ls=":", zorder=0)
        for sp in ["top", "right"]:   ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]: ax.spines[sp].set_color("#ccc")

    _draw_half(ax_l, low_top,  "low_c")
    _draw_half(ax_h, high_top, "high_c")

    ax_l.set_xlabel("Rank Biserial r  (◀ Low-c enriched)",  fontsize=9)
    ax_h.set_xlabel("Rank Biserial r  (▶ High-c enriched)", fontsize=9)
    ax_l.set_title(f"◀  Low cross-weight c   n={n_l}",
                   fontsize=10.5, fontweight="bold",
                   color=GROUP_COLOR[G_LOW], pad=10)
    ax_h.set_title(f"▶  High cross-weight c   n={n_h}",
                   fontsize=10.5, fontweight="bold",
                   color=GROUP_COLOR[G_HIGH], pad=10)

    cbar_ax = fig.add_axes([0.905, 0.18, 0.013, 0.62])
    sm = plt.cm.ScalarMappable(cmap=cmap_mw, norm=norm_mw); sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("−log₁₀(FDR)", fontsize=9, labelpad=6)
    cb.ax.tick_params(labelsize=7.5)

    ns_patches = [mpatches.Patch(facecolor=NS_COLOR[ns], alpha=0.75,
                                 edgecolor="white", label=NS_SHORT[ns])
                  for ns in NS_ORDER]
    fig.legend(handles=ns_patches, title="Namespace",
               loc="lower center", bbox_to_anchor=(0.44, 0.01),
               fontsize=8.5, title_fontsize=9, framealpha=0.9, ncol=3)
    if is_exploratory:
        fig.text(0.44, 0.035, "⚠ Exploratory: no FDR<0.05 terms",
                 ha="center", fontsize=8, color="#e74c3c")

    fig.suptitle(
        f"GO Slim Enrichment — Mann-Whitney  [{title_suffix}]\n"
        f"◀ Low cross-weight c  vs  High cross-weight c ▶  |  {mode_str}",
        fontsize=11.5, fontweight="bold", y=0.975,
    )
    save_fig(fig, out_prefix)


# ─── Run MW plots ──────────────────────────────────────────────────────
if not mw_df.empty:
    print("\n── Full GO MW lollipop ─────────────────────────────")
    plot_mw_lollipop(mw_df, "Full GO", "GO_mw_lollipop")

if not mw_slim_df.empty:
    print("\n── GO Slim MW split lollipop ───────────────────────")
    plot_mw_lollipop_split(mw_slim_df, "GO Slim (goslim_yeast)",
                           "GO_slim_mw_lollipop", top_n=20)


# ─── MW console summary ───────────────────────────────────────────────
print("\n" + "═" * 72)
print("MW summary  (FDR<0.05, top-10 per direction)")
print("═" * 72)
for label, df_tmp in [("Full GO", mw_df), ("GO Slim", mw_slim_df)]:
    print(f"\n[{label}]")
    if df_tmp.empty:
        print("  (no data)")
        continue
    for direction, arrow in [("high_c", "▶"), ("low_c", "◀")]:
        sub = (df_tmp[(df_tmp["p_fdr"] < FDR_CUTOFF) &
                      (df_tmp["direction"] == direction)]
               .nsmallest(10, "p_fdr"))
        print(f"  {arrow} {direction.upper()}")
        if sub.empty:
            print("    (no FDR<0.05 term)")
        else:
            for _, row in sub.iterrows():
                ns = NS_SHORT.get(row["namespace"], "?")
                print(f"    [{ns}] {row['term'][:50]:<50}"
                      f"  r={row['rank_biserial']:+.3f}"
                      f"  FDR={row['p_fdr']:.2e}"
                      f"  n={row['n_annotated']}")




# ══════════════════════════════════════════════════════════════════════
#  7. CANONICAL GROWTH-vs-SPORULATION GENE SIGNATURE TABLE
# ══════════════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("Section 7: Canonical gene-level signature table")
print("═" * 72)
print("""
  Resolves canonical growth-program and sporulation/meiosis genes from the
  literature (ribosomal proteins, glycolysis, IME/NDT80 regulon, synaptonemal
  complex) to ORF IDs via the GAF name->ORF map.  Reports c value, percentile
  rank, and iaa_mean_sens.  Expected: growth genes → top of c distribution;
  sporulation genes → bottom.
""")


def resolve_gene_list(std_names: list[str], name_to_orf: dict, c_series: pd.Series) -> pd.DataFrame:
    """Map standard names to ORF IDs; filter to those present in c_series."""
    rows = []
    for name in std_names:
        orf = name_to_orf.get(name.upper())
        if orf is None:
            continue
        if orf not in c_series.index:
            continue
        rows.append({"std_name": name, "orf_id": orf})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["std_name", "orf_id"])


if USE_SUMMARY and not c_series.empty:
    c_sorted = c_series.sort_values(ascending=False)
    n_total_c = len(c_sorted)

    def _pct_rank(orf):
        """Percentile rank from top (100 = highest c)."""
        pos = c_sorted.index.get_loc(orf) if orf in c_sorted.index else None
        return round(100 * (1 - pos / n_total_c), 1) if pos is not None else float("nan")

    sig_rows = []
    for group_label, gene_list in [
        ("growth", CANONICAL_GROWTH_GENES),
        ("sporulation", CANONICAL_SPORULATION_GENES),
    ]:
        resolved = resolve_gene_list(gene_list, name_to_orf, c_series)
        if resolved.empty:
            print(f"  ⚠ {group_label}: no genes resolved — check GAF name column")
            continue
        for _, row in resolved.iterrows():
            orf = row["orf_id"]
            c_val = float(c_series[orf])
            iaa_val = (float(iaa_series[orf])
                       if iaa_series is not None and orf in iaa_series.index
                       else float("nan"))
            sig_rows.append(dict(
                group=group_label,
                std_name=row["std_name"],
                orf_id=orf,
                cross_weight_c=round(c_val, 5),
                pct_rank_from_top=_pct_rank(orf),
                iaa_mean_sens=round(iaa_val, 5) if not np.isnan(iaa_val) else float("nan"),
            ))

    sig_df = pd.DataFrame(sig_rows)
    if not sig_df.empty:
        sig_csv = os.path.join(PC["output_dir"], "GO_canonical_gene_signature.csv")
        sig_df.to_csv(sig_csv, index=False)
        print(f"  ✓ Canonical gene table → {sig_csv}  (n={len(sig_df)})")

        # Console summary
        for grp in ["growth", "sporulation"]:
            sub = sig_df[sig_df["group"] == grp]
            if sub.empty:
                continue
            print(f"\n  [{grp.upper()}]  n_resolved={len(sub)}")
            print(f"    c range: [{sub['cross_weight_c'].min():.4f}, "
                  f"{sub['cross_weight_c'].max():.4f}]  "
                  f"mean={sub['cross_weight_c'].mean():.4f}  "
                  f"mean_pct_rank={sub['pct_rank_from_top'].mean():.1f}%")

        # MW test: do growth and sporulation genes differ in c?
        c_growth = sig_df[sig_df["group"] == "growth"]["cross_weight_c"].values
        c_spore  = sig_df[sig_df["group"] == "sporulation"]["cross_weight_c"].values
        if len(c_growth) >= 3 and len(c_spore) >= 3:
            _, p_gs = mannwhitneyu(c_growth, c_spore, alternative="two-sided")
            print(f"\n  Growth vs Sporulation c-values  MWU two-sided p = {p_gs:.3e}")
            if p_gs < 0.05:
                print("  ✓ Significant separation — supports growth-vs-differentiation axis")
            else:
                print("  ⚠ Not significant at p<0.05 with resolved gene subset")

        # Stripplot
        try:
            fig, ax = plt.subplots(figsize=(9, 5))
            fig.patch.set_facecolor("white")
            rng = np.random.default_rng(42)
            colors = {"growth": GROUP_COLOR[G_HIGH], "sporulation": GROUP_COLOR[G_LOW]}
            x_pos  = {"growth": 1, "sporulation": 2}
            labels_plotted: set = set()
            for _, row in sig_df.iterrows():
                grp  = row["group"]
                xpos = x_pos[grp] + rng.uniform(-0.15, 0.15)
                col  = colors[grp]
                ax.scatter(xpos, row["cross_weight_c"], color=col,
                           s=55, alpha=0.75, zorder=4, edgecolors="#333", lw=0.4)
                if not np.isnan(row["pct_rank_from_top"]):
                    label_text = row["std_name"]
                    ax.annotate(label_text, (xpos, row["cross_weight_c"]),
                                fontsize=6.2, ha="left", va="center",
                                xytext=(4, 0), textcoords="offset points", color="#333")

            # Overlay median bars
            for grp, pos in x_pos.items():
                sub = sig_df[sig_df["group"] == grp]["cross_weight_c"]
                if not sub.empty:
                    ax.hlines(sub.median(), pos - 0.22, pos + 0.22,
                              color=colors[grp], lw=2.5, zorder=5)

            ax.set_xticks([1, 2])
            ax.set_xticklabels(["Growth / Biosynthesis\n(high-c expected)",
                                 "Sporulation / Meiosis\n(low-c expected)"],
                               fontsize=10)
            ax.set_ylabel("cross_weight c", fontsize=10)
            ax.set_xlim(0.5, 2.8)
            ax.set_title(
                "Canonical gene-level validation: growth vs sporulation programs\n"
                "Horizontal bar = group median",
                fontsize=10.5, fontweight="bold",
            )
            ax.grid(axis="y", color="#ddd", lw=0.6)
            for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
            if len(c_growth) >= 3 and len(c_spore) >= 3:
                ax.text(0.98, 0.96, f"MWU p = {p_gs:.2e}",
                        ha="right", va="top", transform=ax.transAxes,
                        fontsize=9, color="#333",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc"))
            plt.tight_layout()
            out7 = os.path.join(PC["output_dir"], "GO_canonical_gene_stripplot.png")
            fig.savefig(out7, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  ✓ Stripplot → {out7}")
        except Exception as e:
            print(f"  ✗ stripplot failed: {e}")
    else:
        print("  ⚠ No canonical genes resolved (check GAF name_to_orf)")
else:
    print("  (skipped: no c_series)")


# ══════════════════════════════════════════════════════════════════════
#  8. BRAUER 2008 / GASCH 2000 EXTERNAL SIGNATURE COMPARISON
# ══════════════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("Section 8: Brauer 2008 / Gasch 2000 external signature comparison")
print("═" * 72)
print("""
  Tests whether published gene sets from two landmark yeast genomics studies
  are enriched at the high-c or low-c end of the cross-weight distribution.

  Brauer et al. 2008 Mol Biol Cell:
    BRAUER_UP   — genes positively correlated with chemostat growth rate.
                  Expected: HIGH c  (→ confirms c tracks growth program).
    BRAUER_DOWN — genes anti-correlated with growth rate (stress/starvation).
                  Expected: LOW c.

  Gasch et al. 2000 Mol Biol Cell — Environmental Stress Response (ESR):
    GASCH_ESR_INDUCED   — induced under stress, repressed during growth.
                          Expected: LOW c.
    GASCH_ESR_REPRESSED — repressed under stress (i.e., growth-associated).
                          Expected: HIGH c.

  All tests use Mann-Whitney U (two-sided) with rank-biserial effect size.
  Positive r → set has higher c than background (high-c enriched).
  Negative r → set has lower  c than background (low-c enriched).
""")


def external_sig_mw(
    gene_names: list[str],
    label: str,
    name_to_orf: dict,
    c_series: pd.Series,
) -> dict | None:
    """Resolve names → ORFs, run MW vs background, return result dict."""
    orfs = [name_to_orf[n.upper()] for n in gene_names
             if n.upper() in name_to_orf and name_to_orf[n.upper()] in c_series.index]
    orfs = list(set(orfs))
    n = len(orfs)
    if n < 5:
        print(f"  ⚠ {label}: only {n} ORFs resolved (need ≥5) — skipping")
        return None
    c_set = c_series[orfs].values
    c_bg  = c_series[~c_series.index.isin(orfs)].values
    U, p  = mannwhitneyu(c_set, c_bg, alternative="two-sided")
    r_rb  = 2 * U / (n * len(c_bg)) - 1
    print(f"  [{label}]  n_resolved={n}  "
          f"mean_c={c_set.mean():.4f}  r={r_rb:+.4f}  p={p:.3e}")
    return dict(label=label, n_resolved=n, mean_c=float(c_set.mean()),
                rank_biserial=float(r_rb), p_two=float(p))


if USE_SUMMARY and not c_series.empty:
    external_sigs = [
        ("Brauer UP (growth ↑)",     BRAUER_GROWTH_UP,      "high_c"),
        ("Brauer DOWN (growth ↓)",   BRAUER_GROWTH_DOWN,    "low_c"),
        ("Gasch ESR induced",        GASCH_ESR_INDUCED,     "low_c"),
        ("Gasch ESR repressed",      GASCH_ESR_REPRESSED,   "high_c"),
    ]

    ext_rows = []
    c_by_sig: dict[str, np.ndarray] = {}
    for label, gene_list, expected_dir in external_sigs:
        res = external_sig_mw(gene_list, label, name_to_orf, c_series)
        if res is None:
            continue
        res["expected_direction"] = expected_dir
        confirmed = (
            (expected_dir == "high_c" and res["rank_biserial"] > 0) or
            (expected_dir == "low_c"  and res["rank_biserial"] < 0)
        )
        res["direction_confirmed"] = confirmed
        ext_rows.append(res)

        orfs = [name_to_orf[n.upper()] for n in gene_list
                if n.upper() in name_to_orf and name_to_orf[n.upper()] in c_series.index]
        c_by_sig[label] = c_series[list(set(orfs))].values

    if ext_rows:
        ext_df = pd.DataFrame(ext_rows)
        # BH FDR across the 4 tests
        ext_df["p_fdr"] = bh_fdr(ext_df["p_two"].values)
        ext_csv = os.path.join(PC["output_dir"], "GO_external_signature_comparison.csv")
        ext_df.to_csv(ext_csv, index=False)
        print(f"\n  ✓ External signature table → {ext_csv}")

        n_confirmed = ext_df["direction_confirmed"].sum()
        print(f"  Direction confirmed: {n_confirmed}/{len(ext_df)} signatures")
        print(f"  (Positive r = high-c enriched; Negative r = low-c enriched)")

        # Violin / strip plot
        try:
            n_sigs = len(c_by_sig)
            if n_sigs > 0:
                fig, ax = plt.subplots(figsize=(max(10, n_sigs * 2.4), 5.5))
                fig.patch.set_facecolor("white")

                # Background violin
                ax.violinplot([c_series.values],
                              positions=[0],
                              widths=0.55, showmedians=True,
                              bw_method=0.35)
                ax.set_xticks([0])
                ax.set_xticklabels(["Background\n(all genes)"], fontsize=8)

                positions = list(range(1, n_sigs + 1))
                sig_colors = []
                for i, (label, c_arr) in enumerate(c_by_sig.items(), start=1):
                    row = ext_df[ext_df["label"] == label].iloc[0]
                    exp = row["expected_direction"]
                    confirmed = row["direction_confirmed"]
                    col = GROUP_COLOR[G_HIGH] if exp == "high_c" else GROUP_COLOR[G_LOW]
                    sig_colors.append(col)
                    parts = ax.violinplot([c_arr], positions=[i],
                                         widths=0.55, showmedians=True, bw_method=0.35)
                    for pc in parts["bodies"]:
                        pc.set_facecolor(col); pc.set_alpha(0.55)
                    for part_name in ("cbars", "cmins", "cmaxes", "cmedians"):
                        if part_name in parts:
                            parts[part_name].set_color(col)

                    # Annotation: r + p + confirmed
                    ax.text(i, c_arr.max() * 1.01,
                            f"r={row['rank_biserial']:+.3f}\n"
                            f"p={row['p_two']:.2e}\n"
                            f"{'✓' if confirmed else '✗'} {exp}",
                            ha="center", va="bottom", fontsize=7.5,
                            color="#27ae60" if confirmed else "#c0392b")

                # X-tick labels
                all_labels = ["Background\n(all genes)"] + list(c_by_sig.keys())
                ax.set_xticks(range(len(all_labels)))
                ax.set_xticklabels(
                    [l.replace(" (", "\n(") for l in all_labels],
                    fontsize=8, rotation=10, ha="right",
                )
                ax.set_ylabel("cross_weight c", fontsize=10)
                ax.set_title(
                    "External signature validation: Brauer 2008 & Gasch 2000 gene sets\n"
                    "✓ = direction of c shift matches expectation",
                    fontsize=10.5, fontweight="bold",
                )
                ax.grid(axis="y", color="#ddd", lw=0.6)
                for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

                # Legend
                from matplotlib.lines import Line2D as _Line2D
                leg = [
                    mpatches.Patch(facecolor=GROUP_COLOR[G_HIGH], alpha=0.6,
                                   label="Expected HIGH-c signature"),
                    mpatches.Patch(facecolor=GROUP_COLOR[G_LOW],  alpha=0.6,
                                   label="Expected LOW-c signature"),
                ]
                ax.legend(handles=leg, fontsize=8.5, framealpha=0.9, loc="upper right")
                plt.tight_layout()
                out8 = os.path.join(PC["output_dir"], "GO_external_signature_violin.png")
                fig.savefig(out8, dpi=200, bbox_inches="tight")
                plt.close(fig)
                print(f"  ✓ Violin plot → {out8}")
        except Exception as e:
            print(f"  ✗ violin plot failed: {e}")
    else:
        print("  ⚠ No external signatures resolved — check name_to_orf / gene lists")
else:
    print("  (skipped: no c_series)")


# ══════════════════════════════════════════════════════════════════════
#  7. TF-ENCODING GENE SECONDARY ANALYSIS
# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
#  9. TF-ENCODING GENE SECONDARY ANALYSIS  [SUPPLEMENTARY]
# ══════════════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("Section 9: TF-encoding gene secondary analysis  [SUPPLEMENTARY]")
print("═" * 72)
print("""
  NOTE: This analysis is SUPPLEMENTARY.  The four groups (Low-c TF,
  Low-c non-TF, High-c TF, Background) show overlapping IAA sensitivity
  medians (~0.28-0.35), indicating no clear TF-specific signal at the
  current data resolution.  Outputs go to ./result/supp/.
  Context:
    MW enrichment shows low-c genes enriched in 'DNA-binding TF activity'
    MF terms (r ≈ -0.29 to -0.39). These are genes that *encode* TFs
    (annotation = TF function), not genes *regulated by* TFs.
    Question: do low-c TF-encoding genes differ from other low-c genes
    in their IAA-depletion sensitivity?
""")

# Extract TF-encoding gene set
tf_genes = set()
for go_id in TF_GO_TERMS:
    if go_id in go2genes_bg:
        tf_genes.update(go2genes_bg[go_id])
tf_genes = tf_genes & set(c_series.index) if not c_series.empty else set()
print(f"  ✓ TF-encoding genes (GO TF-activity terms): {len(tf_genes)}")

if USE_SUMMARY and not c_series.empty:
    pct_sec7 = max(BEST_PCT, 15)
    k7 = max(int(N_TOTAL * pct_sec7 / 100), 30)
    low_c_set7  = set(ALL_GENES_RANKED[-k7:])
    high_c_set7 = set(ALL_GENES_RANKED[:k7])

    low_c_tf     = low_c_set7  & tf_genes
    low_c_nontf  = low_c_set7  - tf_genes
    high_c_tf    = high_c_set7 & tf_genes
    high_c_nontf = high_c_set7 - tf_genes

    print(f"  [Low-c  {pct_sec7}%]  TF={len(low_c_tf)}  non-TF={len(low_c_nontf)}")
    print(f"  [High-c {pct_sec7}%]  TF={len(high_c_tf)}  non-TF={len(high_c_nontf)}")

    # c-value distribution comparison
    c_bg_tf    = c_series[c_series.index.isin(tf_genes)].values
    c_bg_nontf = c_series[~c_series.index.isin(tf_genes)].values
    _, p_tf = mannwhitneyu(c_bg_tf, c_bg_nontf, alternative="two-sided")
    print(f"\n  Full-background TF vs non-TF c-values  MWU p = {p_tf:.3e}")
    print(f"    TF     mean c = {c_bg_tf.mean():.4f}  (n={len(c_bg_tf)})")
    print(f"    non-TF mean c = {c_bg_nontf.mean():.4f}  (n={len(c_bg_nontf)})")

    # IAA analysis
    if HAS_IAA and iaa_series is not None:
        groups_iaa = {
            "Low-c TF":     iaa_series[iaa_series.index.isin(low_c_tf)].values,
            "Low-c non-TF": iaa_series[iaa_series.index.isin(low_c_nontf)].values,
            "High-c TF":    iaa_series[iaa_series.index.isin(high_c_tf)].values,
            "Background":   iaa_series.values,
        }
        print("\n  IAA sensitivity (iaa_mean_sens) by group:")
        for grp, arr in groups_iaa.items():
            if len(arr) > 0:
                print(f"    {grp:<18s}  n={len(arr):4d}  "
                      f"mean={np.mean(arr):.4f}  median={np.median(arr):.4f}")

        a, b = groups_iaa["Low-c TF"], groups_iaa["Low-c non-TF"]
        if len(a) >= 5 and len(b) >= 5:
            _, p_ab = mannwhitneyu(a, b, alternative="two-sided")
            who = "TF more sensitive" if np.mean(a) > np.mean(b) else "non-TF more sensitive"
            print(f"\n  Low-c TF vs Low-c non-TF  MWU p = {p_ab:.3e}  ({who})")

        # Within-TF Spearman (c vs IAA sensitivity)
        tf_c   = c_series[c_series.index.isin(tf_genes)]
        tf_iaa = iaa_series[iaa_series.index.isin(tf_genes)]
        common = tf_c.index.intersection(tf_iaa.index)
        if len(common) >= 10:
            r_tf, p_tf_sp = spearmanr(tf_c[common], tf_iaa[common])
            print(f"\n  [Within-TF Spearman]  n={len(common)}")
            print(f"    c ~ IAA_sensitivity: r={r_tf:+.4f}  p={p_tf_sp:.3e}")
            if r_tf < 0:
                print(f"    direction: c↑ → IAA↓  "
                      f"(high-c TF-encoding genes less affected by IAA depletion)")
            else:
                print(f"    direction: c↑ → IAA↑  "
                      f"(caution: confounding possible — TF-encoding genes are")
                print(f"    themselves IAA-depletion targets; iaa_mean_sens may")
                print(f"    reflect self-depletion. Suggest excluding IAA-targeted")
                print(f"    TFs in a follow-up run.)")
        else:
            print(f"  ⚠ within-TF Spearman: too few TFs w/ IAA data (n={len(common)} < 10)")

        # Boxplot
        try:
            fig, ax = plt.subplots(figsize=(9, 5))
            fig.patch.set_facecolor("white")
            plot_groups = [(k, v) for k, v in groups_iaa.items() if len(v) > 0]
            bx = ax.boxplot(
                [v for _, v in plot_groups],
                labels=[k for k, _ in plot_groups],
                patch_artist=True, notch=False,
                medianprops=dict(color="black", lw=2),
            )
            colors = ["#1a6fa8", "#5dade2", "#c0392b", "#95a5a6"]
            for patch, c in zip(bx["boxes"], colors[:len(plot_groups)]):
                patch.set_facecolor(c); patch.set_alpha(0.65)
            rng = np.random.default_rng(42)
            for pos, (_, arr) in enumerate(plot_groups, start=1):
                if len(arr) <= 150:
                    jit = rng.uniform(-0.18, 0.18, len(arr))
                    dot = colors[pos - 1] if pos - 1 < len(colors) else "#888"
                    ax.scatter(pos + jit, arr, color=dot, s=14, alpha=0.45, zorder=4)
            ax.set_ylabel("IAA mean sensitivity (|log2FC|)", fontsize=10)
            ax.set_title(
                "IAA sensitivity: Low-c TF-encoding vs non-TF genes\n"
                f"(TF defined by GO:0003700 & related, {pct_sec7}% c cutoff)",
                fontsize=10.5, fontweight="bold",
            )
            ax.grid(axis="y", color="#ddd", lw=0.6)
            for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
            plt.tight_layout()
            out = os.path.join(SUPP_DIR, "GO_tf_iaa_sensitivity.png")
            fig.savefig(out, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  ✓ boxplot → {out}")
        except Exception as e:
            print(f"  ✗ boxplot failed: {e}")
    else:
        print("  ⚠ iaa_mean_sens column absent — skipping IAA comparison")

    # Export low-c TF gene table
    if low_c_tf:
        rows = []
        for gene in sorted(low_c_tf):
            c_val   = float(c_series.get(gene, np.nan))
            iaa_val = (float(iaa_series.get(gene, np.nan))
                       if iaa_series is not None else np.nan)
            terms = []
            for go_id in TF_GO_TERMS:
                if go_id in go2genes_bg and gene in go2genes_bg[go_id]:
                    node = godag.get(go_id)
                    if node:
                        terms.append(node.name)
            rows.append(dict(
                gene=gene, cross_weight_c=c_val, iaa_mean_sens=iaa_val,
                tf_go_terms="; ".join(terms),
            ))
        tf_df = pd.DataFrame(rows).sort_values("cross_weight_c")
        tf_csv = os.path.join(SUPP_DIR, "GO_tf_secondary_analysis.csv")
        tf_df.to_csv(tf_csv, index=False)
        print(f"  ✓ Low-c TF table → {tf_csv}  (n={len(tf_df)})")
else:
    print("  (skipped: no c_series)")


# ══════════════════════════════════════════════════════════════════════
#  10. EISOSOME DEEP-DIVE  [SUPPLEMENTARY]
# ══════════════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("Section 10: Eisosome deep-dive  [SUPPLEMENTARY]")
print("═" * 72)
print("""
  NOTE: This analysis is SUPPLEMENTARY.  Under the current c-axis semantics
  (high-c = growth program), eisosome shows r=+0.24 (p≈0.03, n≈20) — i.e.,
  eisosome genes sit marginally on the HIGH-c (growth) side, consistent with
  their role as housekeeping plasma-membrane structures.  They no longer
  serve as a "low-c negative control"; interpret as "growth-program
  housekeeping component".  Outputs go to ./result/supp/.

  Context:
    Eisosome ([CC], n≈20) shows the strongest low-c rank-biserial among
    *structural* terms in GO Slim MW under the old model.  Eisosomes are
    static PM invaginations regulated by TORC1/PKA in response to nutrient
    state — a textbook case of context-dependent regulation.
""")

eisosome_genes: set[str] = set()
if slim_tab is not None and not slim_tab.empty and not c_series.empty:
    eiso_rows = slim_tab[slim_tab["slim_term"].str.lower().str.strip() == "eisosome"]
    if not eiso_rows.empty:
        EISOSOME_GO = set(eiso_rows["go_id"].unique())
        print(f"  ✓ Eisosome GO IDs from slim tab: {sorted(EISOSOME_GO)}")
        for go_id in EISOSOME_GO:
            if go_id in go2genes_slim:
                eisosome_genes.update(go2genes_slim[go_id])
            elif go_id in go2genes_bg:
                eisosome_genes.update(go2genes_bg[go_id])
        eisosome_genes &= set(c_series.index)

# Fallback: scan GO Slim by term name
if not eisosome_genes and go2genes_slim and not c_series.empty:
    for go_id in go2genes_slim:
        node = godag.get(go_id)
        if node and node.name.lower().strip() == "eisosome":
            eisosome_genes.update(go2genes_slim[go_id])
    eisosome_genes &= set(c_series.index)

# Safety guards
if len(eisosome_genes) < 3:
    print(f"  ⚠ fewer than 3 eisosome genes found (n={len(eisosome_genes)}) — skipping")
    eisosome_genes = set()
elif len(eisosome_genes) > 60:
    print(f"  ⚠ eisosome set unexpectedly large (n={len(eisosome_genes)}) — skipping")
    eisosome_genes = set()

if eisosome_genes:
    c_eiso = c_series[c_series.index.isin(eisosome_genes)].values
    c_bg   = c_series[~c_series.index.isin(eisosome_genes)].values

    _, p_two_eiso = mannwhitneyu(c_eiso, c_bg, alternative="two-sided")
    U_val, _      = mannwhitneyu(c_eiso, c_bg, alternative="two-sided")
    n1, n2        = len(c_eiso), len(c_bg)
    r_rb_eiso     = 2 * U_val / (n1 * n2) - 1

    eiso_dir = "higher" if c_eiso.mean() > c_bg.mean() else "lower"
    alt_one  = "greater" if eiso_dir == "higher" else "less"
    _, p_eiso = mannwhitneyu(c_eiso, c_bg, alternative=alt_one)

    print(f"\n  Eisosome c-value shift: {eiso_dir} than background")
    print(f"    two-sided p = {p_two_eiso:.3e}   one-sided p ({alt_one}) = {p_eiso:.3e}")
    print(f"    rank biserial r = {r_rb_eiso:+.4f}")
    print(f"    eisosome mean c = {c_eiso.mean():.4f}  (n={n1})")
    print(f"    background mean c = {c_bg.mean():.4f}  (n={n2})")

    # IAA comparison
    if HAS_IAA and iaa_series is not None:
        iaa_eiso = iaa_series[iaa_series.index.isin(eisosome_genes)].values
        iaa_bg   = iaa_series[~iaa_series.index.isin(eisosome_genes)].values
        if len(iaa_eiso) >= 3:
            iaa_dir = "greater" if iaa_eiso.mean() > iaa_bg.mean() else "less"
            _, p_iaa_eiso = mannwhitneyu(iaa_eiso, iaa_bg, alternative=iaa_dir)
            print(f"\n  Eisosome IAA sensitivity: "
                  f"{'higher' if iaa_dir=='greater' else 'lower'} than bg")
            print(f"    one-sided p ({iaa_dir}) = {p_iaa_eiso:.3e}")
            print(f"    eisosome mean IAA = {iaa_eiso.mean():.4f}  (n={len(iaa_eiso)})")
            print(f"    background mean IAA = {iaa_bg.mean():.4f}")

    # Boxplot
    try:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.patch.set_facecolor("white")
        bx = ax.boxplot(
            [c_bg, c_eiso],
            labels=["Background", f"Eisosome\n(n={len(c_eiso)})"],
            patch_artist=True, notch=False,
            medianprops=dict(color="black", lw=2),
            widths=0.45,
        )
        for patch, col in zip(bx["boxes"], ["#b0bec5", "#e67e22"]):
            patch.set_facecolor(col); patch.set_alpha(0.7)
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(c_eiso))
        ax.scatter(2 + jitter, c_eiso, color="#c0392b", s=28, alpha=0.7, zorder=4)
        ax.set_ylabel("cross_weight c", fontsize=10)
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(0, y_max)
        ax.set_title(
            f"Eisosome genes have {eiso_dir} cross_weight c than background\n"
            f"rank biserial r={r_rb_eiso:+.3f}  "
            f"MWU one-sided p={p_eiso:.2e}  (n={len(c_eiso)})",
            fontsize=10.5, fontweight="bold",
        )
        ax.grid(axis="y", color="#ddd", lw=0.6)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        plt.tight_layout()
        out = os.path.join(SUPP_DIR, "GO_eisosome_c_distribution.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  ✓ eisosome c distribution → {out}")
    except Exception as e:
        print(f"  ✗ boxplot failed: {e}")

    # Gene list CSV
    rows = []
    for gene in sorted(eisosome_genes):
        c_val   = float(c_series.get(gene, np.nan))
        iaa_val = (float(iaa_series.get(gene, np.nan))
                   if iaa_series is not None else np.nan)
        rows.append(dict(gene=gene, cross_weight_c=c_val, iaa_mean_sens=iaa_val))
    eiso_df = pd.DataFrame(rows).sort_values("cross_weight_c")
    eiso_csv = os.path.join(SUPP_DIR, "GO_eisosome_genes.csv")
    eiso_df.to_csv(eiso_csv, index=False)
    print(f"  ✓ eisosome gene list → {eiso_csv}")
    print(f"    c range: [{eiso_df['cross_weight_c'].min():.4f}, "
          f"{eiso_df['cross_weight_c'].max():.4f}]  "
          f"mean={eiso_df['cross_weight_c'].mean():.4f}")
    print(f"    genes: {', '.join(eiso_df['gene'].tolist())}")


# ══════════════════════════════════════════════════════════════════════
#  11. BOOTSTRAP STABILITY  (validates top GO terms vs. c-value noise)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("Section 11: Bootstrap stability (n=100 perturbations)")
print("═" * 72)
print("""
  Adds 5%-magnitude Gaussian noise to c values and re-runs Fisher enrichment;
  tallies the fraction of runs in which each top GO term reappears.
  Recommendation: only cite terms with stability ≥ 50% in the manuscript.
""")


def bootstrap_go_stability(
    c_series_in: pd.Series, top_n: int = 10, n_boot: int = 100, pct: int = 25,
) -> tuple[dict[str, float], dict[str, float]]:
    from collections import Counter
    counter_h: Counter = Counter()
    counter_l: Counter = Counter()
    rng = np.random.default_rng(0)
    sigma = c_series_in.std() * 0.05

    for _ in range(n_boot):
        noise  = rng.normal(0, sigma, len(c_series_in))
        c_pert = pd.Series(c_series_in.values + noise, index=c_series_in.index)
        c_rank = c_pert.sort_values(ascending=False)
        k_b    = max(int(len(c_rank) * pct / 100), 15)
        high_b = c_rank.index[:k_b].tolist()
        low_b  = c_rank.index[-k_b:].tolist()

        for gene_list, counter in [(high_b, counter_h), (low_b, counter_l)]:
            res = run_fisher(gene_list)
            if res.empty:
                continue
            sig = res[res["p_fdr"] < FDR_CUTOFF]["term"].tolist()
            if not sig:
                sig = res.nsmallest(5, "p_raw")["term"].tolist()
            counter.update(sig)

    stab_h = {t: cnt / n_boot for t, cnt in counter_h.most_common(top_n)}
    stab_l = {t: cnt / n_boot for t, cnt in counter_l.most_common(top_n)}
    return stab_h, stab_l


if USE_SUMMARY and not c_series.empty:
    print("  Running bootstrap (100 iterations; may take a few minutes)…")
    stab_h, stab_l = bootstrap_go_stability(c_series, top_n=10, n_boot=100, pct=25)

    print("\n  [Stability] High-c top terms:")
    for t, s in stab_h.items():
        flag = "✓ robust" if s >= 0.5 else "⚠ unstable"
        print(f"    {s:5.0%}  {flag}  {t}")
    print("\n  [Stability] Low-c top terms:")
    for t, s in stab_l.items():
        flag = "✓ robust" if s >= 0.5 else "⚠ unstable"
        print(f"    {s:5.0%}  {flag}  {t}")

    rows = (
        [dict(group="high_c", term=t, stability=s) for t, s in stab_h.items()] +
        [dict(group="low_c",  term=t, stability=s) for t, s in stab_l.items()]
    )
    stab_csv = os.path.join(PC["output_dir"], "GO_bootstrap_stability.csv")
    pd.DataFrame(rows).to_csv(stab_csv, index=False)
    print(f"\n  ✓ stability table → {stab_csv}")
    print("  → recommend only citing terms with stability ≥ 50% in the paper.")
else:
    print("  (skipped: no c_series)")


# ══════════════════════════════════════════════════════════════════════
#  12. FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("All sections complete. Output files:")
print("═" * 72)
for root_dir in [PC["output_dir"], SUPP_DIR]:
    if not os.path.isdir(root_dir):
        continue
    tag = "(supplementary)" if root_dir == SUPP_DIR else "(main)"
    for fname in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, fname)
        if os.path.isfile(path):
            size = os.path.getsize(path)
            print(f"  {size:>10,} B   {path}  {tag}")
print("\nDone.")
