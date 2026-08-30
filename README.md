# CITRA: Cis-TRans Attention

Official code for the manuscript:

**Sequence is not enough: gated cross-modal attention integrates trans-regulatory signals for gene expression prediction in *Saccharomyces cerevisiae***

Long Jiang, Xiu-Hong Li, Yi-Xuan Qi, Zheng Zhang, Hui Zhang\*, Li Liu\*

## Overview

CITRA is a dual-branch deep learning model that predicts endogenous gene expression
(log TPM) in *Saccharomyces cerevisiae* by fusing **cis-regulatory** signals
(promoter sequence embeddings + promoter activity) with **trans-regulatory** signals
(genome-wide TF ChEC-seq binding profiles) through scalar-gated residual cross-modal
attention — the sequence branch supplies the query, and a learnable TF encoding
supplies the keys and values.

Key results:

- Test Pearson *r* = **0.848** on the primary chromosome hold-out (chr15/16),
  vs. 0.790 for an enhanced reproduction of the closest published method
  (FUN-PROSE, *p* &lt; 0.0001); pooled *r* ≈ 0.852 across three independent hold-outs.
- Factorial ablation attributes the gain to the learnable TF encoders, not added capacity.
- Without biological priors, the model recovers the MAPK pathway (STE12) and the
  ribosomal protein program (RAP1) as the principal axes of trans-regulation.
- The per-gene fusion gate *c* tracks growth and biosynthesis programs, validated
  against external labels (Brauer et al.).

## Data

This repository contains **code only**. The full processed dataset (~3.4 GB),
including the TF ChEC-seq tensor, expression labels, LM embeddings and promoter
activity features, is archived on Zenodo:

- **DOI**: https://doi.org/10.5281/zenodo.22170912
- **Record**: https://zenodo.org/records/22170913

Download and extract the archive into the project root before running the scripts.

Raw data sources: Mahendrawada et al. (GEO: GSE236948), Vaishnav et al.,
Saccharomyces Genome Database, Brauer et al., Gasch et al.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python new55.py
```
