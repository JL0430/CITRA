# Table 1. Benchmark comparison: this method vs. 6 baselines × 3 feature sets

| Model | LM-only | +Activity | Full |
|---|---|---|---|
| Ridge | 0.6836 | 0.6930 | 0.8256 |
| Lasso | 0.6824 | 0.6937 | 0.8290 |
| XGBoost | 0.6934 | 0.6998 | 0.8317 |
| LightGBM | 0.6977 | 0.7062 | 0.8307 |
| GatedMLP | 0.6801 | 0.6898 | 0.8185 |
| FT-Transformer | 0.6083 | 0.6456 | 0.7426 |
| **Ours** | 0.6773 | 0.6988 ± 0.0097 | **0.8482 ± 0.0051** |

Note: All values are test Pearson r on the main holdout (chr15/16, n = 1,108). Our method's Full and +Activity columns are 5-seed paired means ± std over seeds [42, 123, 456, 789, 2024]. Baselines use the same seed protocol as released in comparison_matrix_full.csv. For reference, FUN-PROSE (binding mode, steelman upper bound): 0.7897 ± 0.0068; FUN-PROSE (expression mode, faithful to original): 0.5961 ± 0.0188. See Additional file 1: §S1 for FUN-PROSE reproduction details.