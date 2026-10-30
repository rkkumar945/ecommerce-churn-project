# Model Comparison Report (Day 85)

**Final model (pre-registered on Day 84, chosen on CV only):** XGBoost
**Threshold:** 0.07, chosen from out-of-fold predictions using assumed costs (FN=50, FP=5, TP benefit=40).
**Test set:** n=1057, churners=598. Final model scored once.

## Test performance (all models, reference only; selection was not based on this table)
| model               |   test_auc |   auc_lo |   auc_hi |   test_ap |   ap_lo |   ap_hi |
|:--------------------|-----------:|---------:|---------:|----------:|--------:|--------:|
| LogReg              |      0.811 |    0.783 |    0.836 |     0.837 |   0.806 |   0.866 |
| Tree                |      0.802 |    0.774 |    0.829 |     0.801 |   0.766 |   0.835 |
| RandomForest        |      0.818 |    0.792 |    0.843 |     0.839 |   0.806 |   0.869 |
| XGBoost             |      0.818 |    0.791 |    0.843 |     0.838 |   0.804 |   0.868 |
| Baseline (prior)    |      0.5   |    0.5   |    0.5   |     0.566 |   0.535 |   0.597 |
| LogReg v1 (Month 4) |      0.809 |    0.782 |    0.835 |     0.833 |   0.801 |   0.863 |

## Final model at chosen threshold
|           |   value |    lo |    hi |
|:----------|--------:|------:|------:|
| precision |   0.586 | 0.555 | 0.616 |
| recall    |   0.998 | 0.995 | 1     |
| F1        |   0.738 | 0.713 | 0.762 |

Confusion matrix: TN=37, FP=422, FN=1, TP=597

## Caveats
- Test set has few churners, so intervals are wide. Differences inside the intervals are not real differences.
- Probabilities are inflated by class weighting; the model is meant for ranking, not for calibrated risk percentages.
- Costs are assumptions, not measured business numbers.
