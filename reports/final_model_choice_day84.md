# Pre-registered final model (Day 84)
Decided on CV only. The held-out set was not used for any model-selection decision; it was recorded for the final evaluation and reported on Day 85. Earlier exploratory notebooks also looked at the same split, so the final report is intentionally described as the final, fixed evaluation rather than the first time the test set was ever examined.

- Model: XGBoost
- Params: {"learning_rate": 0.03, "max_depth": 2, "min_child_weight": 3, "n_estimators": 150}
- Nested CV AUC: 0.8028 +/- 0.0158
- Reason: XGBoost had the highest nested-CV ROC-AUC among the four families (0.8028 +/- 0.0158), but its estimate was within the reported noise of Logistic Regression (0.7963 +/- 0.0140) and Random Forest (0.8022 +/- 0.0143). The plain-CV grid's best XGBoost result used n_estimators=300, while the pre-registered model uses 150; the rationale for that difference is not recorded.
Note: this reason was added retrospectively on 2026-11-13; the original reason was not recorded on Day 84.
- Metric for threshold later: chosen from CV out-of-fold predictions, not from test.
