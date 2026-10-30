# Pre-registered final model (Day 84)
Decided on CV only. Test set NOT used for any decision (it will be scored once, on Day 85).

- Model: XGBoost
- Params: {"learning_rate": 0.03, "max_depth": 2, "min_child_weight": 3, "n_estimators": 150}
- Nested CV AUC: 0.8028 +/- 0.0158
- Reason: [apni 2-3 line ki wajah likho: score, simplicity, stability]
- Metric for threshold later: chosen from CV out-of-fold predictions, not from test.
