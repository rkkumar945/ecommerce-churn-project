# Final project review

Review dated 2026-11-13. For the canonical broader assessment, see [docs/final_review.md](../docs/final_review.md).

## Must-fix
- The Day 84 model-choice record had no rationale. A retrospective reason based only on the Day 84 CV artifacts has now been added, and the corresponding known-issues bullet is marked resolved retrospectively. Status: Resolved on 2026-11-13. References: reports/final_model_choice_day84.md; reports/nested_cv_day84.csv; docs/known_issues.md.
- The Day 79 mock review and this Day 98 review contained unfinished template text. Both are now complete, dated, and tied to project evidence. Status: Resolved on 2026-11-13. References: reports/mock_review_day79.md; reports/final_review_day98.md.
- The Day 100 master test report was empty. It now records executed checks and clearly identifies comparisons that cannot be confirmed from the available reports. Status: Resolved on 2026-11-13. Reference: reports/master_test_day100.md.

Sources: docs/known_issues.md; reports/final_model_choice_day84.md; reports/mock_review_day79.md; reports/master_test_day100.md.

## Should-fix
- The alert threshold of 0.07 yields 0.998 recall and 0.586 precision, with 422 false positives and one false negative. The report should be used as a ranking and the alert should not be treated as a selective contact list. Status: Open. References: reports/model_comparison_report.md; README.md, Results and Limitations; docs/model_risk_note.md.
- Validate the ranking on later time periods and measure campaign impact with an appropriate control group before claiming operational or retention value. Status: Open. References: README.md, Limitations; docs/final_review.md; docs/model_risk_note.md.
- The nested-CV scores for XGBoost, Logistic Regression and Random Forest are close relative to their reported standard deviations; the model choice is not statistically decisive. Status: Open. References: reports/nested_cv_day84.csv; README.md, Key findings and Limitations.
- The baseline suite completed with 112 passed and 0 skipped. The baseline repository audit exited successfully but reported seven warnings and four informational findings, including notebook execution-order history, missing root requirements for pillow and tabulate, and model-file placement. These were not changed in this documentation-only pass. Status: Open. References: scripts/repo_audit.py; reports/master_test_day100.md.

Sources: README.md, Results, Key findings and Limitations; reports/model_comparison_report.md; reports/nested_cv_day84.csv; docs/final_review.md; docs/model_risk_note.md.

## Documented limitations
- The data covers one UK retailer and a historical period; transfer to other businesses or later periods is unverified. Status: Open. Reference: README.md, Limitations.
- The evaluation uses one cutoff and a random customer split; there is no out-of-time test. Status: Open. References: README.md, Limitations; docs/known_issues.md.
- Retention impact is not measured because there is no control group. Status: Open. References: README.md, Limitations; docs/model_risk_note.md.
- Class-weighted scores are for ranking and are not calibrated probabilities. Status: Open. References: README.md, Limitations; reports/model_comparison_report.md.
- Drift monitoring is based on historical snapshots, and real performance labels would arrive after the 90-day outcome window. Status: Open. References: README.md, Limitations; docs/model_risk_note.md.
- The app uses free hosting that sleeps when idle and has no uptime guarantee. Status: Open. Reference: README.md, Limitations.

Sources: README.md, Limitations; reports/model_comparison_report.md; docs/known_issues.md; docs/model_risk_note.md.

## Open questions/decisions
- What contact budget and outreach policy should determine whether the team uses a ranking cutoff or an alert threshold? Status: Open. References: README.md, Results and Limitations; docs/model_risk_note.md.
- Which later time window and customer population will provide an out-of-time validation? Status: Open. References: README.md, Limitations; docs/final_review.md.
- What controlled campaign design and outcome measure will establish incremental retention impact? Status: Open. References: README.md, Limitations; docs/model_risk_note.md.

Sources: README.md, Limitations; docs/final_review.md; docs/model_risk_note.md.