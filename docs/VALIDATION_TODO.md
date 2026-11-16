# Validation TODO

This project is a historical churn-ranking prototype. The following validation items remain open and should be completed before any production or campaign claim is made.

1. Raw transaction data and intermediate CSVs are missing from the repo, so the full chain from cleaning to features to labels cannot be independently reconciled. Recreate a versioned acquisition route for the raw data and keep the exact file hashes in a reproducible manifest.
2. There is no out-of-time validation. Freeze the current model and protocol, then collect a new untouched future cohort with mature 90-day outcomes, run rolling-origin evaluation, and document the intended customer group (existing customers vs. new customers) with repeat-customer handling.
3. Calibration must be assessed on a fresh held-out period instead of the current test split. A calibrator should be trained only with training or cross-validated out-of-fold predictions before any deployment claim is made.
4. Business impact requires a randomized campaign holdout with realistic contact costs, operating margins, and confidence intervals; the current utility assumptions are not measured retention uplift.
5. The git history and pre-registration chronology cannot be authenticated from a ZIP bundle alone. Keep the real git history in the repository and document the exact version trail used for the final model.
6. The live deployed website was not audited for security and deployment controls; infrastructure and web review remain pending.
