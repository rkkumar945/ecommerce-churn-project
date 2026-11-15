# Mock review (written Day 79, rewritten 2026-11-13 for the final project)

## 1. Problem
"I built a customer churn-ranking system to help a retention team focus its limited outreach on customers most likely not to order again."

Sources: README.md, Business Problem and Start Here.

## 2. Data
"I worked with Online Retail II: 1,067,371 transactions and 5,942 identified customers. Rows without a Customer ID made up 22.8% of the data, and 19,494 cancellation rows were removed. The modelling table has 5,283 customers, split into 4,226 for training and 1,057 held out; the held-out set contains 598 non-returners, a 56.6% base rate."

Sources: README.md, Dataset and Results; reports/model_comparison_report.md.

## 3. Target design
"My first label used Recency over the whole observation period, while Recency was also a feature. That let the model learn the rule used to create its label. I redesigned the target to use transactions before a cutoff for features and whether the customer ordered in the following 90 days for the outcome. The EDA notebooks retain the whole-period descriptive label; the ML target is forward-looking, so the two labels differ by design."

Sources: README.md, Target definition; docs/interview_prep.md, Question bank.

## 4. Analysis
"For the descriptive EDA label, customers with one order had 77.5% churn, compared with 29.4% for customers with four or more orders. This is a descriptive finding from the whole-period Recency label, which is partly circular and is not evidence of predictive power."

Sources: README.md, Key findings.

## 5. Modeling
"I compared Logistic Regression, Decision Tree, Random Forest and XGBoost using nested cross-validation. Their nested ROC-AUCs were 0.7963, 0.7775, 0.8022 and 0.8028 respectively. I selected XGBoost with learning_rate 0.03, max_depth 2, min_child_weight 3 and n_estimators 150."

Sources: reports/nested_cv_day84.csv; reports/final_model_card.json.

## 6. Result + honesty
"The held-out ROC-AUC was 0.818 (95% bootstrap CI 0.791 to 0.843), and average precision (AP) was 0.838. At the top 10%, precision was 0.925, lift was 1.634x, and the group reached 16.4% of non-returners. The 0.07 alert threshold flags almost everyone. The evaluation uses one cutoff and a random customer split, so there is no out-of-time test."

Sources: reports/model_comparison_report.md; README.md, Results and Limitations.

## 7. Next
"My next planned work is an AI agent and API wrapper around the model, followed by Docker and cloud container deployment."

Sources: README.md, Roadmap.
