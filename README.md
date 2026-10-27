# E-Commerce Customer Churn Prediction

> End-to-end data science project: SQL → Python/Pandas → Statistics & EDA → Machine Learning,
> predicting which customers will stop buying in the next 90 days.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![SQL](https://img.shields.io/badge/SQL-SQLite-lightgrey)
![Status](https://img.shields.io/badge/status-completed-green)

## Table of Contents
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Approach](#approach)
- [Key Findings](#key-findings)
- [Model Results](#model-results)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Author](#author)

## Business Problem
Acquiring a new customer costs far more than retaining an existing one. This project
identifies customers likely to churn **before** they leave, so a retention team can
target a limited campaign budget at the right people.

**Goal:** rank customers by churn risk and flag the top at-risk group for outreach.

## Dataset
**Online Retail II** (UCI / Kaggle): ~1.07M transactions from a UK-based online retailer,
Dec 2009 – Dec 2011, 8 columns (Invoice, StockCode, Description, Quantity, InvoiceDate,
Price, Customer ID, Country).
The raw data is not stored in this repo, see [`data/README.md`](data/README.md) for download steps.

## Approach

### 1. SQL (Month 1)
Loaded the data into SQLite and analysed it with JOINs, window functions
(`ROW_NUMBER`, `RANK`, `NTILE`, `LAG`), chained CTEs for an RFM-style summary, and indexing
with `EXPLAIN QUERY PLAN`. Key queries are in [`src/sql/`](src/sql/).

### 2. Data Cleaning (Month 2)
| Step | Decision |
|---|---|
| Missing `Customer ID` (~22.8%) | Dropped: guest checkouts can't be tracked per customer |
| Duplicates | Removed |
| Cancellations (Invoice starts with `C`) | Removed: they distort monetary value |
| Non-product codes (POST, DOT, M, ADJUST, ...) | Identified and excluded |
| Missing `Description` | Filled with `'Unknown Product'` |

## Key Findings
- **Total Customers:** 5,283
- **Churn rate:** 56.6% of customers (imbalance ratio 1.3:1)
- **Spend is heavily right-skewed:** top customers kept as a VIP segment rather than removed.
- **Geography:** UK vs Rest of World analysis completed using chi-square tests.

![Churn rate dashboard](reports/churn_rate_dashboard.png)

## Model Results
5-fold CV on the training set, followed by a single evaluation on the held-out test set.

| Model | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.654 | 0.712 | 0.682 | 0.745 |
| Decision Tree (tuned) | 0.631 | 0.689 | 0.658 | 0.710 |
| Baseline (majority class) | 0 | 0 | 0 | 0.50 |

![ROC curve](reports/roc_curve_cv.png)

> Accuracy is deliberately not the headline metric: with imbalanced classes a model
> that always predicts the majority class gives a false sense of high performance.

## Repository Structure
```text
├── data/                 # not tracked, see data/README.md
├── notebooks/            # month-wise subfolders (SQL, Python, Stats, ML)
├── src/                  # reusable modules (cleaning, features, sql)
├── reports/              # charts and comparison tables
└── requirements.txt

## Limitations
- Data is from 2009–2011 from a single UK retailer, so patterns may not transfer to other businesses.
- The 90-day churn window is a business assumption.
- Predicted probabilities are not calibrated; the model is meant for **ranking** customers.

## Roadmap
- [x] SQL, Python/Pandas, statistics, EDA
- [x] Leakage-free ML with CV, imbalance handling and evaluation
- [ ] Random Forest / XGBoost, hyperparameter tuning
- [ ] SHAP explainability
- [ ] Streamlit web app + deployment

## Author
**Rajat Kumar**