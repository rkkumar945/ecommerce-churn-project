"Month 1 Complete ✅ — SQL Foundations: SELECT/WHERE/GROUP BY, all JOIN types, Window Functions (ROW_NUMBER, RANK, DENSE_RANK, NTILE, LAG/LEAD), CTEs (including chained CTEs for RFM-style analysis), missing value handling, and query optimization with indexing."



### 🧹 Data Cleaning & Preprocessing Decisions

To ensure accurate downstream analysis (such as RFM modeling, Customer Lifetime Value, and churn prediction), the dataset underwent rigorous data cleaning. Below are the key decisions and steps taken during the cleaning pipeline:

1. **Duplicate Row Removal (`drop_duplicates`)**
   * Identified and removed exact duplicate transaction rows to prevent double-counting of sales volume and metrics.

2. **Handling Order Cancellations (`Invoice` starting with 'C')**
   * **Decision:** Filtered out and removed all cancellation records (invoices starting with `C`, containing negative quantities).
   * **Rationale:** Leaving cancellations in the dataset skews monetary calculations, Net Revenue, and RFM scores by introducing negative sales values that do not represent active consumer purchasing behavior.

3. **Excluding Non-Product StockCodes**
   * **Decision:** Identified and optionally separated non-product entries such as `POST` (postage), `DOT`, `M`, `ADJUST`, `GIFT`, `AMAZONFEE`, and `BANK CHARGES`.
   * **Rationale:** These represent operational fees, postage costs, or manual adjustments rather than actual physical product sales. Two cleaned versions were saved:
     * `step3_all_valid_transactions.csv` (All valid customer purchases excluding cancellations).
     * `step3_products_only.csv` (Excluding both cancellations and non-product operational codes for pure product-level insights).