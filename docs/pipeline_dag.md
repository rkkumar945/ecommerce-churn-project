```mermaid
graph TD
    A[(data/raw/online_retail_II.csv)] --> B[clean_transactions<br/>src/cleaning/clean_data.py]
    B --> C[(step4_datatypes_fixed.csv)]
    C --> D[build_customer_dataset<br/>src/features/build_features.py]
    D --> E[(customer_ml_dataset_v2.csv)]
    E --> F[stratified split<br/>NOTEBOOK ONLY]
    F --> G[X_train_v2, X_test_v2,<br/>y_train, y_test]
    G --> H[grid search + CV + MLflow<br/>src/models, src/tracking]
    H --> I[final_model_choice.md<br/>MANUAL GATE:<br/>pre-registration]
    I --> J[fit final, threshold, test<br/>evaluation ek baar<br/>NOTEBOOK ONLY]
    J --> K[(models/final_model.joblib +<br/>final_model_card.json)]
    K --> L[export_artifacts<br/>src/app/export_artifacts.py]
    E --> L
    L --> M[(app_artifacts/)]
    M --> N[Streamlit app]



--


---

## 1. Step Table
Pipeline ke har step ka status aur audit records Cell 4 ki table mein summarize kiye gaye hain. Yeh table current execution state aur data pipeline ki integrity track karne mein madad karti hai.

## 2. Pipeline Gaps
Audit ke mutabiq `stratified split` aur `finalize` jaise steps abhi bhi Jupyter notebooks mein perform ho rahe hain, jo script-based automation mein rukaavat hain. Iske alawa, data split karne ka logic notebooks aur `export_artifacts.py` dono jagah duplicate hai, jisse future code sync issues ka risk rehta hai.

## 3. Manual Gates (Non-Automated)
* **Model Selection:** Final model ko approve aur choose karna completely manual process rahega (human decision via pre-registration).
* **Test Set Evaluation:** Scheduled retraining ya automated triggers par test set evaluation auto-run nahi hoga, kyunki ek hi test set ko baar-baar evaluate karne se bias badhta hai. Scheduled retrain ke liye naya time-based holdout dataset zaroori hai.
*