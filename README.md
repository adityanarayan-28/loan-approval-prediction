# Loan Approval Prediction using Machine Learning

## Project overview
This is a complete, runnable academic project demonstrating **supervised binary classification**. It compares Logistic Regression, Decision Tree, and Random Forest, evaluates models on a held-out test split, saves the best pipeline, produces charts/reports, and provides a Streamlit interface.

## Important dataset note
`data/loan_data.csv` is a **synthetic demonstration dataset generated for this project**. It is not real bank/customer data and is not suitable for real lending decisions. Metrics from this dataset only demonstrate the workflow. For a final research-quality evaluation, replace it with a documented public dataset and update the report.

## Folder structure
```text
Loan_Approval_Prediction_Project_Fresh/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
├── data/
│   └── loan_data.csv
├── models/
│   └── (generated when training runs)
├── notebooks/
│   └── Loan_Approval_Prediction.ipynb
└── reports/
    └── (generated when training runs)
```

## Run on Windows using VS Code
1. Extract the ZIP file.
2. In VS Code, choose **File → Open Folder** and select `Loan_Approval_Prediction_Project_Fresh`.
3. Choose **Terminal → New Terminal**.
4. (Recommended) Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   If activation is blocked in this terminal only:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```
5. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
6. Train/evaluate the model and create the saved model, charts, and reports:
   ```powershell
   python train_model.py
   ```
7. Launch the app:
   ```powershell
   python -m streamlit run app.py
   ```
8. Open the local URL Streamlit prints (usually `http://localhost:8501`).

If `python` is not recognized, install Python 3 from the official Python website, enable **Add Python to PATH** during installation, restart VS Code, then retry.

## Dataset columns
- Inputs: Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area
- Target: Loan_Status (`Y` = approved label, `N` = not-approved label)

## Methodology
1. Inspect dataset and missing values.
2. Plot target distribution and applicant-income distribution.
3. Split data into training (80%) and test (20%) sets using a stratified split.
4. Impute numeric missing values with median and categorical missing values with most frequent value.
5. Scale numeric values and one-hot encode categorical values inside a scikit-learn pipeline.
6. Train Logistic Regression, Decision Tree, and Random Forest.
7. Compare accuracy, precision, recall, and F1-score for the approved class.
8. Save the best model pipeline and metadata.
9. Use Streamlit to collect input and display a prediction.

## Files produced by training
- `models/loan_model.pkl`
- `models/model_metadata.json`
- `reports/model_comparison.csv`
- `reports/model_comparison.txt`
- `reports/classification_report.txt`
- `reports/loan_status_distribution.png`
- `reports/applicant_income_distribution.png`
- `reports/model_comparison.png`

## Academic integrity and limitations
The model is a classroom demonstration. Synthetic data cannot establish real-world accuracy, fairness, or lending performance. Do not describe synthetic results as results on real banking data. A real lending system requires high-quality verified data, fairness analysis, security, explainability, human review, and regulatory compliance.
