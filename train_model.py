"""Train and evaluate classification models for Loan Approval Prediction."""
from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "loan_data.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

TARGET = "Loan_Status"
NUMERIC = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
CATEGORICAL = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
FEATURES = CATEGORICAL + NUMERIC

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    missing_targets = df[TARGET].isna().sum()
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(str).str.strip()
    df = df[df[TARGET].isin(["Y", "N"])].copy()

    print("\\nDATASET OVERVIEW")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    print("\\nMissing values before preprocessing:")
    print(df[FEATURES].isna().sum().to_string())
    print("\\nTarget distribution:")
    print(df[TARGET].value_counts().to_string())

    # Exploratory data visualizations.
    sns.set_theme()
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET)
    plt.title("Loan Status Distribution (Synthetic Demo Data)")
    plt.xlabel("Loan Status (N = Not Approved, Y = Approved)")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "loan_status_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x="ApplicantIncome", hue=TARGET, bins=25, element="step")
    plt.title("Applicant Income by Loan Status (Synthetic Demo Data)")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "applicant_income_distribution.png", dpi=150)
    plt.close()

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("categorical", categorical_pipeline, CATEGORICAL),
    ])

    algorithms = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, random_state=42, class_weight="balanced",
            min_samples_leaf=2
        ),
    }
    results = []
    fitted = {}
    for name, estimator in algorithms.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision (approved)": precision_score(y_test, predictions, pos_label="Y", zero_division=0),
            "Recall (approved)": recall_score(y_test, predictions, pos_label="Y", zero_division=0),
            "F1 (approved)": f1_score(y_test, predictions, pos_label="Y", zero_division=0),
        })
        fitted[name] = pipeline

    comparison = pd.DataFrame(results).sort_values("F1 (approved)", ascending=False)
    best_name = str(comparison.iloc[0]["Model"])
    best_model = fitted[best_name]
    best_predictions = best_model.predict(X_test)

    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    (REPORT_DIR / "classification_report.txt").write_text(
        f"Selected model: {best_name}\\n\\n"
        + classification_report(y_test, best_predictions, zero_division=0)
        + "\\nConfusion matrix (rows=true [N,Y], columns=predicted [N,Y]):\\n"
        + str(confusion_matrix(y_test, best_predictions, labels=["N", "Y"])),
        encoding="utf-8"
    )
    with open(REPORT_DIR / "model_comparison.txt", "w", encoding="utf-8") as f:
        f.write("MODEL COMPARISON (held-out test set)\\n")
        f.write(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        f.write("\\n\\nNOTE: This is synthetic demonstration data, not real-world lending performance.\\n")

    plt.figure(figsize=(8, 4))
    sns.barplot(data=comparison, x="Model", y="F1 (approved)")
    plt.ylim(0, 1)
    plt.title("Model Comparison: F1-score for Approved Class")
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "model_comparison.png", dpi=150)
    plt.close()

    joblib.dump(best_model, MODEL_DIR / "loan_model.pkl")
    metadata = {
        "best_model": best_name,
        "features": FEATURES,
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "target": TARGET,
        "labels": {"Y": "Likely approved", "N": "Likely not approved"},
        "dataset_note": "Synthetic demonstration data generated for academic practice; not real banking data.",
        "test_rows": int(len(X_test)),
        "train_rows": int(len(X_train)),
        "metrics": comparison.to_dict(orient="records"),
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\\nMODEL COMPARISON")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\\nSelected model: {best_name}")
    print(f"Saved model: {MODEL_DIR / 'loan_model.pkl'}")
    print(f"Saved charts and reports in: {REPORT_DIR}")
    print("\\nReminder: these results describe synthetic demo data only.")

if __name__ == "__main__":
    main()
