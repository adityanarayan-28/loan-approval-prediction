"""Streamlit user interface for Loan Approval Prediction."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="wide")
ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "loan_model.pkl"
META_FILE = ROOT / "models" / "model_metadata.json"

@st.cache_resource
def load_model(path):
    return joblib.load(path)

st.title("🏦 Loan Approval Prediction")
st.subheader("Machine Learning Classification Project")
st.write("Enter applicant information to see the outcome predicted by the trained classification model.")
st.warning("Academic demonstration only. This is not a real bank decision and does not guarantee loan approval.")

if not MODEL_FILE.exists():
    st.error("Trained model not found. Run `python train_model.py` first.")
    st.stop()

try:
    model = load_model(str(MODEL_FILE))
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

metadata = {}
if META_FILE.exists():
    try:
        metadata = json.loads(META_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass

with st.sidebar:
    st.header("Project information")
    st.write("**Task:** Supervised binary classification")
    st.write("**Algorithms compared:** Logistic Regression, Decision Tree, Random Forest")
    st.write(f"**Selected model:** {metadata.get('best_model', 'Random Forest')}")
    st.caption(metadata.get("dataset_note", "See README for dataset details."))

st.markdown("### Applicant Details")
with st.form("applicant_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    with c2:
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        applicant_income = st.number_input("Applicant Income", min_value=0, value=45000, step=1000)
        coapplicant_income = st.number_input("Co-applicant Income", min_value=0, value=10000, step=1000)
        loan_amount = st.number_input("Loan Amount", min_value=1, value=150, step=10)
    with c3:
        loan_term = st.selectbox("Loan Term (months)", [120, 180, 240, 300, 360, 480], index=4)
        credit_history = st.selectbox("Credit History", [1.0, 0.0], format_func=lambda x: "Available / positive (1)" if x == 1.0 else "Unavailable / negative (0)")
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    submitted = st.form_submit_button("Predict Loan Status", type="primary", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
    }])
    try:
        prediction = str(model.predict(row)[0]).strip().upper()
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info("The app inputs must match the features used by train_model.py.")
        st.stop()

    st.markdown("### Prediction Result")
    if prediction == "Y":
        st.success("Prediction: Likely Approved")
    elif prediction == "N":
        st.error("Prediction: Likely Not Approved")
    else:
        st.info(f"Model output: {prediction}")

    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(row)[0]
            classes = model.classes_
            st.markdown("**Model-estimated class probabilities**")
            st.dataframe(pd.DataFrame({
                "Class": [str(c) for c in classes],
                "Estimated probability": [f"{p:.1%}" for p in probs],
            }), hide_index=True, use_container_width=True)
            st.caption("These estimates are not guaranteed real-world probabilities.")
        except Exception:
            pass

    with st.expander("View input values"):
        st.dataframe(row, hide_index=True, use_container_width=True)

st.divider()
st.caption("The included dataset is synthetic and intended for academic demonstration. Do not use this model for real lending decisions.")
