import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

# -----------------------------
# Load trained model and preprocessor
# -----------------------------
MODEL_PATH = "models/laptop_price_model_tuned.pkl"
DATA_PATH = "data/processed/amazon_laptops_features_enhanced_clean.csv"
FIGURES_PATH = "reports/figures/"

st.set_page_config(page_title="💻 PriceSense Dashboard", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    return model

@st.cache_data
def load_reference_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()

model = load_model()
reference_data = load_reference_data()

st.title("💻 PriceSense – Laptop Price Prediction Dashboard")
st.markdown("An AI-powered system for **price prediction, explainability, and monitoring**.")

# -----------------------------
# Sidebar for user input
# -----------------------------
st.sidebar.header("Upload Laptop Specs")

uploaded_file = st.sidebar.file_uploader("Upload a CSV with laptop specs", type=["csv"])

if uploaded_file:
    new_data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Laptop Data", new_data.head())

    # Ensure features align with training data
    try:
        prediction = model.predict(new_data)
        st.success(f"💰 Predicted Laptop Price: **${prediction[0]:,.2f}**")
    except Exception as e:
        st.error(f"Error in prediction: {e}")

# -----------------------------
# Explainability Section
# -----------------------------
st.header("🔎 Explainability with SHAP")

col1, col2 = st.columns(2)

with col1:
    shap_summary = os.path.join(FIGURES_PATH, "shap_summary.png")
    if os.path.exists(shap_summary):
        st.image(shap_summary, caption="SHAP Summary Plot", use_container_width=True)
    else:
        st.warning("SHAP summary plot not found. Run explain.py first.")

with col2:
    shap_bar = os.path.join(FIGURES_PATH, "shap_bar.png")
    if os.path.exists(shap_bar):
        st.image(shap_bar, caption="SHAP Bar Plot", use_container_width=True)
    else:
        st.warning("SHAP bar plot not found. Run explain.py first.")

# -----------------------------
# Drift Monitoring Section
# -----------------------------
st.header("📊 Drift Monitoring")

drift_report_path = "reports/drift_report.csv"
if os.path.exists(drift_report_path):
    drift_df = pd.read_csv(drift_report_path)
    st.dataframe(drift_df)

    # Highlight drifted features
    drifted = drift_df[drift_df["drift_detected"] == True]
    if not drifted.empty:
        st.error("⚠️ Drift detected in some features!")
        st.write(drifted)
    else:
        st.success("✅ No significant drift detected.")
else:
    st.warning("Drift report not found. Run drift_monitor.py first.")
