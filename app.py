import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "models/laptop_price_model_tuned.pkl"
DATA_PATH = "data/processed/amazon_laptops_features_enhanced_clean.csv"
FIGURES_PATH = "reports/figures/"
DRIFT_REPORT_PATH = "reports/drift_report.csv"

st.set_page_config(page_title="PriceSense Dashboard", layout="wide")

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        st.error(f"Model not found at {MODEL_PATH}. Please train the model first.")
        return None

# -----------------------------
# Load reference dataset
# -----------------------------
@st.cache_data
def load_reference_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        return pd.DataFrame()

model = load_model()
reference_data = load_reference_data()

st.title("💻 PriceSense – Laptop Price Prediction Dashboard")
st.markdown("AI-powered system for **laptop price prediction, explainability, and monitoring**.")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Laptop Specs Input")
uploaded_file = st.sidebar.file_uploader("Upload a CSV with laptop specs", type=["csv"])

# Manual input form
with st.sidebar.form("manual_input"):
    st.markdown("Or enter specs manually:")
    ram = st.number_input("RAM (GB)", min_value=2, max_value=128, value=16)
    storage = st.number_input("Storage (GB)", min_value=128, max_value=5000, value=512)
    weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=1.5)
    processor_speed = st.number_input("Processor Speed (GHz)", min_value=1.0, max_value=6.0, value=3.2)
    gpu = st.selectbox("GPU Brand", ["NVIDIA", "AMD", "Intel"])
    brand = st.selectbox("Brand", ["Dell", "HP", "Lenovo", "Asus", "Apple", "Acer", "MSI"])
    submitted = st.form_submit_button("Predict Price")

    manual_data = None
    if submitted:
        manual_data = pd.DataFrame([{
            "ram_gb": ram,
            "storage_gb": storage,
            "weight": weight,
            "processor_speed": processor_speed,
            "gpu": gpu,
            "brand": brand
        }])
        st.write("### Manual Input Data", manual_data)

# -----------------------------
# Prediction
# -----------------------------
if model:
    if uploaded_file:
        new_data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Laptop Data", new_data.head())

        try:
            prediction = model.predict(new_data)
            st.success(f"💰 Predicted Laptop Price: **{prediction[0]:,.2f}**")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

    elif manual_data is not None:
        try:
            prediction = model.predict(manual_data)
            st.success(f"💰 Predicted Laptop Price: **{prediction[0]:,.2f}**")

            # -----------------------------
            # SHAP Explainability
            # -----------------------------
            if not reference_data.empty:
                explainer = shap.Explainer(model, reference_data.sample(min(100, len(reference_data)), random_state=42))
                shap_values = explainer(manual_data)

                st.subheader("🔎 SHAP Explainability for Your Input")
                fig, ax = plt.subplots(figsize=(8, 6))
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig)
            else:
                st.warning("Reference data not found – cannot compute SHAP explanations.")

        except Exception as e:
            st.error(f"Error in manual input prediction: {e}")

# -----------------------------
# Global Explainability
# -----------------------------
st.header("📊 Global Explainability with SHAP")
col1, col2 = st.columns(2)

with col1:
    shap_summary = os.path.join(FIGURES_PATH, "shap_summary.png")
    if os.path.exists(shap_summary):
        st.image(shap_summary, caption="SHAP Summary Plot", use_column_width=True)
    else:
        st.warning("SHAP summary plot not found. Run explain.py first.")

with col2:
    shap_bar = os.path.join(FIGURES_PATH, "shap_bar.png")
    if os.path.exists(shap_bar):
        st.image(shap_bar, caption="SHAP Bar Plot", use_column_width=True)
    else:
        st.warning("SHAP bar plot not found. Run explain.py first.")

# -----------------------------
# Drift Monitoring
# -----------------------------
st.header("📉 Drift Monitoring")
if os.path.exists(DRIFT_REPORT_PATH):
    drift_df = pd.read_csv(DRIFT_REPORT_PATH)
    st.dataframe(drift_df, use_container_width=True)

    drifted = drift_df[drift_df["drift_detected"] == True]
    if not drifted.empty:
        st.error("⚠️ Drift detected in some features!")
        st.write(drifted)
    else:
        st.success("✅ No significant drift detected.")
else:
    st.warning("Drift report not found. Run drift_monitor.py first.")
