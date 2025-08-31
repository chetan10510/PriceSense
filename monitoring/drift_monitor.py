import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import os

reference_path = "data/processed/amazon_laptops_features_enhanced_clean.csv"
new_data_path = "data/processed/new_laptops.csv"

print(f"[INFO] Using reference data: {reference_path}")
reference_data = pd.read_csv(reference_path)

if os.path.exists(new_data_path):
    new_data = pd.read_csv(new_data_path)
else:
    print("[WARNING] new_laptops.csv not found. Using reference data as placeholder...")
    new_data = reference_data.copy()

# Only select numeric columns for drift
numeric_cols = reference_data.select_dtypes(include=[np.number]).columns
drift_report = []

for col in numeric_cols:
    stat, p_val = ks_2samp(reference_data[col], new_data[col])
    drift_report.append({"Feature": col, "p_value": p_val, "drift_detected": p_val < 0.05})

drift_df = pd.DataFrame(drift_report)
drift_df.to_csv("reports/drift_report.csv", index=False)
print("[INFO] Drift report saved: reports/drift_report.csv")

# Plot only numeric feature drift
for col in numeric_cols:
    plt.figure()
    reference_data[col].plot(kind="kde", label="Reference")
    new_data[col].plot(kind="kde", label="New")
    plt.title(f"Drift detection for {col}")
    plt.legend()
    plt.savefig(f"reports/figures/drift_{col}.png")
    plt.close()

print("[INFO] Drift plots saved in reports/figures/")
