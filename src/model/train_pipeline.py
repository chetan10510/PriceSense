# train_pipeline.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Paths
DATA_PATH = "data/processed/amazon_laptops_features.csv"
MODEL_PATH = "models/laptop_price_model.pkl"

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)
print(f"[INFO] Dataset shape: {df.shape}")

# Features & target
X = df[['brand', 'ram_gb', 'storage_gb']]
y = df['price']

# One-hot encode 'brand'
encoder = OneHotEncoder(sparse_output=False, drop='first')
X_brand_encoded = encoder.fit_transform(X[['brand']])
X_encoded = pd.concat([
    pd.DataFrame(X_brand_encoded, columns=encoder.get_feature_names_out(['brand'])),
    X[['ram_gb', 'storage_gb']].reset_index(drop=True)
], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"[INFO] Model Performance:\n  MSE: {mse:.4f}\n  R²:  {r2:.4f}")

# Save model and encoder
joblib.dump({'model': model, 'encoder': encoder}, MODEL_PATH)
print(f"[INFO] Trained model saved → {MODEL_PATH}")
