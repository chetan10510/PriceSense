PriceSense: AI-Powered Laptop Price Prediction
Overview

PriceSense is a complete machine learning system designed to predict laptop prices based on specifications such as RAM, storage, screen size, GPU, and CPU type. The project demonstrates the entire AI lifecycle, from data collection and preprocessing to model training, explainability, and monitoring.

This repository is structured for clarity and production-readiness, making it useful for businesses, researchers, and recruiters evaluating applied AI/ML skills.

Key Features

Laptop Price Prediction
Predicts prices of laptops from structured specifications.

Explainable AI (XAI)
Uses SHAP to provide both global and local interpretability of model predictions.

Interactive Dashboard
Streamlit dashboard with prediction interface, SHAP visualizations, and drift monitoring.

Data Drift Monitoring
Detects changes in incoming production data and reports feature drift.

Model Lifecycle Management
Supports training, saving, loading, and comparing multiple models with versioning.

Project Structure
├── artifacts/                 # Saved trained models
├── dashboard/                 # Streamlit app for user interaction
├── data/                      # Raw and processed datasets
│   ├── raw/                   
│   └── processed/
├── feature/                   # Feature extraction scripts
├── models/                    # Serialized model files
├── monitoring/                # Drift detection and monitoring scripts
├── reports/                   # Reports and generated plots
│   └── figures/
├── src/                       # Source code
│   ├── dashboard/             # Streamlit apps
│   ├── explainability/        # SHAP-based explanations
│   ├── model/                 # Model training and comparison
│   ├── predict/               # Prediction scripts
│   ├── preprocessing/         
│   ├── scraping/              # Web scrapers for data
│   └── visualize/             # Feature importance plots
└── tests/                     # Sample datasets for testing

Tech Stack

Programming Language: Python

Data Processing: pandas, numpy

Modeling: scikit-learn

Explainability: SHAP

Visualization: matplotlib, seaborn

Dashboarding: Streamlit

Persistence: joblib

How It Works

Data Pipeline

Scrape laptop specs and prices.

Clean and preprocess features (RAM, storage, GPU, CPU type, screen size).

Model Training

Train baseline, enhanced, and tuned models.

Final model: Random Forest Regressor with hyperparameter tuning.

Explainability

Global SHAP summary and bar plots.

Local SHAP explanations for single predictions.

Dashboard

Input laptop specs or upload a CSV file to predict prices.

View model explanations and drift reports.

Monitoring

Detects data drift over time and generates drift reports with visualizations.

Usage
Clone the Repository
git clone https://github.com/yourusername/PriceSense.git
cd PriceSense

Install Dependencies
pip install -r requirements.txt

Run the Dashboard
streamlit run dashboard/streamlit_app.py

Make Predictions
python src/predict/predict_price.py --input tests/sample_laptops.csv

Results

Achieved strong predictive performance (R² ~0.73+).

Demonstrated explainability via SHAP plots.

Built a production-ready system with monitoring and reporting.

Business Value

Enables e-commerce platforms to estimate fair laptop prices.

Improves transparency with explainable predictions.

Maintains reliability through drift monitoring and retraining alerts.

Skills Demonstrated

Machine Learning (regression, feature engineering, model tuning)

MLOps (model lifecycle, monitoring, drift detection)

Explainable AI (SHAP integration)

Software Engineering (modularized code, error handling, reproducibility)

Business Impact Thinking (pricing insights for e-commerce)

Future Improvements

Extend to other electronic devices (mobiles, tablets).

Deploy as a cloud-hosted API (FastAPI + Docker).

Automate retraining with CI/CD pipelines.

License

This project is licensed under the MIT License.