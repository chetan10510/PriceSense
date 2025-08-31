# PriceSense – Laptop Price Prediction Dashboard

PriceSense is a machine learning application for predicting laptop prices based on specifications such as RAM, storage, CPU type, screen size, GPU, and brand.  
It provides both **interactive predictions** and **explainability insights** using SHAP, along with data drift monitoring.

---

## Features

- **Laptop Price Prediction**  
  Upload a CSV file or enter laptop specs manually to get price predictions.

- **Model Explainability**  
  - SHAP summary and bar plots for global feature importance.  
  - SHAP waterfall/force plots for individual predictions.

- **Data Drift Monitoring**  
  Detects when incoming data deviates from the training dataset distribution.

- **Interactive Dashboard**  
  Built with [Streamlit](https://streamlit.io) for a clean and responsive UI.

---

## Project Structure

PriceSense/
│
├── data/
│ ├── raw/ # Raw dataset
│ ├── processed/ # Cleaned and feature-engineered datasets
│
├── models/ # Trained ML models
├── notebooks/ # Jupyter notebooks for exploration & training
├── reports/
│ ├── figures/ # SHAP and evaluation plots
│ ├── drift_report.csv # Drift monitoring results
│
├── src/ # Source code for preprocessing, training, monitoring
│ ├── model/ # Model training pipeline
│ ├── monitoring/ # Drift detection scripts
│ └── explain/ # SHAP explainability
│
├── app.py # Streamlit dashboard
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore



---

## Installation

Clone the repository and set up a virtual environment:


git clone https://github.com/YOUR_USERNAME/PriceSense.git
cd PriceSense

# Create virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Usage
Run the Streamlit App
bash
Copy code
streamlit run app.py
Example Manual Input
RAM: 16 GB

Storage: 512 GB

CPU Type: Intel Core i5

Screen Size: 15.6 inch

GPU: Integrated

Brand: HP

Prediction output will display estimated price along with SHAP explanation.

Example CSV Format
csv
Copy code
ram_gb,storage_gb,screen_inch,gpu,brand,cpu_type
16,512,15.6,0,HP,Intel Core i5
8,256,14.0,0,Dell,AMD Ryzen 3
Model Training
The pipeline:

Data Preprocessing

Scaling numerical features

One-hot encoding categorical features

Model

Random Forest Regressor (tuned hyperparameters)

Evaluation

Metrics: MSE, R²

Explainability via SHAP

Results
Achieved R² ≈ 0.73 on test set.

SHAP analysis shows that RAM, CPU type, and storage are the most important factors in determining laptop prices.

Drift monitoring ensures reliability when deployed on new data.

Future Work
Deploy app on cloud (AWS/GCP/Streamlit Cloud).

Add more advanced models (XGBoost, LightGBM).

Extend dataset with GPU benchmarks and battery life.

Build REST API endpoints for integration.

Author
Developed by [Your Name]
AI/ML Engineer | Machine Learning & NLP Enthusiast

License
This project is licensed under the MIT License.
