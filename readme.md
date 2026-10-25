📉 Customer Churn Prediction Engine & Retention Dashboard

📌 Project Overview

Customer attrition (churn) is a critical metric for subscription-based businesses. This project provides an end-to-end Machine Learning solution to predict which customers are at high risk of leaving and identifies the underlying statistical drivers of their behavior.

The project culminates in a Streamlit web application that allows business stakeholders to instantly simulate customer risk profiles, understand feature impacts via SHAP, and evaluate the financial ROI of targeted retention strategies.

🚀 Key Features & Business Impact

Predictive Engine: A hyperparameter-tuned Random Forest classification model trained on complex demographic and subscription billing data.

Model Interpretability (SHAP): De-black-boxed the model using Shapley Additive exPlanations to isolate feature impacts. The model successfully identified customer tenure and monthly charges as the primary drivers of attrition.

Interactive Dashboard: A front-end Streamlit application that outputs risk probabilities and actionable business recommendations (e.g., offering a promotional discount to save high Lifetime Value (LTV) customers).

Production-Ready ML: Serialized optimized models, standard scalers, and data pipelines using joblib to prevent data leakage and ensure consistency during inference.

📂 Repository Structure

Customer_Churn/
│
├── data/                       # Raw and processed Telco customer datasets
├── models/                     # Serialized Random Forest model & Scalers (.pkl / .joblib)
├── notebooks/                  
│   └── 01_eda_and_baseline.ipynb # Exploratory Data Analysis & Model Training
├── src/                        
│   └── app.py                  # Main Streamlit application script
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation


⚙️ Installation & Usage

To run this project locally, follow these steps:

Clone the repository:

git clone https://github.com/TONY00009/Customer_Churn.git
cd Customer_Churn


Create a virtual environment (Recommended):

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate


Install the dependencies:

pip install -r requirements.txt


Run the Streamlit application:

streamlit run src/app.py


📊 Key Insights

During the Exploratory Data Analysis (EDA) and SHAP evaluation, the following patterns emerged:

Tenure is King: Customers with less than 6 months of tenure have the highest churn probability.

Contract Types Matter: Month-to-month contracts strongly correlate with high turnover compared to 1-year or 2-year lock-ins.

Price Sensitivity: High monthly charges without bundled services (like tech support or online security) significantly increase churn risk.

🛠️ Technology Stack

Language: Python

Data Processing & EDA: Pandas, NumPy, Matplotlib, Seaborn

Machine Learning: Scikit-Learn (Random Forest, Data Preprocessing)

Model Explainability: SHAP

Deployment & Frontend: Streamlit