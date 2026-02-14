📊 Customer Churn Analysis & Prediction
🚀 Project Overview

This project performs end-to-end churn analysis, including:

Data cleaning & standardization

Exploratory Data Analysis (EDA)

Feature engineering

Churn rate segmentation

Predictive modeling using Random Forest

Model evaluation

Risk segmentation

Automated export of dashboard-ready files

The final output includes:

Trained model (.pkl)

Cleaned dataset with predictions

Excel dashboard with metrics & feature importance

Auto-saved charts

📁 Project Structure
.
├── data/
│   └── customer_churn_dataset.csv
├── images/
│   └── chart_1.png
│   └── chart_2.png
├── churn_model_rf.pkl
├── churn_prevision.csv
├── churn_dashboard.xlsx
└── main.py

🧠 Business Objective

Identify customers at risk of churn and segment them into:

🟢 Low Risk

🟡 Medium Risk

🔴 High Risk

This allows companies to:

Act before churn happens

Prioritize retention campaigns

Increase customer lifetime value (LTV)

Reduce support-related churn drivers

🔎 Key Steps
1️⃣ Data Acquisition

CSV dataset loaded using pandas

2️⃣ Data Cleaning & Standardization

Column normalization

Object value capitalization

Fuzzy matching with RapidFuzz for:

Contract

Payment method

Internet service

Outlier treatment using IQR method

Missing values handling

3️⃣ Feature Engineering

Created strategic business features:

Tenure_category

Consumption_category

Support_calls_category

Risk segmentation based on churn probability

4️⃣ Exploratory Data Analysis (EDA)

Visual insights generated for:

Tenure distribution

Churn per tenure segment

Contract type vs churn

Support calls impact

Consumption impact

Payment method LTV analysis

Charts are automatically saved inside /images.

5️⃣ Predictive Modeling

Model used:

RandomForestClassifier
- n_estimators=150
- max_depth=10
- random_state=42


Steps:

One-hot encoding

Train/Test split (70/30)

Model training

Probability prediction

Risk segmentation

📈 Model Outputs

The model exports:

📄 churn_prevision.csv

Full dataset with:

Predicted churn

Churn probability

Risk segment

📊 churn_dashboard.xlsx

Includes:

Complete_data

Model_Metrics

Train accuracy

Test accuracy

Total customers

Predicted churn %

High-risk %

Variables_Importance

High_Risk_Customer (Top 20)

🧩 Risk Segmentation Logic
Probability	Segment
0 – 0.3	Low
0.3 – 0.7	Medium
0.7 – 1.0	High
🛠 Technologies Used

Python

Pandas

Seaborn

Matplotlib

RapidFuzz

Scikit-learn

Joblib

🎯 Business Insights Examples

Customers with short tenure show higher churn rates.

Higher support calls correlate strongly with churn.

Certain contract types are more exposed to churn.

Internet service type influences churn probability.

High monthly charges combined with low tenure increases risk.

⚡ How to Run
pip install pandas seaborn matplotlib rapidfuzz scikit-learn joblib


Then:

python main.py

💡 Strategic Value

This project demonstrates:

Data cleaning in real-world messy datasets

Business-driven feature engineering

Customer segmentation logic

Predictive modeling pipeline

Dashboard-ready export

Retention strategy foundation

📌 Future Improvements

Hyperparameter tuning

Cross-validation

SHAP feature importance

Model comparison (XGBoost, LightGBM)

Deployment with FastAPI

Dashboard with Streamlit or Power BI

🏆 Portfolio Positioning

This is not just a model.

It is a customer retention decision system prototype.

You can present it as:

“Churn Prediction & Risk Segmentation Engine for Telecom / Subscription Businesses”

That sounds corporate. Because it is.
