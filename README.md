# 🔮 AI-Powered Customer Churn Prediction & Business Intelligence Dashboard

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow)
![SQL](https://img.shields.io/badge/SQL-MySQL-lightblue)
![Status](https://img.shields.io/badge/Status-In%20Development-green)

## 📋 Project Overview
An end-to-end, production-quality ML + BI solution that predicts which telecom customers are likely to churn and surfaces actionable insights through a 5-page Power BI dashboard.

**Dataset:** IBM Telco Customer Churn (7,043 customers, 21 features)

## 🎯 Business Problem
Telecom companies lose 5–7× more revenue replacing churned customers than retaining them. This project provides a proactive, data-driven churn prediction system.

## 🏗️ Project Architecture
```
Raw Data → Data Cleaning → EDA → Feature Engineering
        → ML Models → Predictions → SQL → Power BI Dashboard
```

## 📁 Project Structure
```
customer_churn_prediction/
├── data/
│   ├── raw/              ← Original IBM Telco CSV (not in Git)
│   ├── processed/        ← Cleaned & feature-engineered datasets
│   └── predictions/      ← ML model output (CustomerID + risk scores)
├── notebooks/            ← Jupyter notebooks (EDA, modelling, analysis)
├── src/
│   ├── data/             ← Data loading & cleaning scripts
│   ├── features/         ← Feature engineering
│   ├── models/           ← Model training, evaluation, selection
│   ├── visualization/    ← EDA charts & report figures
│   └── utils/            ← Logging, helpers, shared utilities
├── sql/
│   ├── tables/           ← CREATE TABLE statements
│   ├── queries/          ← Business queries (churn rate, revenue, etc.)
│   └── views/            ← SQL views for Power BI consumption
├── powerbi/
│   ├── measures/         ← All DAX measures documented
│   └── documentation/    ← Dashboard layout & design notes
├── models/
│   ├── saved/            ← Serialised trained models (.pkl / .joblib)
│   └── evaluation/       ← Metrics, confusion matrices, ROC curves
├── reports/
│   ├── figures/          ← All EDA & model evaluation charts
│   └── insights/         ← Business insight summaries
├── tests/                ← Unit tests for each src module
├── docs/                 ← Project documentation & architecture
├── config/
│   └── settings.py       ← Single source of truth for all paths & params
├── logs/                 ← Pipeline execution logs
├── requirements.txt
└── README.md
```

## 🚀 Quick Start
```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset
# Place WA_Fn-UseC_-Telco-Customer-Churn.csv into data/raw/

# 5. Run the pipeline
python src/data/data_cleaning.py
python src/features/feature_engineering.py
python src/models/train_models.py
python src/models/predict.py
```

## 📊 Technology Stack
| Layer | Tools |
|-------|-------|
| Language | Python 3.11 |
| Data | Pandas, NumPy |
| ML | Scikit-learn, XGBoost, imbalanced-learn |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Database | MySQL, SQLAlchemy |
| BI Dashboard | Power BI, DAX, Power Query |
| Version Control | Git, GitHub |

## 📈 Model Performance (Target)
| Metric | Target |
|--------|--------|
| ROC-AUC | ≥ 0.85 |
| Recall | ≥ 0.80 |
| F1 Score | ≥ 0.78 |

## 👤 Author
Built as a portfolio project demonstrating end-to-end Data Science + BI skills.

## 📄 License
MIT
