# 📡 Telco Customer Churn Analytics Dashboard

A fully interactive Streamlit analytics application that identifies key factors
causing customers to cancel their service and predicts which customers are at the
highest risk of leaving next month.

## Dataset
IBM Telco Customer Churn dataset (`Telco-Customer-Churn.csv`) — 7,043 customers,
21 raw features including demographics, services, contract type, and billing.

## Project Structure

```
telco_churn_app/
├── app.py              # Streamlit frontend — main entry point
├── data_loader.py      # Data loading, cleaning, feature engineering
├── analytics.py        # KPI calculations (Churn Rate, ARPU, CLV, trends)
├── ml_model.py         # Gradient Boosting churn prediction model
├── requirements.txt    # Python dependencies
└── README.md
```

## Features

| Tab | Description |
|-----|-------------|
| 📊 Overview | KPI cards, churn distribution pie, churn by contract/internet, tenure analysis |
| 📈 Monthly Trends | Dual-axis churn trend, region & product tier heatmap, filterable by sidebar |
| 🔍 Segment Analysis | Correlation-based churn drivers, any-dimension deep-dive, CLV violin plots |
| 🤖 ML Model | Gradient Boosting, AUC ~0.84, feature importances, confusion matrix |
| 🚨 At-Risk Customers | Top-N highest-risk table, revenue exposure, churn prob vs charges scatter |
| 🗄️ Data Quality | Missing values report, dtype overview, raw data preview |
| 💡 Recommendations | Findings + 7 prioritised business recommendations with impact table |

## Quick Start

```powershell
# From the repo root (IBM Project folder)
pip install -r telco_churn_app/requirements.txt
cd telco_churn_app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## KPIs Calculated

| Metric | Formula |
|--------|---------|
| **Churn Rate** | `Churned Customers / Total Customers` |
| **ARPU** | `Mean(MonthlyCharges)` across all customers |
| **CLV** | `MonthlyCharges × (1 / churn_rate)` per customer |
| **Revenue at Risk** | `Sum(MonthlyCharges)` of High-risk customers |

## ML Model

- **Algorithm**: Gradient Boosting Classifier (sklearn)
- **Features**: 40+ one-hot encoded features from all service/demographic columns
- **ROC-AUC**: ~0.84 (test) / ~0.84 (5-fold CV)
- **Risk Tiers**: High (≥60%), Medium (35–60%), Low (<35%)

## Dashboard Filters (Sidebar)

All charts and KPIs respond to:
- **Region**: North / South / East / West
- **Product Tier**: Basic / Standard / Premium / Enterprise
- **Contract Type**: Month-to-month / One year / Two year
