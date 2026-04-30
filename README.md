# 💰 Personal Finance Health Tracker

An end-to-end **ML + GenAI powered web application** that analyzes your monthly finances and gives you a personalized Financial Health Score, risk classification, explainable insights, and AI-generated advice.

---

## 🎯 What It Does

- Takes your monthly financial inputs (income, rent, EMI, savings, etc.)
- Scores your **Financial Health (0-100)**
- Classifies you into **Stable / At Risk / Critical**
- Shows **exactly what's hurting or helping** your score (SHAP)
- Generates **personalized financial advice** via Google Gemini AI
- Tracks your **month-over-month progress** via MySQL

---

## 🏗️ Project Architecture
User Input (Streamlit)
↓
Feature Engineering (utils/)
↓
ML Model - XGBoost (model/)
↓
SHAP Explainability (shap_explainer/)
↓
Gemini AI Advisor (gemini/)
↓
MySQL Storage (database/)
↓
Dashboard + History (app/)

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| ML Model | XGBoost, Scikit-learn |
| Explainability | SHAP |
| AI Advisor | Google Gemini API |
| Database | MySQL |
| Frontend | Streamlit, Plotly |
| Data Processing | Pandas, NumPy |
| Environment | Python 3.12, virtualenv |

---
## 📁 Folder Structure

Personal_Finance_Health_Analyzer/
│
├── data/
│   ├── raw/                          # UCI Adult Income dataset
│   └── processed/
│       ├── finance_processed.csv     # With ratio features (visualization)
│       └── finance_processed_raw.csv # Raw columns (ML training)
│
├── model/
│   ├── train_model.py                # ML training pipeline
│   ├── finance_model.pkl             # Saved XGBoost model
│   └── label_encoder.pkl             # Saved LabelEncoder
│
├── shap_explainer/
│   └── explainer.py                  # SHAP explainability logic
│
├── database/
│   ├── schema.sql                    # MySQL table definitions
│   └── db_connect.py                 # DB connection and queries
│
├── gemini/
│   └── advisor.py                    # Gemini API prompt engineering
│
├── app/
│   ├── main.py                       # Streamlit entry point
│   ├── pages/
│   │   ├── input_form.py             # Screen 1 - Input form
│   │   ├── dashboard.py              # Screen 2 - Results
│   │   └── history.py                # Screen 3 - History
│   └── components/
│       ├── charts.py                 # Visualizations
│       └── score_card.py             # Score gauge
│
├── utils/
│   └── feature_engineering.py        # Reusable ratio calculator
│
├── notebooks/
│   └── exploration.ipynb             # Phase 1 - EDA notebook
│
├── .env                              # API keys (not pushed)
├── requirements.txt                  # Dependencies
└── README.md

---

## 🤖 ML Pipeline

### Dataset
- Base: UCI Adult Income Dataset (48,842 rows)
- After cleaning: 45,222 rows
- Simulated expense columns using real financial ratio distributions
- Target classes: **Stable (55%)**, **At Risk (37%)**, **Critical (8%)**

### Feature Engineering
8 ratio-based features engineered from raw expense data:
- `savings_rate`, `emi_ratio`, `rent_ratio`, `food_ratio`
- `need_ratio`, `want_ratio`, `total_expense_ratio`, `disposable_ratio`

### Model Comparison (5-Fold Stratified CV)

| Model | CV F1 | Test F1 |
|---|---|---|
| Logistic Regression | 0.8030 ± 0.0010 | 0.7955 |
| Random Forest | 0.8718 ± 0.0023 | 0.8789 |
| **XGBoost** ✅ | **0.8872 ± 0.0027** | **0.8752** |

### Key Decisions
- **Pipeline** used for Logistic Regression — prevents CV scaling leakage
- **compute_sample_weight** used for XGBoost — correct multiclass imbalance fix
- **CV-based model selection** — test set used only for final evaluation
- **Retrained on full data** before deployment

---

## 🐛 Issues Found and Fixed

### Issue 1 — Data Leakage
- **Problem:** First run showed 99.7% F1 — ratio features used for both labeling and training
- **Fix:** Maintained two separate datasets — raw CSV for training, ratio CSV for visualization
- **Result:** F1 dropped to honest 88.7% (CV)

### Issue 2 — Class Imbalance
- **Problem:** Critical class recall was 58% — model missing 42% of high-risk cases
- **Fix:** Applied `compute_sample_weight('balanced')` for XGBoost, `class_weight='balanced'` for others
- **Result:** Critical recall improved to 89%

### Issue 3 — Incorrect Imbalance Fix
- **Problem:** `scale_pos_weight=3` used for XGBoost — binary classification parameter, silently ignored for multiclass
- **Fix:** Replaced with `compute_sample_weight` — correct multiclass approach

---

## 🚀 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/srishtis2200/personal-finance-health-analyzer.git
cd personal-finance-health-analyzer

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Add your API keys
# Create .env file with:
# GEMINI_API_KEY=your_key_here
# MYSQL_PASSWORD=your_password_here

# Run the app
streamlit run app/main.py
```

---

## 📊 Current Development Status

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Data Engineering + Feature Engineering | ✅ Complete |
| Phase 2 | ML Model Training Pipeline | ✅ Complete |
| Phase 3 | SHAP Explainability | 🔨 In Progress |
| Phase 4 | MySQL Integration | ⏳ Pending |
| Phase 5 | Gemini AI Advisor | ⏳ Pending |
| Phase 6 | Streamlit UI | ⏳ Pending |
| Phase 7 | Deployment | ⏳ Pending |

---
