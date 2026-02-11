# Credit Risk Scoring & Probability of Default (PD) Modeling

**End-to-end credit risk modeling pipeline built the way a rating agency or risk team would expect.**  
This project develops, validates, and deploys an **interpretable Probability of Default (PD) model** to support **credit analysts** with transparent risk scores, calibrated probabilities, and analyst-friendly outputs.

> 🔍 Focus: **Explainability, calibration, documentation, and analyst usability** — not just model accuracy.

---

## Why this project matters
Credit risk models are only useful if analysts **trust** them.  
This project mirrors real-world **quantitative modeling workflows** used by credit rating agencies and financial institutions:

- Clear default definitions & time-based validation  
- Interpretable baseline models (logistic regression)  
- Challenger models (gradient boosting) with controlled complexity  
- Probability **calibration** (critical for PD usage)  
- Analyst-ready risk bands and scoring tools  

---

## What this project delivers
- 📊 **Calibrated Probability of Default (PD)** for each obligor/loan  
- 🧮 **Risk grades** (A–E) derived from PD thresholds  
- 🧠 **Model explainability** (coefficients + SHAP)  
- 🛠️ **Reusable scoring pipeline** (Python + SQL)  
- 📈 **Evaluation metrics** aligned with credit risk practice  
- 🧑‍💼 **Analyst-facing scoring app** (Streamlit)

---

## Tech Stack
**Core**
- Python (pandas, NumPy, scikit-learn)
- SQL (DuckDB for feature engineering)
- Git / GitHub (versioned, reproducible pipeline)

**Modeling**
- Logistic Regression (interpretable baseline)
- XGBoost (challenger model)
- Probability calibration (Platt scaling, isotonic regression)

**Evaluation**
- ROC-AUC, PR-AUC
- KS statistic
- Brier score
- Calibration curves

**Delivery**
- Parquet datasets
- Joblib-serialized models
- Streamlit analyst dashboard

---

## Modeling Approach (High-Level)
1. **Label Definition**  
   Default vs non-default based on loan outcomes  
2. **SQL-based Feature Engineering**  
   Clean, auditable transformations (no hidden magic)  
3. **Time-Aware Train/Test Split**  
   Prevents forward-looking bias  
4. **Baseline Model**  
   Logistic regression with class balancing  
5. **Challenger Model**  
   Gradient boosting for non-linear patterns  
6. **Probability Calibration**  
   Ensures PDs behave like true probabilities  
7. **Risk Band Mapping**  
   PD → Analyst-friendly grades (A–E)  


