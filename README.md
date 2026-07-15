# 🛒 Predictive Markdown Intelligence

A production‑ready machine learning system for fast‑fashion markdown forecasting.  
Built end‑to‑end with synthetic retail data, leakage‑safe feature engineering, XGBoost Tweedie modeling, SHAP interpretability, and a deployable scoring pipeline.

---

## 1. Problem

Fast‑fashion SKUs die young. Demand is:

- Zero‑inflated
- Sparse
- Volatile
- Lifecycle‑driven
- Discount‑sensitive

Classical time‑series models collapse under these conditions.  
This project builds a structured, feature‑driven forecasting system that handles:

- Cold‑start SKUs
- Stock‑out noise
- Short lifecycles
- Mid‑range discount volatility
- Pre‑markdown decision windows

---

## 2. What Worked

The features and modeling choices that moved the needle:

| Feature / Choice | Why It Mattered |
|---|---|
| **Lifecycle position** | `days_since_launch` captured the launch → peak → decay arc |
| **Velocity signals** | `rolling_7d_avg` + `lag_7_units_sold` modeled short‑term momentum |
| **XGBoost Tweedie** | Handled zero‑inflation without a hurdle model |
| **Leakage‑safe windows** | All features computed strictly before the prediction day |
| **Weighted MAE (WMAE)** | Penalised errors on high‑value SKUs more than clearance items |

---

## 3. What Failed

The dead ends — and why they died:

| Dead End | Reason |
|---|---|
| Long lag features (14–21 days) | Too noisy for short‑lifecycle SKUs |
| Mid‑range discounts (21–40%) | Naturally volatile; model struggled here |
| Early leakage | Draft features peeked past markdown events; pipeline rebuilt |
| External signals | Sentiment / web trends added latency and noise; removed |

---

## 4. Tech & Evaluation

**Model Stack**

- XGBoost Tweedie
- SHAP (per‑prediction attribution)
- Gain‑based importance (global ranking)
- SKU‑level holdout (true cold‑start testing)

**Metrics**

- `WMAE` — primary business metric
- `Quantile loss` — tail behaviour
- `MAE by segment` — lifecycle + discount diagnostics

---

## 5. Dataset

Synthetic retail dataset engineered for realism:

- Zero‑inflated SKU‑day demand
- Lifecycle curves
- Seasonal patterns
- Markdown events
- Stock‑out noise
- Cold‑start SKUs
- Volatile mid‑range discounts

Designed to replicate the failure modes of real fast‑fashion pipelines.

---

## 📁 Repository Structure

```
Predictive-Markdown-Intelligence/
│
├── data/                          # Synthetic retail dataset
├── models/                        # Trained XGBoost Tweedie model + artifacts
├── results/                       # SHAP plots, diagnostics, evaluation
│
├── 01_baseline_models.ipynb
├── 02_linear_regression.ipynb
├── 03_random_forest.ipynb
├── 04_xgboost_tweedie.ipynb
├── 05_error_analysis.ipynb
├── 06_feature_importance.ipynb
├── 07_deployment_prep.ipynb
│
├── build_modeling_table.py
├── config.py
├── generate_calendar.py
├── generate_daily_sales.py
├── generate_price_schedule.py
├── generate_sku_master.py
├── visualize_data.py
│
└── README.md
```

---

## 🚀 Deployment

Deployment artifacts include:

- Trained XGBoost Tweedie model (`.pkl`)
- Feature ordering contract
- Preprocessing pipeline
- Scoring function (`predict_demand()`)

Ready for integration into:

- Batch scoring pipelines
- FastAPI microservices
- Markdown optimisation engines

---

## 💡 Business Value

A retailer using this system could:

- Reduce stockouts
- Reduce overstocks
- Improve markdown timing
- Increase gross margin
- Reduce inventory waste
- Improve SKU‑level demand visibility

---

## 🧑‍💻 Author

**John Kirima**  
Data Scientist · Machine Learning · AI Engineering  
Iowa City, IA  
[johkirima.com](https://www.johkirima.com)
