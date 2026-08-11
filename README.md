# Predictive Markdown Intelligence

**A production-ready machine learning system for fast-fashion markdown forecasting.**

Fast-fashion retailers lose margin in two directions: discount too early and you sacrifice full-price revenue; discount too late and inventory dies. The timing window is narrow and the signal is noisy. This system finds it.

Built end-to-end with synthetic retail data, leakage-safe feature engineering, XGBoost Tweedie modeling, SHAP interpretability, and a deployable scoring pipeline.

---

## The Business Problem

Fast-fashion SKUs have a hard lifecycle constraint — once demand peaks, it falls fast. Retailers that rely on instinct or calendar-based markdown schedules leave margin on the table in both directions.

What's needed is a system that reads the demand signal at the SKU level, distinguishes real decay from noise, and gives a markdown trigger at the right moment. That is what this project builds.

---

## What This System Does

- Ingests SKU-level daily demand with lifecycle, velocity, and discount features
- Identifies the pre-markdown decision window with leakage-safe feature engineering
- Scores markdown timing using XGBoost Tweedie (built for zero-inflated, sparse retail demand)
- Produces SHAP attribution per prediction — showing *why* each SKU gets its score
- Outputs a deployable scoring function ready for batch pipeline or FastAPI integration

---

## What Worked

The features and modeling choices that moved the needle:

| Feature / Choice | Why It Mattered |
|-----------------|-----------------|
| **Lifecycle position** | `days_since_launch` captured the launch → peak → decay arc |
| **Velocity signals** | `rolling_7d_avg` + `lag_7_units_sold` modeled short-term momentum |
| **XGBoost Tweedie** | Handled zero-inflation without a hurdle model |
| **Leakage-safe windows** | All features computed strictly before the prediction day |
| **Weighted MAE (WMAE)** | Penalised errors on high-value SKUs more than clearance items |

---

## What Failed

The dead ends — and why they died:

| Dead End | Reason |
|----------|--------|
| Long lag features (14–21 days) | Too noisy for short-lifecycle SKUs |
| Mid-range discounts (21–40%) | Naturally volatile; model struggled here |
| Early leakage | Draft features peeked past markdown events; pipeline rebuilt |
| External signals | Sentiment / web trends added latency and noise; removed |

---

## Tech & Evaluation

**Model Stack**
- XGBoost Tweedie
- SHAP (per-prediction attribution)
- Gain-based importance (global ranking)
- SKU-level holdout (true cold-start testing)

**Metrics**
- `WMAE` — primary business metric
- `Quantile loss` — tail behaviour
- `MAE by segment` — lifecycle + discount diagnostics

---

## Dataset

Synthetic retail dataset engineered to replicate the failure modes of real fast-fashion pipelines:

- Zero-inflated SKU-day demand
- Lifecycle curves (launch → peak → decay)
- Seasonal patterns
- Markdown events
- Stock-out noise
- Cold-start SKUs
- Volatile mid-range discounts

---

## Repository Structure

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
├── 08_retrained_models.ipynb
├── 09_SHAP_interpretability.ipynb
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

## Deployment Artifacts

Trained outputs ready for production integration:

- Trained XGBoost Tweedie model (`.pkl`)
- Feature ordering contract
- Preprocessing pipeline
- Scoring function (`predict_demand()`)

Ready for:
- Batch scoring pipelines
- FastAPI microservices
- Markdown optimisation engines

---

## What I'd Do Differently

Mid-range discount volatility (21–40%) was the hardest segment for the model. With real transaction data, I'd add a demand elasticity curve to the markdown trigger — so the system distinguishes a natural dip from a price-sensitivity response. That distinction changes the markdown recommendation entirely.

I'd also replace the synthetic dataset with scraped or licensed retail panel data to test whether the lifecycle features transfer to real cold-start SKUs or were overfit to the simulation's structure.

---

## Related Projects

- [DataForge](https://github.com/johnkirima/DataForge) — 9-agent pipeline that automated the EDA phase of this project
- [Supply Chain DI](https://github.com/johnkirima/Supply-Chain-DI) — Multi-warehouse inventory optimization under uncertainty
- [Fintech Sentiment Intelligence](https://github.com/johnkirima/Fintech-Sentiment-Intelligence-Analysis) — NLP system for fintech churn detection

---

## Author

**John Kirima** — Applied Data Scientist & Decision Intelligence Engineer  
[johnkirima.com](https://johnkirima.com) · [LinkedIn](https://www.linkedin.com/in/john-kirima/)

---
