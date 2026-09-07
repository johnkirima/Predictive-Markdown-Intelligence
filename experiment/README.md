# Experiment Extension — Does the ML Markdown Model Actually Beat a Calendar Rule?

> A causal-inference / A/B-testing layer on top of the **Predictive Markdown Intelligence** repo.
> The base project predicts **WHEN** to trigger a markdown (XGBoost Tweedie + SHAP, at the SKU × Day grain).
> This extension answers the question every stakeholder eventually asks:
> **"Great, the model has good offline metrics — but does it actually make us more money than the simple calendar rule we use today?"**

---

## 1. What this extension does, and why

The base repo produces an ML model that decides the optimal markdown timing per SKU. Good offline error metrics (Tweedie deviance, SHAP explanations) are **not** proof of business value. A model can look great in backtest and still lose to a dumb rule once deployed, because of demand drift, cannibalization, and the fact that backtests reuse the same history the model trained on.

The only credible way to prove value is a **controlled experiment**: run the ML timing in some stores, run the incumbent calendar rule in others, and measure the *difference in the change* in outcomes. That is exactly what this layer builds.

**Business narrative:** We randomize 100 stores into two arms. **Treatment** stores let the XGBoost model choose markdown timing. **Control** stores keep the incumbent policy — a fixed calendar markdown every 3 weeks. We then estimate the causal lift in gross margin using **Difference-in-Differences (DiD)**. If Treatment stores' margin rises *relative to* Control stores after launch, the model earns its keep. If not, we have saved the company from shipping a model that only looked good on paper.

**Primary outcome:** `gross_margin_dollars` (pre-declared).
**Secondary outcome:** `gross_revenue`.

---

## 2. The 4 locked architectural decisions

| # | Decision | What we locked | Why |
|---|----------|----------------|-----|
| 1 | **Store count & arms** | 100 stores — 50 Treatment (XGBoost ML timing) / 50 Control (fixed calendar, discount every 3 weeks) | Store-level randomization (cluster randomization) is the realistic unit of intervention: you deploy a pricing policy to a *store*, not to a single SKU. 50/50 maximizes power. |
| 2 | **Counterfactual definition** | Control = fixed calendar markdown every 3 weeks (the incumbent policy) | DiD requires a well-defined "what would have happened otherwise." The counterfactual is the business-as-usual rule the ML model is meant to replace — not "no markdowns," which no retailer actually does. |
| 3 | **Causal method** | Difference-in-Differences (DiD) | DiD nets out (a) fixed differences between stores and (b) chain-wide time shocks (seasonality, macro demand). It needs only the parallel-trends assumption, which we can defend with pre-period data. |
| 4 | **Storage / compute** | PostgreSQL-dialect star schema, runnable via **DuckDB** in Python | DuckDB runs the Postgres-dialect DDL/SQL locally with zero server setup, reads CSVs directly, and is fast enough for the full SKU × Store × Day fact table. Portable to a real Postgres warehouse unchanged. |

---

## 3. The 7 gaps this design addresses

A naive "just compare the two groups' averages" analysis is wrong for at least seven reasons. Each is designed for explicitly:

1. **Parallel trends.** DiD is only valid if Treatment and Control would have moved together absent the intervention. We store `pre_exp_avg_daily_rev` and split every fact row with `is_experiment_period`, so we can plot and test pre-period trends before trusting the estimate.
2. **SUTVA / geographic contamination.** One store's treatment must not affect another store's outcome. `dim_stores.market_region` lets Phase 2 guarantee no region holds *both* arms, so customers can't substitute across conditions and regional inventory/marketing can't bleed between arms.
3. **Demand splitting.** Stores are not identical. `capacity_weight` (Flagship > Standard > Boutique) scales baseline demand so total chain demand is realistically distributed across heterogeneous stores, keeping balance and power calculations honest.
4. **Model validity drift.** An ML model can behave differently live than in training. `markdown_stage` + `is_experiment_period` let us compare the model's in-experiment behaviour against its offline behaviour and detect drift.
5. **Counterfactual definition.** `dim_experiment_assignments.assignment_group` freezes the explicit counterfactual (calendar rule) per store, so "what would have happened otherwise" is never ambiguous.
6. **Balance checks.** Randomization can still produce imbalanced arms by chance. `pre_exp_avg_daily_rev` is the pre-period covariate we test for balance across arms (and reuse for parallel trends).
7. **Multiple testing.** Testing many outcomes inflates false positives. We pre-declare `gross_margin_dollars` as the single primary outcome and treat `gross_revenue` (and others) as secondary, controlling the family-wise error rate.

---

## 4. Phased build plan

| Phase | Deliverable | Status |
|-------|-------------|--------|
| **Phase 1** | Schema (`schema.sql`) + store dimension generator (`generate_stores.py`) + this README | ✅ this PR |
| **Phase 2** | `assign_treatment.py` — store-level randomization, **SRM** (sample-ratio mismatch) check, balance checks, and the region-separation rule (Gap 2) | ⬜ next |
| **Phase 3** | `build_fact_table.py` — the SKU × Store × Day fact-table builder (applies `capacity_weight` and `config.py` elasticities) | ⬜ |
| **Phase 4** | `04_did_analysis.ipynb` — parallel-trends diagnostics + the DiD estimate with clustered standard errors | ⬜ |

Phase 1 touches **only** the new `experiment/` folder. No existing files (notebooks 01–09, `config.py`, `build_modeling_table.py`, etc.) are modified. The existing `config.py` elasticities remain the single source of truth and are carried into `dim_products.price_elasticity` in Phase 3.

---

## 5. Interview talking points

Use these to answer the questions retail/supply-chain DS interviewers actually ask:

- **"Why not just compare Treatment vs Control averages after launch?"**
  Because stores differ *before* launch and the whole chain is hit by the same seasonality. DiD subtracts both: it compares the *change* (post − pre) in Treatment against the *change* in Control, removing fixed store effects and common time shocks.

- **"What's your identifying assumption?"**
  Parallel trends — absent the intervention, both arms would have followed the same trajectory. I defend it with pre-period data (`pre_exp_avg_daily_rev`, `is_experiment_period`) via a parallel-trends plot and a placebo/pre-trend test.

- **"How did you randomize, and why at the store level?"**
  Cluster randomization at the store level, because a pricing policy is deployed per store; randomizing SKUs within a store would let the two policies interfere on the same shelf. I verify the split with an SRM check.

- **"How do you handle spillover / SUTVA?"**
  Region-level separation — no `market_region` contains both arms — so customers can't substitute across conditions and shared regional inventory/marketing can't contaminate the comparison.

- **"How do you avoid p-hacking across metrics?"**
  One pre-declared primary outcome (`gross_margin_dollars`); everything else is secondary with a multiple-testing correction.

- **"How do you know the model didn't just get lucky / drift?"**
  `markdown_stage` + `is_experiment_period` let me compare live vs offline behaviour, and clustered standard errors keep inference honest at the store level.

---

## 6. How to run

Requirements: Python 3.9+, `pandas`, `numpy`, and `duckdb` (`pip install pandas numpy duckdb`).

### 6.1 Generate the store dimension

```bash
cd experiment
python generate_stores.py
```

This writes `experiment/data/dim_stores.csv` (100 stores) and prints a summary (count by tier, count by region, capacity-weight stats).

### 6.2 Create the schema and load data into DuckDB

The DDL is written in the PostgreSQL dialect and runs unmodified in DuckDB:

```python
import duckdb

con = duckdb.connect("experiment/markdown_experiment.duckdb")

# 1) create the star schema
with open("experiment/schema.sql") as f:
    con.execute(f.read())

# 2) load the generated store dimension
con.execute("""
    INSERT INTO dim_stores
    SELECT * FROM read_csv_auto('experiment/data/dim_stores.csv', header=True)
""")

# 3) sanity check
print(con.execute("SELECT store_tier, COUNT(*) FROM dim_stores GROUP BY 1").df())
print(con.execute("SELECT market_region, COUNT(*) FROM dim_stores GROUP BY 1").df())

con.close()
```

`dim_products`, `dim_experiment_assignments`, and `fact_daily_sales` are populated in Phases 2–3. The same `schema.sql` can be applied to a real PostgreSQL instance without changes.
