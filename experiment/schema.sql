-- ============================================================================
-- experiment/schema.sql
-- Star schema for the Markdown A/B Test + Difference-in-Differences (DiD) layer
-- ----------------------------------------------------------------------------
-- Dialect:  PostgreSQL. Fully runnable via DuckDB in Python (DuckDB parses the
--           PostgreSQL dialect used here). See experiment/README.md ("How to run").
--
-- Grain:    fact_daily_sales is at the SKU x Store x Day grain. This is the
--           unit of observation for the DiD estimator: we need per-store daily
--           outcomes so we can compare Treatment vs Control store trends before
--           and after the intervention.
--
-- Purpose:  The base repo answers "WHEN should we mark a SKU down?" (XGBoost
--           Tweedie timing at SKU x Day). This schema powers the follow-up
--           question a stakeholder will always ask: "Does the ML timing model
--           ACTUALLY beat a simple calendar rule?" We answer it causally with a
--           store-level randomized experiment and Difference-in-Differences.
--
-- Design map -> the 7 gaps (see README for full narrative):
--   Gap 1  Parallel trends              -> pre_exp_avg_daily_rev + is_experiment_period
--   Gap 2  SUTVA / geographic spillover -> dim_stores.market_region
--   Gap 3  Demand splitting             -> dim_stores.capacity_weight
--   Gap 4  Model validity drift         -> markdown_stage + is_experiment_period
--   Gap 5  Counterfactual definition    -> dim_experiment_assignments.assignment_group
--   Gap 6  Balance checks               -> dim_experiment_assignments.pre_exp_avg_daily_rev
--   Gap 7  Multiple testing             -> gross_margin_dollars vs gross_revenue (pre-declared outcomes)
-- ============================================================================


-- ----------------------------------------------------------------------------
-- DIMENSION: dim_stores
-- One row per physical store. 100 stores total.
-- Gap 2 (SUTVA): market_region lets us guarantee that Treatment and Control
--   stores never share a region, preventing cross-condition contamination
--   (a Control store's customers driving to a nearby Treatment store, or
--   shared regional inventory/marketing bleeding across conditions).
-- Gap 3 (Demand splitting): capacity_weight scales each store's baseline demand
--   so that total chain demand is realistically split across heterogeneous
--   stores instead of every store being an identical clone.
-- ----------------------------------------------------------------------------
CREATE TABLE dim_stores (
    store_id        VARCHAR      NOT NULL,   -- e.g. 'STORE_001'
    store_name      VARCHAR      NOT NULL,   -- human-readable label
    market_region   VARCHAR      NOT NULL,   -- Great_Lakes_East | Plains_North | Upper_Midwest | Central_Valley
    store_tier      VARCHAR      NOT NULL,   -- Flagship | Standard | Boutique
    capacity_weight DOUBLE       NOT NULL,   -- demand scaler; Flagship>Standard>Boutique (Gap 3)
    PRIMARY KEY (store_id)
);


-- ----------------------------------------------------------------------------
-- DIMENSION: dim_products
-- One row per SKU. Mirrors the base repo's SKU master + config.py elasticities.
-- price_elasticity is carried here so the fact-table builder (Phase 3) can apply
-- the SAME elasticity assumptions defined in config.py (category_elasticity) —
-- the existing config.py elasticities stay the single source of truth.
-- ----------------------------------------------------------------------------
CREATE TABLE dim_products (
    sku_id             VARCHAR   NOT NULL,   -- e.g. 'SKU_0001'
    category           VARCHAR   NOT NULL,   -- tshirts | jeans | jackets | shoes
    popularity_segment VARCHAR   NOT NULL,   -- winner | normal | dead
    base_price         DOUBLE    NOT NULL,   -- original (pre-markdown) price in USD
    price_elasticity   DOUBLE    NOT NULL,   -- from config.category_elasticity
    PRIMARY KEY (sku_id)
);


-- ----------------------------------------------------------------------------
-- DIMENSION: dim_experiment_assignments
-- One row per store. The randomization ledger produced in Phase 2.
-- Gap 5 (Counterfactual definition): assignment_group is the explicit, frozen
--   definition of the counterfactual. TREATMENT stores use XGBoost ML markdown
--   timing; CONTROL stores use a fixed calendar rule (discount every 3 weeks).
--   Without this row-level ledger there is no well-defined "what would have
--   happened otherwise".
-- Gap 6 (Balance checks): pre_exp_avg_daily_rev is the pre-period covariate we
--   test for balance across arms (and use to check parallel trends, Gap 1).
-- ----------------------------------------------------------------------------
CREATE TABLE dim_experiment_assignments (
    store_id             VARCHAR   NOT NULL,   -- FK -> dim_stores.store_id
    assignment_group     VARCHAR   NOT NULL,   -- 'TREATMENT' | 'CONTROL'
    pre_exp_avg_daily_rev DOUBLE   NOT NULL,   -- avg daily revenue in the pre-period (Gap 1 & Gap 6)
    PRIMARY KEY (store_id),
    FOREIGN KEY (store_id) REFERENCES dim_stores (store_id),
    CONSTRAINT chk_assignment_group
        CHECK (assignment_group IN ('TREATMENT', 'CONTROL'))
);


-- ----------------------------------------------------------------------------
-- FACT: fact_daily_sales
-- Grain: one row per (date, store_id, sku_id).
-- This is the observation table for the DiD estimator.
--   DiD compares the change in outcome (post - pre) for TREATMENT stores
--   against the change for CONTROL stores. is_experiment_period splits pre/post.
-- Gap 4 (Model validity drift): markdown_stage + is_experiment_period let us
--   inspect whether the ML model's behaviour during the live experiment matches
--   its training-time behaviour (i.e. detect drift between offline and online).
-- Gap 7 (Multiple testing): we store BOTH gross_revenue and gross_margin_dollars
--   so the analysis can pre-declare a single primary outcome and treat the
--   others as secondary, controlling the family-wise error rate.
-- ----------------------------------------------------------------------------
CREATE TABLE fact_daily_sales (
    fact_id               VARCHAR   NOT NULL,  -- composite key: date | store_id | sku_id
    date                  DATE      NOT NULL,
    store_id              VARCHAR   NOT NULL,  -- FK -> dim_stores.store_id
    sku_id                VARCHAR   NOT NULL,  -- FK -> dim_products.sku_id
    markdown_stage        INTEGER   NOT NULL,  -- 0..5 markdown depth stage (Gap 4)
    current_price         DOUBLE    NOT NULL,  -- price actually charged that day
    units_sold            INTEGER   NOT NULL,
    gross_revenue         DOUBLE    NOT NULL,  -- units_sold * current_price (secondary outcome)
    gross_margin_dollars  DOUBLE    NOT NULL,  -- primary outcome for the DiD (Gap 7)
    stockout_flag         BOOLEAN   NOT NULL,  -- TRUE if the SKU stocked out that day
    is_experiment_period  BOOLEAN   NOT NULL,  -- FALSE = pre-period, TRUE = experiment window (Gap 1 & 4)
    PRIMARY KEY (fact_id),
    FOREIGN KEY (store_id) REFERENCES dim_stores (store_id),
    FOREIGN KEY (sku_id)   REFERENCES dim_products (sku_id)
);


-- ----------------------------------------------------------------------------
-- INDEXES for analytical performance
-- ----------------------------------------------------------------------------
-- Primary analytical access path: group/scan by store x sku across the timeline
-- (used by every DiD aggregation and the parallel-trends plots).
CREATE INDEX idx_fact_store_sku_date
    ON fact_daily_sales (store_id, sku_id, date);

-- The DiD estimator constantly filters/splits on pre vs post period.
CREATE INDEX idx_fact_experiment_period
    ON fact_daily_sales (is_experiment_period);

-- Fast joins from the fact table back to the assignment ledger.
CREATE INDEX idx_assignments_group
    ON dim_experiment_assignments (assignment_group);

-- Region-level scans for the SUTVA / contamination checks (Gap 2).
CREATE INDEX idx_stores_region
    ON dim_stores (market_region);
