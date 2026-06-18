# build_modeling_table.py
# ──────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Turn the four raw simulation tables (sku_master, calendar, price_schedule,
#   daily_sales) into ONE clean, leakage-free, model-ready table whose target
#   variable is `units_sold`.
#
#   The whole point of this project is to *predict demand* and pick smart
#   markdowns. That means every feature we feed the model must be something we
#   would genuinely know BEFORE the sale happens. A huge part of this script is
#   therefore about (a) removing rows that corrupt the demand signal and
#   (b) dropping "leakage" columns that secretly encode the answer.
#
# WHAT THIS SCRIPT DOES (high level)
#   1. Load all 4 CSVs from data/.
#   2. Join them into one unified table at the (sku_id, date) grain.
#   3. Filter out post-stockout rows (units_sold there is censored, not demand).
#   4. Engineer momentum / freshness / scarcity / price features.
#   5. One-hot encode category and day_of_week.
#   6. Keep season, is_holiday, markdown_stage as-is.
#   7. Drop leakage columns (latent_demand, markdown_depth, days_at_current_price).
#   8. Save to data/modeling_table.csv.
#   9. Print a validation summary (shape, columns, nulls, sample rows).
#
# NOTE ON THE REAL SCHEMA
#   The raw files use `original_price` (not `base_price`), and there is NO
#   `starting_inventory` column anywhere — it only exists implicitly inside the
#   data generator. We reconstruct it purely from data (see Step 4d). We also
#   treat `original_price` as the "base price" for the discount calculation.
# ──────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np

DATA_DIR = "data"
OUTPUT_PATH = f"{DATA_DIR}/modeling_table.csv"


def load_raw_tables():
    """Step 1 — Load all four raw CSVs.

    Each table plays a distinct role:
      • sku_master    : one row per product (the "what")   -> launch_date, tier, segment
      • calendar      : one row per date    (the "when")   -> day_of_week, holidays, season
      • price_schedule: sku x date prices   (the "how much")-> current_price, markdown_stage
      • daily_sales   : sku x date outcomes (the "result") -> units_sold, inventory, flags
    We parse date columns up front so joins and date math are reliable.
    """
    sku_master = pd.read_csv(f"{DATA_DIR}/sku_master.csv", parse_dates=["launch_date"])
    calendar = pd.read_csv(f"{DATA_DIR}/calendar.csv", parse_dates=["date"])
    price_schedule = pd.read_csv(f"{DATA_DIR}/price_schedule.csv", parse_dates=["date"])
    daily_sales = pd.read_csv(f"{DATA_DIR}/daily_sales.csv", parse_dates=["date"])

    print("Loaded raw tables:")
    print(f"  sku_master     : {sku_master.shape}")
    print(f"  calendar       : {calendar.shape}")
    print(f"  price_schedule : {price_schedule.shape}")
    print(f"  daily_sales    : {daily_sales.shape}")
    return sku_master, calendar, price_schedule, daily_sales


def join_tables(sku_master, calendar, price_schedule, daily_sales):
    """Step 2 — Join everything into one unified table at the (sku_id, date) grain.

    `daily_sales` is already largely denormalized (it carries category, season,
    prices, calendar fields, etc.), so it is the natural "spine" of the join.
    To keep the result clean we only pull in the columns each side genuinely
    *adds* — we never duplicate a column that already exists on the spine.

      • From sku_master   : launch_date, price_tier, base_demand (new info).
      • From price_schedule: nothing new (all already on daily_sales) — joined
                             only to demonstrate/verify the relationship.
      • From calendar     : nothing new (all already on daily_sales).
    """
    df = daily_sales.copy()

    # --- Join sku_master: bring only columns not already present on the spine ---
    sku_cols_to_add = [c for c in sku_master.columns
                       if c not in df.columns or c == "sku_id"]
    df = df.merge(sku_master[sku_cols_to_add], on="sku_id", how="left")

    # --- Join price_schedule: add only genuinely-new columns (keyed sku_id+date) ---
    ps_new = [c for c in price_schedule.columns
              if c not in df.columns or c in ("sku_id", "date")]
    if len(ps_new) > 2:  # more than just the join keys
        df = df.merge(price_schedule[ps_new], on=["sku_id", "date"], how="left")

    # --- Join calendar: add only genuinely-new columns (keyed by date) -----------
    cal_new = [c for c in calendar.columns if c not in df.columns or c == "date"]
    if len(cal_new) > 1:  # more than just the join key
        df = df.merge(calendar[cal_new], on="date", how="left")

    print(f"\nUnified table after joins: {df.shape}")
    return df


def filter_post_stockout(df):
    """Step 3 — Remove rows where post_stockout_flag == 1.

    WHY THIS MATTERS (this is the single most important data-quality step):
      Once a SKU sells out, `units_sold` is no longer driven by *demand* — it is
      driven by *zero inventory*. Those rows show artificially low (often zero)
      sales that have nothing to do with how much customers actually wanted the
      product. Training on them would teach the model that demand collapses,
      when in reality we simply had nothing left to sell. We drop them BEFORE any
      feature engineering so corrupted rows can't even leak into lags/rolling
      averages.
    """
    before = len(df)
    df = df[df["post_stockout_flag"] == 0].copy()
    removed = before - len(df)
    print(f"\nFiltered post-stockout rows: removed {removed:,} of {before:,} "
          f"({removed / before:.1%}) -> {len(df):,} rows remain")
    return df


def engineer_features(df, daily_sales_raw):
    """Step 4 — Feature engineering. Each block explains the intuition first."""

    # Sort so every time-based feature (lags, rolling) is computed in calendar
    # order WITHIN each SKU. Without this, shift()/rolling() would be nonsense.
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)

    # ── 4a. lag_7_units_sold — MOMENTUM ──────────────────────────────────────
    # Units sold for the SAME SKU exactly 7 days ago. Last week's sales are one
    # of the strongest predictors of this week's sales (demand has inertia).
    # Using a 7-day lag also aligns same-weekday-to-same-weekday in many cases.
    df["lag_7_units_sold"] = df.groupby("sku_id")["units_sold"].shift(7)

    # ── 4b. lag_14_units_sold — LONGER-TERM TREND ────────────────────────────
    # Sales two weeks ago. Combined with the 7-day lag, the model can sense
    # whether demand is accelerating or decaying over a fortnight.
    df["lag_14_units_sold"] = df.groupby("sku_id")["units_sold"].shift(14)

    # ── 4c. rolling_7d_avg — NOISE SMOOTHING ─────────────────────────────────
    # A 7-day trailing average of units sold smooths out lumpy day-to-day noise
    # (the data is generated with Negative-Binomial dispersion, so single days
    # are spiky). CRITICAL ANTI-LEAKAGE DETAIL: we shift(1) FIRST so the window
    # covers the 7 days *before* today and never includes today's units_sold
    # (which is the target). Including today would leak the answer.
    df["rolling_7d_avg"] = (
        df.groupby("sku_id")["units_sold"]
          .transform(lambda s: s.shift(1).rolling(window=7, min_periods=1).mean())
    )

    # ── 4d. days_since_launch — FRESHNESS DECAY ──────────────────────────────
    # How many days since the SKU launched. New products feel fresh and sell
    # better; appeal decays slowly with age. Computed from launch_date so the
    # feature is explicit and not reliant on any pre-baked column.
    df["days_since_launch"] = (df["date"] - df["launch_date"]).dt.days

    # ── 4e. inventory_ratio — SCARCITY EFFECT ────────────────────────────────
    # inventory_remaining / starting_inventory. As stock dwindles, scarcity can
    # nudge demand and certainly constrains supply. `starting_inventory` is not
    # stored in any file, so we reconstruct it from data: because inventory only
    # ever decreases, (inventory_remaining + units_sold) on day 1 equals the
    # opening stock, and its MAX over a SKU's life recovers that opening value.
    start_inv = (
        daily_sales_raw.assign(_si=daily_sales_raw["inventory_remaining"]
                               + daily_sales_raw["units_sold"])
        .groupby("sku_id")["_si"].max()
        .rename("starting_inventory")
    )
    df = df.merge(start_inv, on="sku_id", how="left")
    df["inventory_ratio"] = df["inventory_remaining"] / df["starting_inventory"]

    # ── 4f. discount_pct — CLEAN PRICE-ELASTICITY SIGNAL ─────────────────────
    # (base_price - current_price) / base_price. Here `original_price` is the
    # base/list price. This gives a normalized 0..1 discount depth that is a far
    # cleaner elasticity signal than raw price (it is comparable across SKUs and
    # price tiers). 0 = full price, 0.4 = 40% off.
    df["discount_pct"] = (df["original_price"] - df["current_price"]) / df["original_price"]

    # ── 4g. days_to_season_end — URGENCY (kept from calendar) ────────────────
    # Already present from the calendar join; we simply keep it. Fewer days left
    # in the season => stronger pressure to clear stock (and stronger urgency
    # multiplier in the demand process).
    # (no transformation needed)

    print("\nEngineered features: lag_7_units_sold, lag_14_units_sold, "
          "rolling_7d_avg, days_since_launch, inventory_ratio, discount_pct")
    print("Kept from calendar: days_to_season_end")
    return df


def encode_and_select(df):
    """Steps 5–7 — One-hot encode, keep-as-is, drop leakage, select final cols."""

    # ── Step 5. One-hot encode category and day_of_week ──────────────────────
    # These are nominal categoricals with no inherent order, so one-hot encoding
    # (one 0/1 column per level) lets a model use them without inventing a fake
    # ranking. We keep all levels (drop_first=False) for interpretability.
    df = pd.get_dummies(df, columns=["category", "day_of_week"],
                        prefix=["category", "dow"], dtype=int)

    # ── Step 6. Keep season, is_holiday, markdown_stage AS-IS ────────────────
    # • season       : low-cardinality string kept verbatim (can be encoded later).
    # • is_holiday   : already a clean 0/1 flag.
    # • markdown_stage: an ordinal 0..5 stage that is meaningful as a number.
    # Nothing to do — they pass through untouched.

    # ── Step 7. Drop LEAKAGE columns ─────────────────────────────────────────
    # These secretly encode the answer and must never be features:
    #   • latent_demand        : the TRUE demand the simulator drew units from —
    #                            essentially the target with the noise removed.
    #   • markdown_depth       : derived directly from price depth at sale time
    #                            (not present in this dataset, dropped safely).
    #   • days_at_current_price: used inside the demand-generating "fatigue"
    #                            driver, so it co-moves with the target in a way
    #                            we could not legitimately exploit pre-sale.
    leakage_cols = ["latent_demand", "markdown_depth", "days_at_current_price"]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

    # ── Final column selection ───────────────────────────────────────────────
    # Keep identifiers (for traceability), the target, the engineered features,
    # the as-is keepers, and the one-hot columns. We deliberately drop raw
    # helper/intermediate columns that were only needed to build features or
    # that are post-filter constant / non-predictive:
    #   original_price, current_price, inventory_remaining, starting_inventory
    #   (used to build discount_pct / inventory_ratio), plus day_number,
    #   holiday_name, is_weekend, stockout_flag, post_stockout_flag,
    #   popularity_segment, base_demand, price_tier, launch_date.
    keep_base = ["sku_id", "date", "units_sold",
                 "lag_7_units_sold", "lag_14_units_sold", "rolling_7d_avg",
                 "days_since_launch", "inventory_ratio", "discount_pct",
                 "days_to_season_end",
                 "season", "is_holiday", "markdown_stage"]
    onehot_cols = [c for c in df.columns
                   if c.startswith("category_") or c.startswith("dow_")]
    final_cols = keep_base + onehot_cols
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols]

    print("\nOne-hot encoded: category, day_of_week")
    print("Kept as-is: season, is_holiday, markdown_stage")
    print("Dropped leakage: latent_demand, markdown_depth, days_at_current_price")
    return df


def print_validation_summary(df):
    """Step 9 — Print a validation summary so we can sanity-check the output."""
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY — data/modeling_table.csv")
    print("=" * 70)

    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    print(f"\nColumns ({len(df.columns)}):")
    for c in df.columns:
        print(f"  - {c} [{df[c].dtype}]")

    print("\nNull counts (only columns with nulls shown):")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls) == 0:
        print("  (no nulls)")
    else:
        for c, n in nulls.items():
            print(f"  - {c}: {n:,} ({n / len(df):.2%})  "
                  f"[expected for early-life lag/rolling rows]")

    print("\nTarget (units_sold) describe:")
    print(df["units_sold"].describe().round(2).to_string())

    print("\nSample rows (5 random):")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.sample(5, random_state=42).to_string(index=False))


def main():
    # Step 1
    sku_master, calendar, price_schedule, daily_sales = load_raw_tables()
    # Keep a pristine copy of daily_sales for the starting-inventory reconstruction
    daily_sales_raw = daily_sales.copy()
    # Step 2
    df = join_tables(sku_master, calendar, price_schedule, daily_sales)
    # Step 3 (filter BEFORE feature engineering)
    df = filter_post_stockout(df)
    # Step 4
    df = engineer_features(df, daily_sales_raw)
    # Steps 5-7
    df = encode_and_select(df)
    # Step 8 — save
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved modeling table -> {OUTPUT_PATH}")
    # Step 9 — validate
    print_validation_summary(df)


if __name__ == "__main__":
    main()
