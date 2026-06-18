# generate_sku_master.py
# Builds the SKU master table — one row per product
# This is the foundation. Every other table references sku_id.

import pandas as pd
import numpy as np
from config import (
    categories,
    popularity_segments,
    N_SKUS,
    base_daily_demand,
    PRICE_TIERS,
    RANDOM_SEED
)


def generate_sku_master():

    rng = np.random.default_rng(RANDOM_SEED)

    # ── Step 1: Assign popularity segments ────────────────────────────────────
    segments = list(popularity_segments.keys())    # ["winner", "normal", "dead"]
    weights  = list(popularity_segments.values())  # [0.20, 0.50, 0.30]

    popularity = rng.choice(segments, size=N_SKUS, p=weights)

    # ── Step 2: Assign categories ─────────────────────────────────────────────
    category = rng.choice(categories, size=N_SKUS)

    # ── Step 3: Assign price tiers ────────────────────────────────────────────
    tiers = ["budget", "mid", "premium"]
    price_tier = rng.choice(tiers, size=N_SKUS)

    # ── Step 4: Look up original price from PRICE_TIERS ───────────────────────
    original_price = [
        PRICE_TIERS[cat][tier]
        for cat, tier in zip(category, price_tier)
    ]

    # ── Step 5: Look up base demand from base_daily_demand ────────────────────
    base_demand = [base_daily_demand[seg] for seg in popularity]

    # ── Step 6: Spread launch dates across first half of simulation ───────────
    launch_dates = pd.date_range(start="2022-01-01", end="2022-06-30", periods=N_SKUS)
    launch_dates = launch_dates.normalize()

    # ── Step 7: Build the dataframe ───────────────────────────────────────────
    sku_master = pd.DataFrame({
        "sku_id":             [f"SKU_{str(i).zfill(4)}" for i in range(1, N_SKUS + 1)],
        "category":           category,
        "price_tier":         price_tier,
        "original_price":     original_price,
        "popularity_segment": popularity,
        "base_demand":        base_demand,
        "launch_date":        launch_dates
    })

    return sku_master


if __name__ == "__main__":
    df = generate_sku_master()
    df.to_csv("data/sku_master.csv", index=False)
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"\nPopularity breakdown:\n{df['popularity_segment'].value_counts()}")
    print(f"\nCategory breakdown:\n{df['category'].value_counts()}")