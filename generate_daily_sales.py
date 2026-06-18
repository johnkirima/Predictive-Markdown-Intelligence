# generate_daily_sales.py

import pandas as pd
import numpy as np
from config import (
    category_elasticity,
    category_urgency_alpha,
    scarcity_beta,
    fatigue_lambda,
    day_of_week_multipliers,
    holiday_gamma,
    lifecycle_delta,
    dispersion,
    RANDOM_SEED
)


def compute_latent_demand(row, rng):

    base        = row["base_demand"]
    category    = row["category"]
    segment     = row["popularity_segment"]
    orig_price  = row["original_price"]
    curr_price  = row["current_price"]
    stage       = row["markdown_stage"]
    days_at_p   = row["days_at_current_price"]
    days_to_end = row["days_to_season_end"]
    is_holiday  = row["is_holiday"]
    dow         = row["day_of_week"]
    day_number  = row["day_number"]

    # Driver 1: Price Elasticity
    elasticity   = category_elasticity[category]
    discount_pct = (orig_price - curr_price) / orig_price
    price_effect = 1 + (elasticity * discount_pct)

    # Driver 2: Season Urgency
    alpha           = category_urgency_alpha[category]
    season_fraction = max(days_to_end / 90, 0)
    urgency         = 1 + alpha * (1 - season_fraction) ** 2

    # Driver 3: Markdown Fatigue
    fatigue = np.exp(-fatigue_lambda * days_at_p)

    # Driver 4: Scarcity Boost
    scarcity = (1 + scarcity_beta) if (segment == "winner" and stage >= 3) else 1.0

    # Driver 5: Day of Week
    dow_effect = day_of_week_multipliers[dow]

    # Driver 6: Holiday Effect
    discount_depth = (orig_price - curr_price) / orig_price
    holiday_effect = 1 + holiday_gamma * is_holiday * (1 + discount_depth)

    # Driver 7: Lifecycle Decay
    decay = np.exp(-lifecycle_delta * day_number)

    # Combine
    latent = base * price_effect * urgency * fatigue * scarcity * dow_effect * holiday_effect * decay
    latent = max(latent, 0)

    # Negative Binomial noise
    n          = dispersion
    p          = n / (n + latent + 1e-6)
    units_sold = rng.negative_binomial(n=n, p=p)

    return round(latent, 2), int(units_sold)


def generate_daily_sales(sku_master, calendar, price_schedule):

    rng = np.random.default_rng(RANDOM_SEED)

    # Drop duplicate original_price from sku_master before merging
    sku_clean = sku_master.drop(columns=["original_price"], errors="ignore")

    df = price_schedule.merge(sku_clean, on="sku_id", how="left")
    df = df.merge(calendar, on="date", how="left")

    df["launch_date"] = pd.to_datetime(df["launch_date"])
    df["date"]        = pd.to_datetime(df["date"])
    df["day_number"]  = (df["date"] - df["launch_date"]).dt.days

    inventory_by_segment = {"winner": 11000, "normal": 2700, "dead": 450}
    all_records = []

    for sku_id, sku_df in df.groupby("sku_id"):

        inventory = inventory_by_segment[sku_df["popularity_segment"].iloc[0]]
        post_stockout = False

        for _, row in sku_df.iterrows():

            if inventory <= 0:
                post_stockout = True

            latent_demand, units_sold = compute_latent_demand(row, rng)

            units_sold    = min(units_sold, inventory)
            stockout_flag = 1 if inventory <= 0 else 0
            inventory     = max(inventory - units_sold, 0)

            all_records.append({
                "sku_id":                sku_id,
                "date":                  row["date"],
                "category":              row["category"],
                "popularity_segment":    row["popularity_segment"],
                "original_price":        row["original_price"],
                "current_price":         row["current_price"],
                "markdown_stage":        row["markdown_stage"],
                "days_at_current_price": row["days_at_current_price"],
                "season":                row["season"],
                "days_to_season_end":    row["days_to_season_end"],
                "is_holiday":            row["is_holiday"],
                "holiday_name":          row["holiday_name"],
                "day_of_week":           row["day_of_week"],
                "is_weekend":            row["is_weekend"],
                "day_number":            row["day_number"],
                "latent_demand":         latent_demand,
                "units_sold":            units_sold,
                "inventory_remaining":   inventory,
                "stockout_flag":         stockout_flag,
                "post_stockout_flag":    1 if post_stockout else 0
            })

    return pd.DataFrame(all_records)


if __name__ == "__main__":
    sku_master     = pd.read_csv("data/sku_master.csv",     parse_dates=["launch_date"])
    calendar       = pd.read_csv("data/calendar.csv",       parse_dates=["date"])
    price_schedule = pd.read_csv("data/price_schedule.csv", parse_dates=["date"])

    df = generate_daily_sales(sku_master, calendar, price_schedule)
    df.to_csv("data/daily_sales.csv", index=False)

    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"\nStockout rate: {df['stockout_flag'].mean():.2%}")
    print(f"\nPost-stockout rows: {df['post_stockout_flag'].sum()}")
    print(f"\nAvg latent demand by segment:")
    print(df.groupby("popularity_segment")["latent_demand"].mean().round(2))
    print(f"\nAvg units sold by segment:")
    print(df.groupby("popularity_segment")["units_sold"].mean().round(2))