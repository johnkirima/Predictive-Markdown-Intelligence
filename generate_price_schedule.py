# generate_price_schedule.py
# Builds the price schedule table — one row per SKU per day
# Decides when each SKU gets marked down and by how much
# Rules: prices only go down, max 5 markdown stages per SKU per season

import pandas as pd
import numpy as np
from config import (
    MARKDOWN_STAGES,
    START_DATE,
    END_DATE,
    RANDOM_SEED
)


def generate_price_schedule(sku_master, calendar):

    rng = np.random.default_rng(RANDOM_SEED)

    all_records = []

    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="D")

    for _, sku_row in sku_master.iterrows():

        sku_id         = sku_row["sku_id"]
        original_price = sku_row["original_price"]
        launch_date    = pd.Timestamp(sku_row["launch_date"])

        # Track current markdown stage index (starts at 0 = full price)
        current_stage = 0
        current_price = original_price
        days_at_price = 0

        for date in dates:

            # SKU has not launched yet — skip
            if date < launch_date:
                continue

            # ── Decide whether to markdown today ──────────────────────────────
            # Only markdown if:
            # 1. We are not already at the deepest stage (50% off)
            # 2. SKU has been at current price for at least 14 days
            # 3. Random chance fires (10% probability per day)

            if (
                current_stage < len(MARKDOWN_STAGES) - 1
                and days_at_price >= 21
                and rng.random() < 0.04
            ):
                current_stage += 1
                current_price  = round(original_price * (1 - MARKDOWN_STAGES[current_stage]), 2)
                days_at_price  = 0

            else:
                days_at_price += 1

            all_records.append({
                "sku_id":                sku_id,
                "date":                  date,
                "original_price":        original_price,
                "current_price":         current_price,
                "markdown_stage":        current_stage,
                "days_at_current_price": days_at_price
            })

    price_schedule = pd.DataFrame(all_records)
    return price_schedule


if __name__ == "__main__":
    sku_master = pd.read_csv("data/sku_master.csv", parse_dates=["launch_date"])
    calendar   = pd.read_csv("data/calendar.csv",   parse_dates=["date"])

    df = generate_price_schedule(sku_master, calendar)
    df.to_csv("data/price_schedule.csv", index=False)

  

    print(df.head(15))
    print(f"\nShape: {df.shape}")
    print(f"\nMarkdown stage breakdown:\n{df['markdown_stage'].value_counts().sort_index()}")
    print(f"\nSample SKU price journey:")
    print(df[df["sku_id"] == "SKU_0005"][["date", "current_price", "markdown_stage", "days_at_current_price"]].head(30))