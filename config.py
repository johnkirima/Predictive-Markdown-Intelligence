#Every magic number in this project lives here

import numpy as np

#NOW LET US DO THE SKU setup
categories = ['tshirts', 'jeans', 'jackets', 'shoes']

popularity_segments = {'winner': 0.20, 'normal': 0.50, 'dead': 0.30} # 20% of SKUs are winners, 50% are normal, 30% are dead 

#now let us have the number of SKUs
N_SKUS = 500

#now let us have driver 1
#DRIVER 1: BASE DAILY DEMAND

base_daily_demand = {'winner':40, 'normal':15, 'dead':3}


# - DRIVER 2+3: PRICE ELASTICITY BY CATEGORY
# Higher = more sensitive to price changes
# Formula: price_effect = (current_price / original_price) ** (-elasticity)
category_elasticity = {'tshirts':2.0, 'jeans': 1.2, 'jackets': 0.8, 'shoes': 1.5}

# ── DRIVER 4: SEASON URGENCY BY CATEGORY ──────────────────────────────────────
# Higher alpha = urgency multiplier grows faster near season end
# Formula: season_urgency = 1 + alpha * (1 - season_fraction) ** 2

category_urgency_alpha = {'jackets': 1.5, 'shoes': 0.8, 'jeans': 0.4, 'tshirts': 0.3}

# ── DRIVER 5: SCARCITY BOOST ───────────────────────────────────────────────────
# Small lift as inventory depletes
# Formula: scarcity_boost = 1 + SCARCITY_BETA * (1 - inventory_ratio)

scarcity_beta = 0.3


# ── DRIVER 6: MARKDOWN FATIGUE ────────────────────────────────────────────────
# Discount lift decays the longer price stays the same
# Formula: fatigue_decay = exp(-FATIGUE_LAMBDA * days_at_current_price)
fatigue_lambda = 0.05

# # ── DRIVER 7: DAY OF WEEK MULTIPLIERS ─────────────────────────────────────────
day_of_week_multipliers = {
    'Monday': 0.85,
    'Tuesday': 0.85,
    'Wednesday': 0.95,
    'Thursday': 1.00,
    'Friday': 1.10,
    'Saturday': 1.25,
    'Sunday': 1.20
}

# ── DRIVER 8: HOLIDAY EFFECT ──────────────────────────────────────────────────
# Multiplicative spike — stronger when combined with deep discounts
# Formula: holiday_multiplier = 1 + HOLIDAY_GAMMA * is_holiday * (1 + markdown_depth)
holiday_gamma = 0.5

# ── DRIVER 9: LIFECYCLE DECAY ─────────────────────────────────────────────────
# New products feel fresh — old products lose appeal slowly
# Formula: lifecycle_multiplier = 0.7 + 0.3 * exp(-LIFECYCLE_DELTA * product_age_weeks)
lifecycle_delta = 0.001

# ── DRIVER 10: NOISE ──────────────────────────────────────────────────────────
# Negative Binomial dispersion — lower = lumpier daily sales
dispersion = 5

# ── MARKDOWN STAGES ───────────────────────────────────────────────────────────
# Prices only go down. Max 4 stages per SKU per season.

MARKDOWN_STAGES = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]

# ── PRICE TIERS BY CATEGORY (original_price ranges in USD) ───────────────────

PRICE_TIERS = {
    "tshirts": {"budget": 15,  "mid": 30,  "premium": 55},
    "jeans":   {"budget": 40,  "mid": 70,  "premium": 120},
    "jackets": {"budget": 60,  "mid": 120, "premium": 220},
    "shoes":   {"budget": 50,  "mid": 95,  "premium": 180}
}

# ── SIMULATION PERIOD ─────────────────────────────────────────────────────────

START_DATE = "2022-01-01"
END_DATE   = "2023-12-31"

# ── RANDOM SEED (reproducibility) ─────────────────────────────────────────────

RANDOM_SEED = 42