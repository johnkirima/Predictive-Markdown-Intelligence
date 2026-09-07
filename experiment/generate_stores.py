# experiment/generate_stores.py
# ----------------------------------------------------------------------------
# Builds the STORE DIMENSION (dim_stores) for the markdown A/B test + DiD layer.
# One row per physical store. 100 stores total.
#
# WHY THIS SCRIPT EXISTS — two experiment-design gaps are addressed HERE:
#
#   Gap 2 (SUTVA / geographic contamination):
#       The Stable Unit Treatment Value Assumption requires that one store's
#       treatment does not affect another store's outcome. If a Treatment store
#       (ML timing) and a Control store (calendar rule) sit in the SAME region,
#       customers can substitute between them, and shared regional inventory or
#       marketing can bleed across conditions — violating SUTVA and biasing the
#       DiD estimate. To make clean separation POSSIBLE downstream, this script
#       assigns every store a `market_region`. Phase 2 (assign_treatment.py) then
#       enforces the rule that no region contains both Treatment and Control
#       stores. Getting regions right here is the prerequisite for that.
#
#   Gap 3 (Demand splitting):
#       Real chains are not made of identical stores. If every store had the same
#       baseline demand, the simulated data would be unrealistically homogeneous
#       and balance/power calculations would be misleading. `capacity_weight`
#       scales each store's baseline demand by tier (Flagship > Standard >
#       Boutique), so total chain demand is split realistically across
#       heterogeneous stores. The fact-table builder (Phase 3) multiplies base
#       SKU demand by this weight.
#
# Output: experiment/data/dim_stores.csv
# ----------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd

# ── Locked configuration ─────────────────────────────────────────────────────
N_STORES = 100
RANDOM_SEED = 42  # reproducibility — matches the base repo's config.RANDOM_SEED

# Four geographic regions. No Treatment+Control mix within a region is enforced
# later in Phase 2 (assign_treatment.py). (Gap 2)
REGIONS = [
    "Great_Lakes_East",
    "Plains_North",
    "Upper_Midwest",
    "Central_Valley",
]

# Store tiers with their target share of the fleet.
TIER_WEIGHTS = {
    "Flagship": 0.20,
    "Standard": 0.60,
    "Boutique": 0.20,
}

# capacity_weight ranges per tier (uniform draw within the range). (Gap 3)
CAPACITY_RANGES = {
    "Flagship": (1.40, 1.80),
    "Standard": (0.85, 1.15),
    "Boutique": (0.40, 0.60),
}


def generate_stores(n_stores: int = N_STORES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate the store dimension table.

    Args:
        n_stores: number of stores to create (default 100).
        seed: random seed for reproducibility (default 42).

    Returns:
        DataFrame with columns:
            store_id, store_name, market_region, store_tier, capacity_weight
    """
    np.random.seed(seed)

    # ── Step 1: Assign store tiers by weight (Flagship 20 / Standard 60 / Boutique 20)
    tiers = list(TIER_WEIGHTS.keys())
    tier_probs = list(TIER_WEIGHTS.values())
    store_tier = np.random.choice(tiers, size=n_stores, p=tier_probs)

    # ── Step 2: Assign a geographic region to each store (Gap 2) ──────────────
    # Even, deterministic spread across the 4 regions so each region has enough
    # stores to later hold an all-Treatment or all-Control block.
    market_region = np.array([REGIONS[i % len(REGIONS)] for i in range(n_stores)])

    # ── Step 3: Draw capacity_weight from the tier-specific range (Gap 3) ─────
    capacity_weight = np.array([
        round(np.random.uniform(*CAPACITY_RANGES[tier]), 3)
        for tier in store_tier
    ])

    # ── Step 4: Build identifiers ─────────────────────────────────────────────
    store_id = [f"STORE_{str(i).zfill(3)}" for i in range(1, n_stores + 1)]
    store_name = [
        f"{region.replace('_', ' ')} {tier} #{i:03d}"
        for i, (region, tier) in enumerate(zip(market_region, store_tier), start=1)
    ]

    # ── Step 5: Assemble the dataframe ────────────────────────────────────────
    stores = pd.DataFrame({
        "store_id":        store_id,
        "store_name":      store_name,
        "market_region":   market_region,
        "store_tier":      store_tier,
        "capacity_weight": capacity_weight,
    })

    return stores


def print_summary(df: pd.DataFrame) -> None:
    """Print a human-readable summary used for a quick sanity check."""
    print("=" * 60)
    print(f"dim_stores generated: {len(df)} stores")
    print("=" * 60)

    print("\nCount by tier:")
    print(df["store_tier"].value_counts().to_string())

    print("\nCount by region:")
    print(df["market_region"].value_counts().to_string())

    print("\ncapacity_weight stats (overall):")
    print(df["capacity_weight"].describe().round(3).to_string())

    print("\ncapacity_weight stats by tier:")
    print(
        df.groupby("store_tier")["capacity_weight"]
        .agg(["count", "min", "mean", "max"])
        .round(3)
        .to_string()
    )
    print()


if __name__ == "__main__":
    df = generate_stores()

    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dim_stores.csv")
    df.to_csv(out_path, index=False)

    print_summary(df)
    print(f"Saved -> {out_path}")
