# visualize_data.py  ── Clean Story Charts
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

# ── Load data ──────────────────────────────────────────────────────────────
daily  = pd.read_csv("data/daily_sales.csv",  parse_dates=["date"])
skus   = pd.read_csv("data/sku_master.csv")
cal    = pd.read_csv("data/calendar.csv",     parse_dates=["date"])
prices = pd.read_csv("data/price_schedule.csv", parse_dates=["date"])

# Keep only pre-stockout rows for clean demand signal
clean  = daily[daily["post_stockout_flag"] == 0].copy()

COLORS = {"winner": "#27ae60", "normal": "#2980b9", "dead": "#c0392b"}
plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "#fafafa"})

# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — "What does demand look like for each type of SKU?"
# ══════════════════════════════════════════════════════════════════════════════
winner_id = skus[skus["popularity_segment"] == "winner"]["sku_id"].iloc[2]
normal_id = skus[skus["popularity_segment"] == "normal"]["sku_id"].iloc[2]
dead_id   = skus[skus["popularity_segment"] == "dead"]["sku_id"].iloc[2]

fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
fig.suptitle("Chart 1 — Daily Demand by SKU Type\n(pre-stockout rows only)",
             fontsize=13, fontweight="bold", y=1.01)

for ax, seg, sku_id in zip(axes,
                            ["winner", "normal", "dead"],
                            [winner_id, normal_id, dead_id]):
    df = clean[clean["sku_id"] == sku_id].copy()
    # 7-day rolling average for readability
    df["rolling"] = df["latent_demand"].rolling(7, min_periods=1).mean()
    ax.fill_between(df["date"], df["latent_demand"],
                    alpha=0.15, color=COLORS[seg])
    ax.plot(df["date"], df["rolling"],
            color=COLORS[seg], linewidth=1.8, label="7-day avg")
    ax.axhline(df["latent_demand"].mean(), color="black",
               linestyle="--", linewidth=0.8, alpha=0.5, label="Overall mean")
    mean_val = df["latent_demand"].mean()
    ax.set_ylabel("Units / day", fontsize=9)
    ax.set_title(
        f"{seg.upper()}  |  {sku_id}  |  avg {mean_val:.1f} units/day",
        fontsize=10, color=COLORS[seg], fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")

axes[-1].set_xlabel("Date")
plt.tight_layout()
plt.savefig("data/chart1_demand_by_segment.png", dpi=150, bbox_inches="tight")
plt.show()
print("Chart 1 saved.")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — "How does price markdown drive demand? (elasticity in action)"
# ══════════════════════════════════════════════════════════════════════════════
w = clean[clean["sku_id"] == winner_id].copy()
w["rolling_units"] = w["units_sold"].rolling(7, min_periods=1).mean()

fig, ax1 = plt.subplots(figsize=(15, 5))
ax2 = ax1.twinx()

ax1.plot(w["date"], w["current_price"],
         color="#8e44ad", linewidth=2, label="Current Price ($)", zorder=3)
ax2.bar(w["date"], w["rolling_units"],
        color="#27ae60", alpha=0.4, width=1, label="Units Sold (7d avg)")

ax1.set_ylabel("Price ($)", color="#8e44ad", fontsize=10)
ax2.set_ylabel("Units Sold", color="#27ae60", fontsize=10)
ax1.set_title(
    f"Chart 2 — Price vs Units Sold  |  {winner_id} (Winner)\n"
    "When price drops → units sold rises = elasticity working",
    fontsize=12, fontweight="bold")

lines1, lab1 = ax1.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig("data/chart2_price_vs_demand.png", dpi=150, bbox_inches="tight")
plt.show()
print("Chart 2 saved.")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — "What is stockout censoring? Latent vs Actual"
# ══════════════════════════════════════════════════════════════════════════════
# Pick a normal SKU that actually stocked out
stocked_out_skus = (daily[daily["stockout_flag"] == 1]["sku_id"].value_counts())
if len(stocked_out_skus):
    so_sku = stocked_out_skus.index[0]
    df_so  = daily[daily["sku_id"] == so_sku].copy()

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df_so["date"], df_so["latent_demand"],
            color="#2980b9", linewidth=1.5, label="Latent Demand (true want)")
    ax.plot(df_so["date"], df_so["units_sold"],
            color="#e67e22", linewidth=1.5, label="Units Sold (what we observe)")
    ax.fill_between(df_so["date"],
                    df_so["units_sold"], df_so["latent_demand"],
                    where=(df_so["units_sold"] < df_so["latent_demand"]),
                    alpha=0.25, color="red", label="Hidden demand gap")

    first_so = df_so[df_so["stockout_flag"] == 1]["date"].min()
    ax.axvline(first_so, color="red", linestyle="--", linewidth=1.5,
               label=f"First Stockout: {first_so.date()}")
    ax.annotate("← Model must NOT train on\nrows after this line",
                xy=(first_so, ax.get_ylim()[1] * 0.6),
                xytext=(first_so + pd.Timedelta(days=30),
                        ax.get_ylim()[1] * 0.7),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=9, color="red")

    ax.set_title(
        f"Chart 3 — Stockout Censoring  |  {so_sku}\n"
        "Units sold flatlines at 0 after stockout — but real demand continued",
        fontsize=12, fontweight="bold")
    ax.set_ylabel("Units")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("data/chart3_censoring.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Chart 3 saved.")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 — "Dataset summary — what is in our tables?"
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Chart 4 — Dataset Summary Across All Tables",
             fontsize=13, fontweight="bold")

# 4a: SKU distribution
seg_counts = skus["popularity_segment"].value_counts().reindex(["winner","normal","dead"])
axes[0,0].bar(seg_counts.index, seg_counts.values,
              color=[COLORS[s] for s in seg_counts.index], edgecolor="white")
for i, v in enumerate(seg_counts.values):
    axes[0,0].text(i, v + 1, str(v), ha="center", fontsize=10, fontweight="bold")
axes[0,0].set_title("SKU Master — SKUs by Segment", fontweight="bold")
axes[0,0].set_ylabel("Count")

# 4b: Category distribution
cat_counts = skus["category"].value_counts()
axes[0,1].barh(cat_counts.index, cat_counts.values, color="#3498db", edgecolor="white")
axes[0,1].set_title("SKU Master — SKUs by Category", fontweight="bold")
axes[0,1].set_xlabel("Count")

# 4c: Avg daily units sold by segment and markdown stage
pivot = (clean.groupby(["markdown_stage","popularity_segment"])["units_sold"]
         .mean().unstack())
for seg in ["winner","normal","dead"]:
    if seg in pivot.columns:
        axes[1,0].plot(pivot.index, pivot[seg],
                       marker="o", label=seg, color=COLORS[seg], linewidth=2)
axes[1,0].set_title("Daily Sales — Avg Units Sold by Markdown Stage",
                    fontweight="bold")
axes[1,0].set_xlabel("Markdown Stage (0=full price → 5=deepest discount)")
axes[1,0].set_ylabel("Avg Units Sold/day")
axes[1,0].legend(fontsize=9)

# 4d: Stockout rate by segment
so_rate = daily.groupby("popularity_segment")["stockout_flag"].mean() * 100
so_rate = so_rate.reindex(["winner","normal","dead"])
bars = axes[1,1].bar(so_rate.index, so_rate.values,
                     color=[COLORS[s] for s in so_rate.index], edgecolor="white")
for bar, v in zip(bars, so_rate.values):
    axes[1,1].text(bar.get_x() + bar.get_width()/2, v + 0.3,
                   f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
axes[1,1].set_title("Daily Sales — Stockout Rate by Segment", fontweight="bold")
axes[1,1].set_ylabel("Stockout %")
axes[1,1].axhline(18.16, color="black", linestyle="--",
                  linewidth=0.8, alpha=0.5, label="Overall 18.16%")
axes[1,1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("data/chart4_summary.png", dpi=150, bbox_inches="tight")
plt.show()
print("Chart 4 saved.")
print("\n✅ All 4 charts saved to data/")