# generate_calendar.py
# Builds the calendar table — one row per date across the simulation window
# Every other table joins to this to get season info, holidays, day of week

import pandas as pd
from config import START_DATE, END_DATE


# ── US Retail Holidays ─────────────────────────────────────────────────────────
HOLIDAYS = {
    # 2022
    "2022-01-01": "New Year's Day",
    "2022-01-17": "MLK Day",
    "2022-02-14": "Valentine's Day",
    "2022-02-20": "Presidents Day",
    "2022-05-08": "Mother's Day",
    "2022-05-30": "Memorial Day",
    "2022-06-19": "Juneteenth / Father's Day",
    "2022-07-04": "Independence Day",
    "2022-08-15": "Back to School",
    "2022-09-05": "Labor Day",
    "2022-10-31": "Halloween",
    "2022-11-25": "Thanksgiving",
    "2022-11-26": "Black Friday",
    "2022-12-24": "Christmas Eve",
    "2022-12-25": "Christmas Day",
    "2022-12-31": "New Year's Eve",

    # 2023
    "2023-01-01": "New Year's Day",
    "2023-01-16": "MLK Day",
    "2023-02-14": "Valentine's Day",
    "2023-02-19": "Presidents Day",
    "2023-05-14": "Mother's Day",
    "2023-05-29": "Memorial Day",
    "2023-06-18": "Father's Day",
    "2023-06-19": "Juneteenth",
    "2023-07-04": "Independence Day",
    "2023-08-15": "Back to School",
    "2023-09-04": "Labor Day",
    "2023-10-31": "Halloween",
    "2023-11-23": "Thanksgiving",
    "2023-11-24": "Black Friday",
    "2023-12-24": "Christmas Eve",
    "2023-12-25": "Christmas Day",
    "2023-12-31": "New Year's Eve",
}

# ── Season definitions (month ranges) ─────────────────────────────────────────
def get_season(month):
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "fall"

# ── Season end dates (last day of each season) ────────────────────────────────
SEASON_END_DATES = {
    "winter": [28, 2],   # Feb 28
    "spring": [31, 5],   # May 31
    "summer": [31, 8],   # Aug 31
    "fall":   [30, 11],  # Nov 30
}


def generate_calendar():

    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="D")

    records = []

    for date in dates:

        # ── Basic date fields ──────────────────────────────────────────────────
        day_of_week = date.strftime("%A")
        is_weekend  = 1 if day_of_week in ["Saturday", "Sunday"] else 0

        # ── Holiday fields ─────────────────────────────────────────────────────
        date_str     = date.strftime("%Y-%m-%d")
        is_holiday   = 1 if date_str in HOLIDAYS else 0
        holiday_name = HOLIDAYS.get(date_str, None)

        # ── Season fields ──────────────────────────────────────────────────────
        season = get_season(date.month)

        # ── Days to season end ─────────────────────────────────────────────────
        end_day, end_month = SEASON_END_DATES[season]
        end_year = date.year if date.month <= end_month else date.year + 1
        season_end = pd.Timestamp(year=end_year, month=end_month, day=end_day)
        days_to_season_end = (season_end - date).days

        records.append({
            "date":               date,
            "day_of_week":        day_of_week,
            "is_weekend":         is_weekend,
            "is_holiday":         is_holiday,
            "holiday_name":       holiday_name,
            "season":             season,
            "days_to_season_end": days_to_season_end
        })

    calendar = pd.DataFrame(records)
    return calendar


if __name__ == "__main__":
    df = generate_calendar()
    df.to_csv("data/calendar.csv", index=False)
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"\nSeason breakdown:\n{df['season'].value_counts()}")
    print(f"\nHolidays:\n{df[df['is_holiday'] == 1][['date', 'holiday_name']]}")