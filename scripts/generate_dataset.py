"""
generate_dataset.py
====================
Generates a realistic synthetic dataset for Delhi NCR ride-hailing (Ola/Uber-style)
covering Jan–Dec 2023.

Realism anchors:
  - 8 dispatch zones mirroring real Delhi NCR hubs
  - Indian public holidays & cultural events (Holi, Diwali, IPL finals, etc.)
  - Delhi winter fog events (Jan–Feb) causing demand crashes
  - Monsoon period (Jul–Sep) slight demand suppression
  - Weekly weekday/weekend patterns (Mon–Thu peak, Fri–Sat nightlife surge)
  - Morning rush (8–10 AM) and evening rush (6–9 PM) intraday spikes
  - Gradual growth trend over the year (+18% YoY estimate)

Data source note:
  No public Ola/Uber trip dataset is released for India. This dataset is
  synthesized using real operational patterns from:
    - Delhi IIIT-D Open Transit Data (otd.delhi.gov.in)
    - RedSeer Consulting India Mobility Report 2023
    - Press reports on NCR ride volumes (Indian Express, Mint, ET)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

np.random.seed(42)

# ──────────────────────────────────────────────────────────
# 1.  ZONE DEFINITIONS
# ──────────────────────────────────────────────────────────
ZONES = {
    "Z01_CP_NewDelhi":       {"tier": "mega",   "base_vehicles": 3800, "base_trips": 32000},
    "Z02_Gurugram_Cyber":    {"tier": "mega",   "base_vehicles": 3200, "base_trips": 27000},
    "Z03_Noida_Sector18":    {"tier": "high",   "base_vehicles": 1900, "base_trips": 16000},
    "Z04_Dwarka_Airport":    {"tier": "high",   "base_vehicles": 1600, "base_trips": 13500},
    "Z05_Lajpat_SouthDelhi": {"tier": "medium", "base_vehicles": 950,  "base_trips": 8000},
    "Z06_Rohini_NorthWest":  {"tier": "medium", "base_vehicles": 780,  "base_trips": 6500},
    "Z07_Faridabad":         {"tier": "low",    "base_vehicles": 420,  "base_trips": 3200},
    "Z08_Ghaziabad":         {"tier": "low",    "base_vehicles": 380,  "base_trips": 2900},
}

# ──────────────────────────────────────────────────────────
# 2.  EVENT CALENDAR  (date → multiplier)
# ──────────────────────────────────────────────────────────
EVENTS = {
    # Fog / weather crashes
    "2023-01-07": ("Delhi_Winter_Fog_Severe",    0.45),
    "2023-01-11": ("Delhi_Winter_Fog_Moderate",  0.62),
    "2023-01-19": ("Delhi_Winter_Fog_Severe",    0.48),
    "2023-02-03": ("Delhi_Winter_Fog_Light",     0.71),
    "2023-07-09": ("Monsoon_Flooding_North",     0.68),
    "2023-08-24": ("Monsoon_Heavy_Citywide",     0.61),
    # Demand spikes — festivals & events
    "2023-03-08": ("Holi_Eve",                   1.72),
    "2023-03-09": ("Holi_Day",                   0.52),   # streets empty
    "2023-04-14": ("Baisakhi_IPL_Opening",       1.55),
    "2023-05-29": ("IPL_Final_Ahmedabad_spillover", 1.38),
    "2023-08-15": ("Independence_Day",           1.41),
    "2023-10-24": ("Diwali_Eve",                 1.89),
    "2023-10-25": ("Diwali_Day",                 0.44),   # roads empty
    "2023-11-12": ("Diwali_Chhath_Puja_peak",    1.62),
    "2023-12-24": ("Christmas_Eve",              1.51),
    "2023-12-31": ("New_Year_Eve",               1.95),
}

# ──────────────────────────────────────────────────────────
# 3.  HELPER: daily multiplier
# ──────────────────────────────────────────────────────────
def daily_multiplier(date):
    dow = date.weekday()   # 0=Mon
    doy = date.timetuple().tm_yday

    # Trend: ~18% growth over the year, linear
    trend = 1.0 + 0.18 * (doy / 365)

    # Weekly pattern
    week_pattern = {0: 0.92, 1: 0.94, 2: 0.96, 3: 0.97,
                    4: 1.08, 5: 1.18, 6: 1.03}
    week_mult = week_pattern[dow]

    # Seasonal: monsoon suppression
    month = date.month
    seasonal = 1.0
    if month in [7, 8]:
        seasonal = 0.88
    elif month in [1, 2]:
        seasonal = 0.93   # fog season general suppression
    elif month in [10, 11]:
        seasonal = 1.06   # festive season

    event_key = date.strftime("%Y-%m-%d")
    event_mult = EVENTS.get(event_key, (None, 1.0))[1]

    return trend * week_mult * seasonal * event_mult


# ──────────────────────────────────────────────────────────
# 4.  GENERATE RECORDS
# ──────────────────────────────────────────────────────────
def generate_dataset(year=2023):
    records = []
    start = datetime(year, 1, 1)
    end   = datetime(year, 12, 31)
    delta = timedelta(days=1)
    current = start

    while current <= end:
        dm = daily_multiplier(current)
        event_name = EVENTS.get(current.strftime("%Y-%m-%d"), (None, 1.0))[0]

        for zone_id, zinfo in ZONES.items():
            base_v = zinfo["base_vehicles"]
            base_t = zinfo["base_trips"]

            # Vehicles active that day
            vehicles = int(base_v * dm * np.random.uniform(0.94, 1.06))
            vehicles = max(50, vehicles)

            # Trips: correlated with vehicles + noise
            trips_per_vehicle = np.random.normal(8.5, 1.2)
            trips = int(vehicles * trips_per_vehicle * np.random.uniform(0.90, 1.10))
            trips = max(100, trips)

            # Trips-per-vehicle ratio (efficiency metric)
            tpv = round(trips / vehicles, 2)

            records.append({
                "dispatching_zone_id": zone_id,
                "zone_tier":           zinfo["tier"],
                "date":                current.strftime("%Y-%m-%d"),
                "month":               current.month,
                "day_of_week":         current.strftime("%A"),
                "is_weekend":          int(current.weekday() >= 5),
                "active_vehicles":     vehicles,
                "trips":               trips,
                "trips_per_vehicle":   tpv,
                "event_label":         event_name if event_name else "normal",
                "daily_multiplier":    round(dm, 4),
            })
        current += delta

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)

    df = generate_dataset(2023)
    out_path = os.path.join(out_dir, "delhi_ncr_fleet_2023.csv")
    df.to_csv(out_path, index=False)

    print(f"✅  Dataset saved → {out_path}")
    print(f"    Shape        : {df.shape}")
    print(f"    Date range   : {df['date'].min()} → {df['date'].max()}")
    print(f"    Zones        : {df['dispatching_zone_id'].nunique()}")
    print(f"    Total records: {len(df):,}")
    print("\nSample rows:")
    print(df.head(8).to_string(index=False))
