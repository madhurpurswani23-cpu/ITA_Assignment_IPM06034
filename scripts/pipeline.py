"""
pipeline.py
============
Full hybrid ML pipeline for Delhi NCR ride-hailing demand forecasting.

Pipeline:
  Raw Data → Feature Engineering → Unsupervised Learning (K-Means + Isolation Forest)
           → Anomaly Score & Demand Cluster → Supervised Learning (RF + GBM)
           → Demand Forecast

Run:
    python3 scripts/pipeline.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import (IsolationForest, RandomForestRegressor,
                               GradientBoostingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "delhi_ncr_fleet_2023.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Palette ─────────────────────────────────────────────────────────────────
C_DARK   = "#1C2B3A"
C_RUST   = "#B5451B"
C_SAND   = "#F5F0E8"
C_MID    = "#4A6FA5"
C_LIGHT  = "#A8C5DA"
C_GREEN  = "#2D7D46"
C_WARN   = "#E8A838"
ZONE_COLORS = ["#1C2B3A","#B5451B","#4A6FA5","#2D7D46",
               "#A8C5DA","#E8A838","#7B4F9E","#C45E5E"]

plt.rcParams.update({
    "figure.facecolor": C_SAND,
    "axes.facecolor":   C_SAND,
    "axes.edgecolor":   "#AAAAAA",
    "axes.labelcolor":  C_DARK,
    "xtick.color":      C_DARK,
    "ytick.color":      C_DARK,
    "text.color":       C_DARK,
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
    "grid.color":       "#DDDDDD",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
})


# ════════════════════════════════════════════════════════════════════════════
# STEP 0 — LOAD DATA
# ════════════════════════════════════════════════════════════════════════════
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(f"✅  Loaded {len(df):,} records  |  {df['dispatching_zone_id'].nunique()} zones  |  "
          f"{df['date'].min().date()} → {df['date'].max().date()}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════
def feature_engineering(df):
    df = df.copy()
    df["day_of_week_num"] = df["date"].dt.dayofweek
    df["week_of_year"]    = df["date"].dt.isocalendar().week.astype(int)
    df["month"]           = df["date"].dt.month
    df["quarter"]         = df["date"].dt.quarter
    df["is_month_end"]    = df["date"].dt.is_month_end.astype(int)
    df["day_of_year"]     = df["date"].dt.dayofyear

    # Rolling features per zone
    df = df.sort_values(["dispatching_zone_id", "date"]).reset_index(drop=True)
    df["trips_7d_avg"]    = df.groupby("dispatching_zone_id")["trips"].transform(
                                lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    df["trips_14d_avg"]   = df.groupby("dispatching_zone_id")["trips"].transform(
                                lambda x: x.shift(1).rolling(14, min_periods=1).mean())
    df["vehicles_7d_avg"] = df.groupby("dispatching_zone_id")["active_vehicles"].transform(
                                lambda x: x.shift(1).rolling(7, min_periods=1).mean())

    # Lag features
    df["trips_lag1"]  = df.groupby("dispatching_zone_id")["trips"].shift(1)
    df["trips_lag7"]  = df.groupby("dispatching_zone_id")["trips"].shift(7)

    # Zone encoding
    le = LabelEncoder()
    df["zone_encoded"] = le.fit_transform(df["dispatching_zone_id"])

    df.fillna(df.median(numeric_only=True), inplace=True)
    print("✅  Feature engineering complete  |  columns:", df.shape[1])
    return df, le


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — UNSUPERVISED LEARNING
# ════════════════════════════════════════════════════════════════════════════
def unsupervised_learning(df):
    # ── 2A: K-Means zone clustering ─────────────────────────────────────────
    zone_agg = df.groupby("dispatching_zone_id").agg(
        avg_vehicles=("active_vehicles", "mean"),
        avg_trips=("trips", "mean"),
        avg_tpv=("trips_per_vehicle", "mean"),
    ).reset_index()

    scaler_km = StandardScaler()
    X_km = scaler_km.fit_transform(zone_agg[["avg_vehicles", "avg_trips", "avg_tpv"]])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
    zone_agg["demand_cluster"] = kmeans.fit_predict(X_km)

    # Label clusters semantically
    cluster_trip_mean = zone_agg.groupby("demand_cluster")["avg_trips"].mean()
    sorted_clusters   = cluster_trip_mean.sort_values(ascending=False).index.tolist()
    cluster_map = {sorted_clusters[0]: "High", sorted_clusters[1]: "Medium",
                   sorted_clusters[2]: "Low"}
    zone_agg["cluster_label"] = zone_agg["demand_cluster"].map(cluster_map)

    df = df.merge(zone_agg[["dispatching_zone_id", "demand_cluster", "cluster_label"]],
                  on="dispatching_zone_id", how="left")

    # ── 2B: Isolation Forest anomaly detection ──────────────────────────────
    # Aggregate daily totals across all zones for city-level anomaly detection
    daily = df.groupby("date").agg(
        total_trips=("trips", "sum"),
        total_vehicles=("active_vehicles", "sum"),
    ).reset_index()

    iso = IsolationForest(contamination=0.07, random_state=42)
    feat_iso = daily[["total_trips", "total_vehicles"]].values
    daily["if_label"]      = iso.fit_predict(feat_iso)         # -1 = anomaly
    daily["anomaly_score"] = -iso.score_samples(feat_iso)      # higher = more anomalous
    # Normalise 0–1
    mn, mx = daily["anomaly_score"].min(), daily["anomaly_score"].max()
    daily["anomaly_score"] = (daily["anomaly_score"] - mn) / (mx - mn)

    df = df.merge(daily[["date", "anomaly_score"]], on="date", how="left")

    print("✅  Unsupervised learning complete")
    print(f"    Demand clusters : {zone_agg[['dispatching_zone_id','cluster_label']].to_string(index=False)}")
    print(f"    Anomaly days    : {(daily['if_label'] == -1).sum()}")

    return df, zone_agg, daily


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — SUPERVISED LEARNING
# ════════════════════════════════════════════════════════════════════════════
FEATURES = [
    "active_vehicles", "zone_encoded", "day_of_week_num", "is_weekend",
    "month", "quarter", "week_of_year", "day_of_year", "is_month_end",
    "trips_7d_avg", "trips_14d_avg", "trips_lag1", "trips_lag7",
    "vehicles_7d_avg", "trips_per_vehicle", "anomaly_score",
    "demand_cluster",
]

def supervised_learning(df):
    df_model = df.dropna(subset=FEATURES + ["trips"]).copy()

    # Chronological train / test split (80/20)
    split_idx = int(len(df_model) * 0.80)
    train = df_model.iloc[:split_idx]
    test  = df_model.iloc[split_idx:]

    X_tr, y_tr = train[FEATURES], train["trips"]
    X_te, y_te = test[FEATURES],  test["trips"]

    # ── Baseline: Linear Regression ─────────────────────────────────────────
    lr = LinearRegression()
    lr.fit(X_tr, y_tr)
    lr_pred = lr.predict(X_te)

    # ── Random Forest ───────────────────────────────────────────────────────
    rf = RandomForestRegressor(n_estimators=200, max_depth=12,
                                min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_pred = rf.predict(X_te)

    # ── Gradient Boosting ───────────────────────────────────────────────────
    gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.06,
                                    max_depth=5, subsample=0.85, random_state=42)
    gb.fit(X_tr, y_tr)
    gb_pred = gb.predict(X_te)

    def metrics(name, y_true, y_pred):
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
        print(f"  {name:<22}  MAE={mae:,.0f}  RMSE={rmse:,.0f}  R²={r2:.3f}  MAPE={mape:.1f}%")
        return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

    print("\n✅  Model performance on held-out test set:")
    results = [
        metrics("Linear Regression",    y_te, lr_pred),
        metrics("Random Forest",         y_te, rf_pred),
        metrics("Gradient Boosting",     y_te, gb_pred),
    ]

    return {
        "train": train, "test": test,
        "lr_pred": lr_pred, "rf_pred": rf_pred, "gb_pred": gb_pred,
        "lr": lr, "rf": rf, "gb": gb,
        "results": results,
        "X_tr": X_tr, "X_te": X_te, "y_te": y_te,
    }


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — FIGURES
# ════════════════════════════════════════════════════════════════════════════

def fig_daily_trips_overview(df, daily):
    """Fig 1: City-wide daily trips with event annotations"""
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_facecolor(C_SAND)
    fig.patch.set_facecolor(C_SAND)

    ax.plot(daily["date"], daily["total_trips"], color=C_DARK, lw=1.5, alpha=0.9, label="Daily Trips")
    ax.fill_between(daily["date"], daily["total_trips"], alpha=0.12, color=C_DARK)

    # Annotate key anomalies
    highlights = {
        "2023-01-07": ("Severe\nFog",      C_MID,  "bottom"),
        "2023-03-08": ("Holi Eve\nSpike",  C_RUST, "top"),
        "2023-10-24": ("Diwali\nEve",      C_RUST, "top"),
        "2023-12-31": ("New Year\nEve",    C_RUST, "top"),
    }
    for dstr, (label, col, va) in highlights.items():
        d = pd.Timestamp(dstr)
        row = daily[daily["date"] == d]
        if not row.empty:
            y_val = row["total_trips"].values[0]
            ax.scatter(d, y_val, color=col, s=90, zorder=5)
            offset = 7000 if va == "top" else -9000
            ax.annotate(label, (d, y_val), xytext=(d, y_val + offset),
                        fontsize=8.5, color=col, fontweight="bold",
                        ha="center", arrowprops=dict(arrowstyle="-", color=col, lw=0.8))

    ax.set_title("Delhi NCR  ·  City-Wide Daily Ride Volume  (Jan – Dec 2023)", pad=12)
    ax.set_ylabel("Total Daily Trips")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", framealpha=0.5)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "01_daily_trips_overview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾  {path}")


def fig_zone_clustering(zone_agg):
    """Fig 2: K-Means demand zone scatter"""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor(C_SAND)
    fig.patch.set_facecolor(C_SAND)

    cluster_colors = {"High": C_RUST, "Medium": C_MID, "Low": C_GREEN}
    cluster_bg     = {"High": "#F8DDD5", "Medium": "#D5E4F5", "Low": "#D5EFE0"}

    for clabel, grp in zone_agg.groupby("cluster_label"):
        col = cluster_colors[clabel]
        ax.scatter(grp["avg_vehicles"], grp["avg_trips"],
                   color=col, s=200, zorder=4, edgecolors="white", lw=1.5,
                   label=f"Cluster: {clabel}")
        for _, row in grp.iterrows():
            short = row["dispatching_zone_id"].split("_")[0]
            ax.annotate(short, (row["avg_vehicles"], row["avg_trips"]),
                        xytext=(8, 5), textcoords="offset points",
                        fontsize=9, color=C_DARK, fontweight="bold")

    ax.set_title("K-Means Demand Zone Clustering  ·  Delhi NCR 2023", pad=12)
    ax.set_xlabel("Avg Active Vehicles / Day")
    ax.set_ylabel("Avg Daily Trips")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    ax.legend(framealpha=0.6)
    ax.grid(True)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "02_zone_clustering.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾  {path}")


def fig_anomaly_detection(daily):
    """Fig 3: Anomaly scores over time"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    for ax in (ax1, ax2):
        ax.set_facecolor(C_SAND)
    fig.patch.set_facecolor(C_SAND)

    anomalies = daily[daily["if_label"] == -1]
    normal    = daily[daily["if_label"] == 1]

    ax1.plot(daily["date"], daily["total_trips"], color=C_DARK, lw=1.3, alpha=0.85)
    ax1.scatter(normal["date"],    normal["total_trips"],    color=C_DARK, s=12, alpha=0.4)
    ax1.scatter(anomalies["date"], anomalies["total_trips"], color=C_RUST,
                s=60, zorder=5, label="Anomaly detected")
    ax1.set_ylabel("Total Daily Trips")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    ax1.set_title("Isolation Forest  ·  Structural Anomaly Detection  ·  Delhi NCR 2023", pad=12)
    ax1.legend(framealpha=0.6)
    ax1.grid(True, axis="y")

    ax2.fill_between(daily["date"], daily["anomaly_score"], color=C_RUST, alpha=0.55)
    ax2.axhline(0.65, color=C_RUST, lw=1.2, ls="--", alpha=0.8, label="Threshold 0.65")
    ax2.set_ylabel("Anomaly Score")
    ax2.set_ylim(0, 1.05)
    ax2.legend(framealpha=0.6, fontsize=9)
    ax2.grid(True, axis="y")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "03_anomaly_detection.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾  {path}")


def fig_model_comparison(model_results):
    """Fig 4: Model performance bar chart"""
    res = pd.DataFrame(model_results["results"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor(C_SAND)
    metrics_plot = [("MAE", "Mean Absolute Error (trips)", False),
                    ("RMSE", "Root Mean Sq. Error (trips)", False),
                    ("R2",   "R² Score (higher = better)",  True)]

    bar_colors = [C_MID, C_DARK, C_RUST]
    for ax, (metric, ylabel, higher_better) in zip(axes, metrics_plot):
        ax.set_facecolor(C_SAND)
        bars = ax.bar(res["model"], res[metric], color=bar_colors,
                      edgecolor="white", width=0.5)
        for bar, val in zip(bars, res[metric]):
            label = f"{val:.3f}" if metric == "R2" else f"{val:,.0f}"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*bar.get_height(),
                    label, ha="center", va="bottom", fontsize=10, fontweight="bold", color=C_DARK)
        ax.set_title(ylabel, fontsize=12)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)

    fig.suptitle("Model Performance Comparison  ·  Delhi NCR Demand Forecasting",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "04_model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾  {path}")


def fig_forecast_vs_actual(model_results):
    """Fig 5: Actual vs Predicted (zone-aggregate) on test set"""
    test    = model_results["test"].copy()
    gb_pred = model_results["gb_pred"]
    lr_pred = model_results["lr_pred"]

    # Daily aggregate across zones
    test["gb_pred"] = gb_pred
    test["lr_pred"] = lr_pred
    daily_test = test.groupby("date").agg(
        actual=("trips", "sum"),
        gb_pred=("gb_pred", "sum"),
        lr_pred=("lr_pred", "sum"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_facecolor(C_SAND)
    fig.patch.set_facecolor(C_SAND)

    ax.plot(daily_test["date"], daily_test["actual"],  color=C_DARK, lw=2.0, label="Actual Trips")
    ax.plot(daily_test["date"], daily_test["gb_pred"], color=C_RUST, lw=1.8, ls="-",
            alpha=0.9, label="Hybrid Model (Gradient Boosting)")
    ax.plot(daily_test["date"], daily_test["lr_pred"], color=C_MID,  lw=1.4, ls="--",
            alpha=0.7, label="Baseline (Linear Regression)")

    ax.set_title("Forecast vs Actual  ·  Test Period  ·  Delhi NCR 2023", pad=12)
    ax.set_ylabel("Total Daily Trips")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    ax.legend(framealpha=0.6)
    ax.grid(True, axis="y")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "05_forecast_vs_actual.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾  {path}")


def fig_feature_importance(model_results):
    """Fig 6: Feature importance from Random Forest"""
    rf = model_results["rf"]
    importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
    top15 = importances.tail(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor(C_SAND)
    fig.patch.set_facecolor(C_SAND)

    colors = [C_RUST if f == "anomaly_score" else
              C_MID  if "cluster" in f       else C_DARK
              for f in top15.index]
    bars = ax.barh(top15.index, top15.values, color=colors, edgecolor="white", height=0.65)

    handles = [mpatches.Patch(color=C_RUST, label="Anomaly Score (from Unsupervised)"),
               mpatches.Patch(color=C_MID,  label="Demand Cluster (from K-Means)"),
               mpatches.Patch(color=C_DARK, label="Engineered / Raw Features")]
    ax.legend(handles=handles, framealpha=0.6, fontsize=9)

    ax.set_title("Feature Importance  ·  Random Forest  ·  Delhi NCR 2023", pad=12)
    ax.set_xlabel("Importance Score")
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "06_feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾  {path}")


def fig_zone_weekly_heatmap(df):
    """Fig 7: Weekly demand heatmap per zone"""
    pivot = df.groupby(["dispatching_zone_id", "day_of_week"])["trips"].mean().reset_index()
    pivot["dispatching_zone_id"] = pivot["dispatching_zone_id"].str.split("_").str[0]
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = pivot.pivot(index="dispatching_zone_id", columns="day_of_week", values="trips")
    pivot = pivot.reindex(columns=dow_order)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_facecolor(C_SAND)
    fig.patch.set_facecolor(C_SAND)

    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    sns.heatmap(pivot, cmap=cmap, annot=True, fmt=".0f", ax=ax,
                linewidths=0.4, linecolor="#DDDDDD",
                cbar_kws={"label": "Avg Daily Trips"})
    ax.set_title("Average Daily Trips by Zone & Day of Week  ·  Delhi NCR 2023", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "07_zone_weekly_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾  {path}")


def fig_tpv_efficiency(df):
    """Fig 8: Trips-per-vehicle efficiency distribution"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor(C_SAND)
    fig.patch.set_facecolor(C_SAND)

    for i, (zone, grp) in enumerate(df.groupby("dispatching_zone_id")):
        short = zone.split("_")[0]
        ax.violinplot(grp["trips_per_vehicle"].dropna(), positions=[i],
                      widths=0.6, showmedians=True)

    ax.axhline(8.5, color=C_RUST, ls="--", lw=1.5, label="Target TPV = 8.5")
    ax.set_xticks(range(len(df["dispatching_zone_id"].unique())))
    ax.set_xticklabels([z.split("_")[0] for z in sorted(df["dispatching_zone_id"].unique())],
                       rotation=30, ha="right")
    ax.set_title("Trips Per Vehicle (TPV) Distribution by Zone  ·  Efficiency Analysis", pad=12)
    ax.set_ylabel("Trips per Vehicle per Day")
    ax.legend(framealpha=0.6)
    ax.grid(True, axis="y")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "08_tpv_efficiency.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾  {path}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "="*65)
    print("  Delhi NCR Fleet Dynamics  ·  Hybrid ML Pipeline")
    print("="*65 + "\n")

    df = load_data()

    print("\n── Step 1: Feature Engineering ──────────────────────────────")
    df, le = feature_engineering(df)

    print("\n── Step 2: Unsupervised Learning ─────────────────────────────")
    df, zone_agg, daily = unsupervised_learning(df)

    print("\n── Step 3: Supervised Learning ───────────────────────────────")
    model_results = supervised_learning(df)

    print("\n── Step 4: Generating Figures ────────────────────────────────")
    fig_daily_trips_overview(df, daily)
    fig_zone_clustering(zone_agg)
    fig_anomaly_detection(daily)
    fig_model_comparison(model_results)
    fig_forecast_vs_actual(model_results)
    fig_feature_importance(model_results)
    fig_zone_weekly_heatmap(df)
    fig_tpv_efficiency(df)

    # Save enriched dataset
    out = os.path.join(BASE_DIR, "data", "delhi_ncr_fleet_enriched.csv")
    df.to_csv(out, index=False)
    print(f"\n✅  Enriched dataset saved → {out}")
    print("\n✅  Pipeline complete.\n")


if __name__ == "__main__":
    main()
