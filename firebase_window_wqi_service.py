#!/usr/bin/env python3
"""
firebase_window_wqi_service.py

• Reads sensor_readings + waterLogs from Firebase RTDB
• Aligns by timestamp (as-of merge)
• Builds sliding windows
• Computes WQI + rule-based drinkability
• Optionally runs HMM + XGBoost model
• Pushes clean, JSON-safe results to /wqi_window_results
"""

import os
import time
import json
import math
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import firebase_admin
from firebase_admin import credentials, db
from joblib import load as joblib_load

warnings.filterwarnings("ignore")

# =====================================================
# CONFIG (Environment Variables)
# =====================================================
FIREBASE_DB_URL = os.getenv(
    "FIREBASE_DB_URL",
    "https://mywaterproject-e6489-default-rtdb.asia-southeast1.firebasedatabase.app/"
)
SERVICE_ACCOUNT_JSON = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS",
    "serviceAccountKey.json"
)

PATH_SENSOR = os.getenv("FB_PATH_SENSOR", "sensor_readings")
PATH_WATERLOGS = os.getenv("FB_PATH_WATERLOGS", "waterLogs")
PATH_RES_WINDOWS = os.getenv("FB_PATH_RES_WINDOWS", "wqi_window_results")

TIME_FIELD = os.getenv("TIME_FIELD", "time")
ASOF_TOL_SEC = int(os.getenv("ASOF_TOL_SEC", "30"))

WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "10"))
STEP_SIZE = int(os.getenv("STEP_SIZE", "3"))

RUN_CONTINUOUS = os.getenv("RUN_CONTINUOUS", "false").lower() == "true"
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "20"))
LOOKBACK_MIN = int(os.getenv("LOOKBACK_MIN", "120"))

USE_MODEL = os.getenv("USE_MODEL", "true").lower() == "true"
ART_DIR = os.getenv("ARTIFACT_DIR", "./artifacts")
ACCEPTABLE_MODEL_CLASSES = set(
    json.loads(os.getenv("ACCEPTABLE_MODEL_CLASSES", '["low"]'))
)

# =====================================================
# WQI CONFIG
# =====================================================
WQI_WEIGHTS = {
    "pH": 0.20,
    "dissolvedO2": 0.20,
    "turbidity": 0.20,
    "tds": 0.20,
    "temp": 0.10,
    "chlorophyll": 0.10
}

DRINK_LIMITS = {
    "pH": {"min": 6.5, "max": 8.5},
    "dissolvedO2": {"min": 5.0, "max": float("inf")},
    "turbidity": {"min": 0.0, "max": 5.0},
    "tds": {"min": 0.0, "max": 500.0},
    "temp": {"min": 5.0, "max": 30.0},
    "chlorophyll": {"min": 0.0, "max": 30.0}
}

# =====================================================
# FIELD NORMALIZATION (CRITICAL FIX)
# =====================================================
FIELD_ALIASES_ALL = {
    "tempC": "temp",
    "temperature": "temp",
    "chlorophyll_ug_per_L": "chlorophyll",
    "blue_green_algae_cells_per_mL": "bga",
    "bga_temp_C": "bga_temp",
    "chl_temp_C": "chl_temp",
    "dissolved_oxygen": "dissolvedO2",
    "do": "dissolvedO2",
}

SENSOR_FIELDS = [
    "pH", "dissolvedO2", "turbidity", "tds", "temp",
    "chlorophyll", "orp", "bga", "bga_temp", "chl_temp", "lat", "lon"
]

REQUIRED_CORE = ["pH", "tds", "turbidity", "temp"]

# =====================================================
# HELPERS
# =====================================================
def json_safe(v):
    if v is None:
        return None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v

def window_is_valid(stats, max_missing_ratio=0.5):
    vals = [stats.get(k, np.nan) for k in REQUIRED_CORE]
    return np.mean([pd.isna(v) for v in vals]) <= max_missing_ratio

def parse_any_ts(series):
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan)
            continue
        s = str(v).strip()
        try:
            if float(s) > 1e12:
                out.append(datetime.fromtimestamp(float(s) / 1000))
                continue
            if float(s) > 1e9:
                out.append(datetime.fromtimestamp(float(s)))
                continue
        except:
            pass
        try:
            out.append(pd.to_datetime(s, errors="coerce"))
        except:
            out.append(np.nan)
    return pd.Series(out, dtype="datetime64[ns]")

# =====================================================
# FIREBASE
# =====================================================
def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_JSON)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})

def fetch_node(path):
    ref = db.reference(path)
    data = ref.get() or {}
    if not isinstance(data, dict):
        return pd.DataFrame()
    df = pd.DataFrame([v for v in data.values() if isinstance(v, dict)])
    df = df.rename(columns=FIELD_ALIASES_ALL)
    for cand in ["timestamp", "time", "createdAt"]:
        if cand in df.columns:
            df["ts"] = parse_any_ts(df[cand])
            break
    if "ts" not in df:
        return pd.DataFrame()
    return df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

# =====================================================
# WINDOW + WQI
# =====================================================
def aggregate_window(df):
    stats = {}
    for c in SENSOR_FIELDS:
        if c in df:
            stats[c] = float(pd.to_numeric(df[c], errors="coerce").mean())
    stats["n_readings"] = len(df)
    return stats

def compute_wqi(stats):
    score = 0
    for k, w in WQI_WEIGHTS.items():
        v = stats.get(k)
        if v is None or pd.isna(v):
            continue
        if k == "pH":
            s = 100 if 7 <= v <= 8 else 60
        elif k == "dissolvedO2":
            s = min(100, v * 12.5)
        elif k == "turbidity":
            s = max(0, 100 - 15 * v)
        elif k == "tds":
            s = 100 if v <= 300 else max(0, 100 - (v - 300) * 0.1)
        elif k == "temp":
            s = 100 if 15 <= v <= 25 else 70
        else:
            s = max(0, 100 - v * 2)
        score += w * s
    return round(score, 2)

# =====================================================
# OPTIONAL ML
# =====================================================
class OptionalModel:
    def __init__(self, art_dir):
        self.scaler = joblib_load(os.path.join(art_dir, "scaler.joblib"))
        self.hmm = joblib_load(os.path.join(art_dir, "hmm_model.joblib"))
        self.xgb = joblib_load(os.path.join(art_dir, "xgb_model.joblib"))
        with open(os.path.join(art_dir, "final_feature_order.json")) as f:
            self.feature_order = json.load(f)
        self.ok = True

    def predict(self, df):
        X = df[self.feature_order].fillna(0).to_numpy()
        Xs = self.scaler.transform(X)
        logp, post = self.hmm.score_samples(Xs)
        Xh = np.hstack([Xs, post, logp.reshape(-1, 1)])
        return self.xgb.predict(Xh)

# =====================================================
# MAIN
# =====================================================
def main():
    init_firebase()
    model = OptionalModel(ART_DIR) if USE_MODEL else None

    df1 = fetch_node(PATH_SENSOR)
    df2 = fetch_node(PATH_WATERLOGS)

    df = pd.merge_asof(
        df1, df2, on="ts",
        tolerance=pd.Timedelta(seconds=ASOF_TOL_SEC),
        direction="nearest"
    )

    results = []
    for i in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        win = df.iloc[i:i+WINDOW_SIZE]
        stats = aggregate_window(win)
        if not window_is_valid(stats):
            continue
        row = stats.copy()
        row["wqi"] = compute_wqi(stats)
        results.append(row)

    if not results:
        print("No valid windows")
        return

    out = pd.DataFrame(results)

    if model:
        out["model_class"] = model.predict(out)

    out_ref = db.reference(PATH_RES_WINDOWS)
    for _, r in out.iterrows():
        payload = {k: json_safe(v) for k, v in r.items()}
        out_ref.push(payload)

    print(f"Pushed {len(out)} windows")

if __name__ == "__main__":
    main()
