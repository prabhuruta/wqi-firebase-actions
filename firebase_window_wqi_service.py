#!/usr/bin/env python3
"""
firebase_window_wqi_service.py
Single-pass WQI window processor with ML (HMM + XGBoost)
"""

import os, json, math, time, warnings
from datetime import datetime
import numpy as np
import pandas as pd

import firebase_admin
from firebase_admin import credentials, db
from joblib import load as joblib_load

warnings.filterwarnings("ignore")

# ===============================
# ENV CONFIG
# ===============================
FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL")
SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

PATH_SENSOR = os.getenv("FB_PATH_SENSOR", "sensor_readings")
PATH_WATERLOGS = os.getenv("FB_PATH_WATERLOGS", "waterLogs")
PATH_OUT = os.getenv("FB_PATH_RES_WINDOWS", "wqi_window_results")

TIME_FIELD = os.getenv("TIME_FIELD", "timestamp")
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "10"))
STEP_SIZE = int(os.getenv("STEP_SIZE", "3"))
USE_MODEL = os.getenv("USE_MODEL", "true").lower() == "true"
ART_DIR = os.getenv("ARTIFACT_DIR", "./artifacts")

# ===============================
# WQI CONFIG
# ===============================
WQI_WEIGHTS = {
    "pH": 0.2,
    "dissolvedO2": 0.2,
    "turbidity": 0.2,
    "tds": 0.2,
    "temp": 0.1,
    "chlorophyll": 0.1
}

SENSOR_FIELDS = [
    "pH", "dissolvedO2", "turbidity", "tds", "temp",
    "chlorophyll", "orp", "bga", "bga_temp", "chl_temp", "lat", "lon"
]

FIELD_ALIASES = {
    "chlorophyll_ug_per_L": "chlorophyll",
    "blue_green_algae_cells_per_mL": "bga",
    "bga_temp_C": "bga_temp",
    "chl_temp_C": "chl_temp"
}

# ===============================
# FIREBASE INIT
# ===============================
def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_JSON)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
    print("✅ Connected to Firebase")

# ===============================
# TIME PARSER (NO TZ CONVERSION)
# ===============================
def parse_ts(v):
    if v is None:
        return None
    s = str(v)
    try:
        return datetime.strptime(s, "%d-%m-%Y %H:%M:%S")
    except:
        try:
            return pd.to_datetime(s).to_pydatetime()
        except:
            return None

# ===============================
# FETCH NODE
# ===============================
def fetch_node(path):
    ref = db.reference(path)
    raw = ref.get() or {}
    rows = []
    for r in raw.values():
        if not isinstance(r, dict):
            continue
        for k, v in FIELD_ALIASES.items():
            if k in r:
                r[v] = r.pop(k)
        ts = parse_ts(r.get(TIME_FIELD))
        if ts:
            r["ts"] = ts
            rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("ts").reset_index(drop=True)

# ===============================
# WINDOW UTILS
# ===============================
def sliding_windows(n, size, step):
    i = 0
    while i + size <= n:
        yield i, i + size
        i += step

def aggregate_window(df):
    stats = {}
    for f in SENSOR_FIELDS:
        if f in df:
            stats[f] = float(pd.to_numeric(df[f], errors="coerce").mean())
    stats["n_readings"] = int(len(df))
    return stats

# ===============================
# WQI
# ===============================
def score_param(k, v):
    if v is None or math.isnan(v):
        return 0
    if k == "pH":
        return 100 if 7 <= v <= 8 else 60
    if k == "dissolvedO2":
        return min(100, v * 12.5)
    if k == "turbidity":
        return max(0, 100 - 15 * v)
    if k == "tds":
        return 100 if v <= 300 else max(0, 100 - (v - 300) * 0.1)
    if k == "temp":
        return 100 if 15 <= v <= 25 else 70
    if k == "chlorophyll":
        return max(0, 100 - v * 2)
    return 50

def compute_wqi(stats):
    s, w = 0, 0
    for k, wt in WQI_WEIGHTS.items():
        v = stats.get(k)
        sc = score_param(k, v)
        s += sc * wt
        w += wt
    wqi = round(s / w, 2)
    cat = "Excellent" if wqi >= 90 else "Good" if wqi >= 70 else "Fair" if wqi >= 50 else "Poor"
    return wqi, cat

# ===============================
# OPTIONAL ML MODEL
# ===============================
class MLModel:
    def __init__(self, art):
        self.scaler = joblib_load(f"{art}/scaler.joblib")
        self.hmm = joblib_load(f"{art}/hmm_model.joblib")
        self.xgb = joblib_load(f"{art}/xgb_model.joblib")
        self.le = joblib_load(f"{art}/label_encoder.joblib")
        with open(f"{art}/final_feature_order.json") as f:
            self.order = json.load(f)
        print("✅ ML artifacts loaded")

    def predict(self, df):
        X = []
        for c in self.order:
            X.append(pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0).values)
        X = np.column_stack(X)
        Xs = self.scaler.transform(X)
        _, post = self.hmm.score_samples(Xs)
        Xh = np.hstack([Xs, post])
        y = self.xgb.predict(Xh)
        conf = self.xgb.predict_proba(Xh).max(axis=1)
        return self.le.inverse_transform(y), conf

# ===============================
# MAIN PROCESS
# ===============================
def main():
    init_firebase()

    df1 = fetch_node(PATH_SENSOR)
    df2 = fetch_node(PATH_WATERLOGS)
    df = pd.concat([df1, df2], ignore_index=True).sort_values("ts")

    if df.empty:
        print("❌ No data in Firebase")
        return

    model = MLModel(ART_DIR) if USE_MODEL else None
    out_ref = db.reference(PATH_OUT)

    pushed = 0
    for s, e in sliding_windows(len(df), WINDOW_SIZE, STEP_SIZE):
        win = df.iloc[s:e]
        stats = aggregate_window(win)
        wqi, cat = compute_wqi(stats)

        row = {
            **stats,
            "wqi": wqi,
            "wqi_category": cat,
            "window_start_iso": win["ts"].iloc[0].isoformat(),
            "window_end_iso": win["ts"].iloc[-1].isoformat(),
            "window_start_epoch": int(win["ts"].iloc[0].timestamp() * 1000),
            "window_end_epoch": int(win["ts"].iloc[-1].timestamp() * 1000),
        }

        out_df = pd.DataFrame([row])
        if model:
            cls, conf = model.predict(out_df)
            row["model_class"] = cls[0]
            row["model_conf"] = float(conf[0])

        # JSON-safe
        payload = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}
        out_ref.push(payload)
        pushed += 1

    print(f"✅ Pushed {pushed} window records")

# ===============================
if __name__ == "__main__":
    main()
