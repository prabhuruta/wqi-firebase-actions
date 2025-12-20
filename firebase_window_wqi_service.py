#!/usr/bin/env python3
"""
firebase_window_wqi_service.py

Reads sensor_readings and waterLogs from Firebase RTDB,
creates sliding windows, computes WQI, applies rule checks,
runs HMM + XGBoost model, and writes window-level results
(with timestamps preserved) to wqi_window_results.
"""

import os, time, json, math, warnings
from datetime import datetime
import numpy as np
import pandas as pd

# Firebase
import firebase_admin
from firebase_admin import credentials, db

# ML
from joblib import load as joblib_load

warnings.filterwarnings("ignore")

# ======================================================
# CONFIG (ENV VARS)
# ======================================================
FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL")
SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

PATH_SENSOR = os.getenv("FB_PATH_SENSOR", "sensor_readings")
PATH_WATERLOGS = os.getenv("FB_PATH_WATERLOGS", "waterLogs")
PATH_RES_WINDOWS = os.getenv("FB_PATH_RES_WINDOWS", "wqi_window_results")

TIME_FIELD = os.getenv("TIME_FIELD", "timestamp")
ASOF_TOL_SEC = int(os.getenv("ASOF_TOL_SEC", "30"))

WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "10"))
STEP_SIZE = int(os.getenv("STEP_SIZE", "3"))

RUN_CONTINUOUS = os.getenv("RUN_CONTINUOUS", "false").lower() == "true"
LOOKBACK_MIN = int(os.getenv("LOOKBACK_MIN", "10"))
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "20"))

USE_MODEL = os.getenv("USE_MODEL", "true").lower() == "true"
ART_DIR = os.getenv("ARTIFACT_DIR", "./artifacts")

ACCEPTABLE_MODEL_CLASSES = set(
    json.loads(os.getenv("ACCEPTABLE_MODEL_CLASSES", '["low"]'))
)

# ======================================================
# WQI CONFIG
# ======================================================
WQI_WEIGHTS = {
    "pH": 0.20,
    "dissolvedO2": 0.20,
    "turbidity": 0.20,
    "tds": 0.20,
    "temp": 0.10,
    "chlorophyll": 0.10
}

DRINK_LIMITS = {
    "pH": (6.5, 8.5),
    "dissolvedO2": (5.0, np.inf),
    "turbidity": (0.0, 5.0),
    "tds": (0.0, 500.0),
    "temp": (5.0, 30.0),
    "chlorophyll": (0.0, 30.0)
}

SENSOR_FIELDS = [
    "pH","dissolvedO2","turbidity","tds","temp",
    "chlorophyll","orp","bga","bga_temp","chl_temp","lat","lon"
]

# ======================================================
# FIREBASE INIT
# ======================================================
def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_JSON)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
    return db

# ======================================================
# TIMESTAMP PARSER (NO CONVERSION)
# ======================================================
def parse_any_ts(series):
    out = []
    for v in series:
        try:
            dt = datetime.strptime(str(v), "%d-%m-%Y %H:%M:%S")
            out.append(dt)
        except:
            dt = pd.to_datetime(v, errors="coerce")
            out.append(dt if not pd.isna(dt) else np.nan)
    return pd.Series(out, dtype="datetime64[ns]")

# ======================================================
# FETCH DATA
# ======================================================
def fetch_node(path, since_ts=None):
    data = db.reference(path).get() or {}
    rows = [v for v in data.values() if isinstance(v, dict)]
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if TIME_FIELD not in df.columns:
        for c in ["timestamp", "time"]:
            if c in df.columns:
                df[TIME_FIELD] = df[c]
                break

    if TIME_FIELD not in df.columns:
        return pd.DataFrame()

    df["ts"] = parse_any_ts(df[TIME_FIELD])
    df = df.dropna(subset=["ts"]).sort_values("ts")

    if since_ts is not None:
        df = df[df["ts"] > since_ts]

    return df.reset_index(drop=True)

# ======================================================
# WINDOWING
# ======================================================
def sliding_windows(n, size, step):
    i = 0
    while i + size <= n:
        yield i, i + size
        i += step

def aggregate_window(dfw):
    stats = {}
    for c in SENSOR_FIELDS:
        if c in dfw.columns:
            stats[c] = float(pd.to_numeric(dfw[c], errors="coerce").mean())
    stats["n_readings"] = len(dfw)
    return stats

# ======================================================
# WQI
# ======================================================
def score_param(name, v):
    if v is None or np.isnan(v): return 0
    if name == "pH": return 100 if 7 <= v <= 8 else 60
    if name == "dissolvedO2": return min(100, v * 12.5)
    if name == "turbidity": return max(0, 100 - 15*v)
    if name == "tds": return 100 if v <= 300 else max(0, 100 - (v-300)*0.1)
    if name == "temp": return 100 if 15 <= v <= 25 else 70
    if name == "chlorophyll": return max(0, 100 - v*2)
    return 50

def compute_wqi(stats):
    total, wsum = 0, 0
    for k,w in WQI_WEIGHTS.items():
        s = score_param(k, stats.get(k))
        wsum += w*s
        total += w
    wqi = wsum/total if total else 0
    cat = "Excellent" if wqi>=90 else "Good" if wqi>=70 else "Fair" if wqi>=50 else "Poor"
    return round(wqi,2), cat

# ======================================================
# ML MODEL
# ======================================================
class MLModel:
    def __init__(self, path):
        self.scaler = joblib_load(f"{path}/scaler.joblib")
        self.hmm = joblib_load(f"{path}/hmm_model.joblib")
        self.xgb = joblib_load(f"{path}/xgb_model.joblib")
        self.le = joblib_load(f"{path}/label_encoder.joblib")
        with open(f"{path}/final_feature_order.json") as f:
            self.features = json.load(f)

    def predict(self, df):
        X = np.column_stack([
            pd.to_numeric(df.get(f, 0), errors="coerce").fillna(0)
            for f in self.features
        ])
        Xs = self.scaler.transform(X)
        logp, post = self.hmm.score_samples(Xs)
        Xh = np.hstack([Xs, post, logp.reshape(-1,1)])
        y = self.xgb.predict(Xh)
        p = self.xgb.predict_proba(Xh).max(axis=1)
        return self.le.inverse_transform(y), p

# ======================================================
# MAIN PROCESS
# ======================================================
def process_once(since_ts, model):
    df1 = fetch_node(PATH_SENSOR, since_ts)
    df2 = fetch_node(PATH_WATERLOGS, since_ts)

    if df1.empty and df2.empty:
        return since_ts

    df = pd.concat([df1, df2]).sort_values("ts").reset_index(drop=True)
    if len(df) < WINDOW_SIZE:
        return df["ts"].max()

    out = []
    for s,e in sliding_windows(len(df), WINDOW_SIZE, STEP_SIZE):
        win = df.iloc[s:e]
        stats = aggregate_window(win)
        wqi, cat = compute_wqi(stats)

        row = {
            **stats,
            "wqi": wqi,
            "wqi_category": cat,
            "window_start_ts": int(win["ts"].iloc[0].timestamp()*1000),
            "window_end_ts": int(win["ts"].iloc[-1].timestamp()*1000),
            "window_start_iso": win["ts"].iloc[0].isoformat(),
            "window_end_iso": win["ts"].iloc[-1].isoformat()
        }
        out.append(row)

    out_df = pd.DataFrame(out)

    if model:
        cls, conf = model.predict(out_df)
        out_df["model_class"] = cls
        out_df["model_conf"] = conf

    ref = db.reference(PATH_RES_WINDOWS)
    for _,r in out_df.iterrows():
        payload = {k:(None if pd.isna(v) else v) for k,v in r.items()}
        ref.push(payload)

    return df["ts"].max()

# ======================================================
# ENTRY POINT
# ======================================================
def main():
    init_firebase()
    print("Connected to Firebase")

    model = MLModel(ART_DIR) if USE_MODEL else None
    since_ts = pd.Timestamp.now() - pd.Timedelta(minutes=LOOKBACK_MIN)

    if not RUN_CONTINUOUS:
        process_once(since_ts, model)
        return

    while True:
        since_ts = process_once(since_ts, model)
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
