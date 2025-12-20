#!/usr/bin/env python3
# firebase_window_wqi_service.py

import os
import json
import time
import math
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import firebase_admin
from firebase_admin import credentials, db
from joblib import load as joblib_load

warnings.filterwarnings("ignore")

# ======================================================
# CONFIG
# ======================================================
FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL")
SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

PATH_SENSOR = os.getenv("FB_PATH_SENSOR", "sensor_readings")
PATH_WATERLOGS = os.getenv("FB_PATH_WATERLOGS", "waterLogs")
PATH_OUT = os.getenv("FB_PATH_RES_WINDOWS", "wqi_window_results")

TIME_FIELD = os.getenv("TIME_FIELD", "timestamp")

WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "10"))
STEP_SIZE = int(os.getenv("STEP_SIZE", "3"))
LOOKBACK_MIN = int(os.getenv("LOOKBACK_MIN", "10"))

USE_MODEL = os.getenv("USE_MODEL", "true").lower() == "true"
ART_DIR = os.getenv("ARTIFACT_DIR", "./artifacts")

# ======================================================
# WQI CONFIG
# ======================================================
WQI_WEIGHTS = {
    "pH": 0.2,
    "dissolvedO2": 0.2,
    "turbidity": 0.2,
    "tds": 0.2,
    "temp": 0.1,
    "chlorophyll": 0.1,
}

# ======================================================
# FIREBASE INIT
# ======================================================
def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_JSON)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
    print("✅ Connected to Firebase")

# ======================================================
# UTILS
# ======================================================
def parse_ts(v):
    try:
        return pd.to_datetime(v, dayfirst=True, errors="coerce")
    except:
        return pd.NaT

def safe_concat_by_ts(dfs):
    valid = [d for d in dfs if isinstance(d, pd.DataFrame) and not d.empty and "ts" in d.columns]
    if not valid:
        return pd.DataFrame()
    df = pd.concat(valid, ignore_index=True)
    df = df.dropna(subset=["ts"])
    return df.sort_values("ts").reset_index(drop=True)

def json_safe(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return 0
    return x

# ======================================================
# FETCH DATA
# ======================================================
def fetch_node(path):
    data = db.reference(path).get() or {}
    if not isinstance(data, dict):
        return pd.DataFrame()
    df = pd.DataFrame(data.values())
    if TIME_FIELD not in df.columns:
        for alt in ["time", "timestamp"]:
            if alt in df.columns:
                df[TIME_FIELD] = df[alt]
                break
    if TIME_FIELD not in df.columns:
        return pd.DataFrame()
    df["ts"] = df[TIME_FIELD].apply(parse_ts)
    df = df.dropna(subset=["ts"])
    return df

# ======================================================
# WQI
# ======================================================
def score_param(k, v):
    if v is None:
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
    total, acc = 0, 0
    for k, w in WQI_WEIGHTS.items():
        acc += w * score_param(k, stats.get(k))
        total += w
    wqi = acc / total if total else 0
    cat = "Excellent" if wqi >= 90 else "Good" if wqi >= 70 else "Fair" if wqi >= 50 else "Poor"
    return round(wqi, 2), cat

# ======================================================
# ML MODEL
# ======================================================
class MLModel:
    def __init__(self, d):
        self.scaler = joblib_load(os.path.join(d, "scaler.joblib"))
        self.hmm = joblib_load(os.path.join(d, "hmm_model.joblib"))
        self.xgb = joblib_load(os.path.join(d, "xgb_model.joblib"))
        self.le = joblib_load(os.path.join(d, "label_encoder.joblib"))
        with open(os.path.join(d, "final_feature_order.json")) as f:
            self.features = json.load(f)
        print("✅ ML artifacts loaded")

    def predict(self, df):
        X = []
        for c in self.features:
            if c in df.columns:
                X.append(pd.to_numeric(df[c], errors="coerce").fillna(0).values)
            else:
                X.append(np.zeros(len(df)))
        X = np.column_stack(X)
        Xs = self.scaler.transform(X)
        _, post = self.hmm.score_samples(Xs)
        Xh = np.hstack([Xs, post])
        y = self.xgb.predict(Xh)
        conf = self.xgb.predict_proba(Xh).max(axis=1)
        return self.le.inverse_transform(y), conf

# ======================================================
# MAIN
# ======================================================
def main():
    init_firebase()

    df1 = fetch_node(PATH_SENSOR)
    df2 = fetch_node(PATH_WATERLOGS)

    df = safe_concat_by_ts([df1, df2])
    if df.empty or len(df) < WINDOW_SIZE:
        print("⚠️ Not enough data")
        return

    model = MLModel(ART_DIR) if USE_MODEL else None

    out_ref = db.reference(PATH_OUT)

    for i in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        win = df.iloc[i:i + WINDOW_SIZE]

        stats = {}
        for c in WQI_WEIGHTS:
            if c in win.columns:
                stats[c] = float(pd.to_numeric(win[c], errors="coerce").mean())

        wqi, cat = compute_wqi(stats)

        row = {
            **{k: json_safe(v) for k, v in stats.items()},
            "wqi": wqi,
            "wqi_category": cat,
            "n_readings": len(win),
            "window_start_iso": win["ts"].iloc[0].isoformat(),
            "window_end_iso": win["ts"].iloc[-1].isoformat(),
        }

        if model:
            cls, conf = model.predict(pd.DataFrame([row]))
            row["model_class"] = cls[0]
            row["model_conf"] = float(conf[0])

        out_ref.push(row)

    print("✅ WQI windows pushed")

# ======================================================
if __name__ == "__main__":
    main()
