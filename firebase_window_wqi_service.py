#!/usr/bin/env python3
"""
firebase_window_wqi_service.py
Stable + ML-safe + Firebase-safe version
"""

import os
import json
import math
import time
from datetime import datetime

import numpy as np
import pandas as pd

import firebase_admin
from firebase_admin import credentials, db
from joblib import load as joblib_load

# ==============================
# ENV CONFIG
# ==============================
FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL")
SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

PATH_SENSOR = os.getenv("FB_PATH_SENSOR", "sensor_readings")
PATH_WATERLOGS = os.getenv("FB_PATH_WATERLOGS", "waterLogs")
PATH_OUT = os.getenv("FB_PATH_RES_WINDOWS", "wqi_window_results")

TIME_FIELD = os.getenv("TIME_FIELD", "timestamp")
LOOKBACK_MIN = int(os.getenv("LOOKBACK_MIN", "60"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "10"))
STEP_SIZE = int(os.getenv("STEP_SIZE", "3"))

USE_MODEL = os.getenv("USE_MODEL", "true").lower() == "true"
ART_DIR = os.getenv("ARTIFACT_DIR", "./artifacts")

# ==============================
# SENSOR FIELDS (TRAINING-COMPATIBLE)
# ==============================
FEATURE_FIELDS = [
    "pH", "dissolvedO2", "turbidity", "tds", "temp",
    "chlorophyll", "orp", "bga", "bga_temp", "chl_temp"
]

# ==============================
# INIT FIREBASE
# ==============================
def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_JSON)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
    print("✅ Connected to Firebase")

# ==============================
# SAFE TIMESTAMP PARSER
# ==============================
def parse_ts(v):
    try:
        return datetime.strptime(str(v), "%d-%m-%Y %H:%M:%S")
    except Exception:
        try:
            return pd.to_datetime(v).to_pydatetime()
        except Exception:
            return None

# ==============================
# FETCH NODE
# ==============================
def fetch_node(path, since_ts):
    data = db.reference(path).get() or {}
    rows = []
    for k, v in data.items():
        if not isinstance(v, dict):
            continue
        ts = parse_ts(v.get(TIME_FIELD))
        if ts and ts > since_ts:
            v["ts"] = ts
            rows.append(v)
    return pd.DataFrame(rows)

# ==============================
# WINDOW AGGREGATION
# ==============================
def aggregate_window(df):
    out = {}
    for c in FEATURE_FIELDS:
        out[c] = float(pd.to_numeric(df.get(c, 0), errors="coerce").mean())
    out["n_readings"] = len(df)
    return out

# ==============================
# WQI
# ==============================
def compute_wqi(stats):
    score = 0
    weights = {
        "pH": 0.2, "dissolvedO2": 0.2, "turbidity": 0.2,
        "tds": 0.2, "temp": 0.1, "chlorophyll": 0.1
    }
    for k, w in weights.items():
        v = stats.get(k, 0)
        if np.isnan(v):
            v = 0
        score += w * min(100, max(0, v * 10))
    return round(score, 2)

# ==============================
# ML MODEL (FIXED)
# ==============================
class MLModel:
    def __init__(self, art_dir):
        self.scaler = joblib_load(os.path.join(art_dir, "scaler.joblib"))
        self.hmm = joblib_load(os.path.join(art_dir, "hmm_model.joblib"))
        self.xgb = joblib_load(os.path.join(art_dir, "xgb_model.joblib"))
        self.le = joblib_load(os.path.join(art_dir, "label_encoder.joblib"))
        with open(os.path.join(art_dir, "final_feature_order.json")) as f:
            self.features = json.load(f)
        print("✅ ML artifacts loaded")

    def predict(self, df):
        X = []
        for c in self.features:
            X.append(pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0).values)
        X = np.column_stack(X)

        Xs = self.scaler.transform(X)

        # 🔥 CRITICAL FIX: add log-likelihood feature
        logprob, post = self.hmm.score_samples(Xs)
        logfeat = np.full((len(Xs), 1), logprob / max(1, len(Xs)))

        Xh = np.hstack([Xs, post, logfeat])

        y = self.xgb.predict(Xh)
        conf = self.xgb.predict_proba(Xh).max(axis=1)

        return self.le.inverse_transform(y), conf

# ==============================
# MAIN
# ==============================
def main():
    init_firebase()

    since_ts = datetime.now() - pd.Timedelta(minutes=LOOKBACK_MIN)

    df1 = fetch_node(PATH_SENSOR, since_ts)
    df2 = fetch_node(PATH_WATERLOGS, since_ts)

    df = pd.concat([df1, df2], ignore_index=True)
    if df.empty:
        print("⚠️ No new data")
        return

    df = df.sort_values("ts").reset_index(drop=True)

    model = MLModel(ART_DIR) if USE_MODEL else None
    ref = db.reference(PATH_OUT)

    for i in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        win = df.iloc[i:i + WINDOW_SIZE]
        stats = aggregate_window(win)

        payload = {
            **stats,
            "wqi": compute_wqi(stats),
            "window_start_iso": win["ts"].iloc[0].isoformat(),
            "window_end_iso": win["ts"].iloc[-1].isoformat(),
            "created_at": datetime.utcnow().isoformat()
        }

        if model:
            cls, conf = model.predict(pd.DataFrame([stats]))
            payload["model_class"] = cls[0]
            payload["model_conf"] = float(conf[0])

        ref.push(payload)

    print("✅ WQI windows pushed")

if __name__ == "__main__":
    main()
