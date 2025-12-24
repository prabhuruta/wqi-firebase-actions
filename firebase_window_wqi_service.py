#!/usr/bin/env python3
"""
firebase_window_wqi_service.py
FINAL STABLE VERSION
"""

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd

import firebase_admin
from firebase_admin import credentials, db
from joblib import load as joblib_load

# =========================
# ENV
# =========================
FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL")
SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

PATH_SENSOR = os.getenv("FB_PATH_SENSOR", "sensor_readings")
PATH_WATERLOGS = os.getenv("FB_PATH_WATERLOGS", "waterLogs")
PATH_OUT = os.getenv("FB_PATH_RES_WINDOWS", "wqi_window_results")

TIME_FIELD = os.getenv("TIME_FIELD", "timestamp")
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "10"))
STEP_SIZE = int(os.getenv("STEP_SIZE", "3"))

ART_DIR = os.getenv("ARTIFACT_DIR", "./artifacts")

# =========================
# FEATURE SET (TRAINING MATCH)
# =========================
FEATURE_FIELDS = [
    "pH", "dissolvedO2", "turbidity", "tds", "temp",
    "chlorophyll", "orp", "bga", "bga_temp", "chl_temp"
]

# Firebase → Model field mapping
COLUMN_ALIASES = {
    "tempC": "temp",
    "bga_temp_C": "bga_temp",
    "chl_temp_C": "chl_temp",
    "chlorophylll": "chlorophyll",
    "chlorophyll_ug_per_L": "chlorophyll",
    "blue_green_algae_cells_per_mL": "bga"
}

# =========================
# FIREBASE INIT
# =========================
def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_JSON)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
    print("✅ Firebase connected")

# =========================
# SAFE TIMESTAMP
# =========================
def parse_ts(v):
    if v is None:
        return None
    try:
        return datetime.strptime(str(v), "%d-%m-%Y %H:%M:%S")
    except Exception:
        try:
            return pd.to_datetime(v).to_pydatetime()
        except Exception:
            return None

# =========================
# FETCH ALL DATA
# =========================
def fetch_node(path):
    data = db.reference(path).get() or {}
    rows = []
    for v in data.values():
        if not isinstance(v, dict):
            continue
        ts = parse_ts(v.get(TIME_FIELD))
        if ts:
            v["ts"] = ts
            rows.append(v)
    return pd.DataFrame(rows)

# =========================
# WINDOW AGGREGATION (SAFE)
# =========================
def aggregate_window(df):
    out = {}
    for c in FEATURE_FIELDS:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            out[c] = float(s.mean()) if not s.isna().all() else 0.0
        else:
            out[c] = 0.0
    out["n_readings"] = int(len(df))
    return out

# =========================
# WQI (SIMPLE FORM)
# =========================
def compute_wqi(stats):
    weights = {
        "pH": 0.2, "dissolvedO2": 0.2, "turbidity": 0.2,
        "tds": 0.2, "temp": 0.1, "chlorophyll": 0.1
    }
    wqi = 0.0
    for k, w in weights.items():
        v = stats.get(k, 0.0)
        v = 0.0 if np.isnan(v) else v
        wqi += w * min(100.0, max(0.0, v * 10.0))
    return round(wqi, 2)

# =========================
# ML MODEL (HMM + XGB)
# =========================
class MLModel:
    def __init__(self, art_dir):
        self.scaler = joblib_load(f"{art_dir}/scaler.joblib")
        self.hmm = joblib_load(f"{art_dir}/hmm_model.joblib")
        self.xgb = joblib_load(f"{art_dir}/xgb_model.joblib")
        self.le = joblib_load(f"{art_dir}/label_encoder.joblib")
        with open(f"{art_dir}/final_feature_order.json") as f:
            self.features = json.load(f)
        print("✅ ML artifacts loaded")

    def predict(self, stats):
        X = []
        for c in self.features:
            X.append([float(stats.get(c, 0.0))])
        X = np.array(X).T

        Xs = self.scaler.transform(X)

        logprob, post = self.hmm.score_samples(Xs)
        logfeat = np.array([[logprob]])

        Xh = np.hstack([Xs, post, logfeat])

        y = self.xgb.predict(Xh)
        prob = self.xgb.predict_proba(Xh).max(axis=1)

        return self.le.inverse_transform(y)[0], float(prob[0])

# =========================
# MAIN
# =========================
def main():
    init_firebase()

    print("📥 Fetching ALL historical data...")
    df1 = fetch_node(PATH_SENSOR)
    df2 = fetch_node(PATH_WATERLOGS)

    df = pd.concat([df1, df2], ignore_index=True)
    if df.empty:
        print("⚠️ No data found")
        return

    df = df.rename(columns=COLUMN_ALIASES)
    df = df.sort_values("ts").reset_index(drop=True)

    model = MLModel(ART_DIR)
    ref = db.reference(PATH_OUT)

    pushed = 0
    for i in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        win = df.iloc[i:i + WINDOW_SIZE]
        stats = aggregate_window(win)

        wqi = compute_wqi(stats)
        cls, conf = model.predict(stats)

        payload = {
            **stats,
            "wqi": wqi,
            "model_class": cls,
            "model_conf": conf,
            "window_start_iso": win["ts"].iloc[0].isoformat(),
            "window_end_iso": win["ts"].iloc[-1].isoformat(),
            "created_at": datetime.utcnow().isoformat()
        }

        ref.push(payload)
        pushed += 1

    print(f"✅ {pushed} WQI windows pushed")

# =========================
if __name__ == "__main__":
    main()
