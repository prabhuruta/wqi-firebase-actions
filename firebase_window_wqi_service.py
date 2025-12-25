#!/usr/bin/env python3
"""
firebase_window_wqi_service.py
FINAL – ML SAFE + LOCATION SAFE
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

PATH_SENSOR = "sensor_readings"
PATH_WATERLOGS = "waterLogs"
PATH_OUT = "wqi_window_results"

TIME_FIELD = "timestamp"
WINDOW_SIZE = 10
STEP_SIZE = 3
ART_DIR = "./artifacts"

# =========================
# FEATURES (TRAINING MATCH)
# =========================
FEATURE_FIELDS = [
    "pH", "dissolvedO2", "turbidity", "tds", "temp",
    "chlorophyll", "orp", "bga", "bga_temp", "chl_temp"
]

COLUMN_ALIASES = {
    "tempC": "temp",
    "bga_temp_C": "bga_temp",
    "chl_temp_C": "chl_temp",
    "chlorophyll_ug_per_L": "chlorophyll",
    "blue_green_algae_cells_per_mL": "bga"
}

# =========================
# FIREBASE INIT
# =========================
def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_JSON)
        firebase_admin.initialize_app(
            cred, {"databaseURL": FIREBASE_DB_URL}
        )
    print("✅ Firebase connected")

# =========================
# TIMESTAMP
# =========================
def parse_ts(v):
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
# AGGREGATE WINDOW
# =========================
def aggregate_window(df):
    out = {}

    # ---- sensor features ----
    for c in FEATURE_FIELDS:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            out[c] = float(s.mean()) if not s.isna().all() else 0.0
        else:
            out[c] = 0.0

    # ---- location aggregation ----
    if "lat" in df.columns and "lon" in df.columns:
        lat_s = pd.to_numeric(df["lat"], errors="coerce")
        lon_s = pd.to_numeric(df["lon"], errors="coerce")

        lat_s = lat_s[(lat_s != 0) & lat_s.notna()]
        lon_s = lon_s[(lon_s != 0) & lon_s.notna()]

        out["lat"] = float(lat_s.mean()) if len(lat_s) > 0 else None
        out["lon"] = float(lon_s.mean()) if len(lon_s) > 0 else None
    else:
        out["lat"] = None
        out["lon"] = None

    out["n_readings"] = len(df)
    return out


# =========================
# WQI
# =========================
def compute_wqi(stats):
    weights = {
        "pH": 0.2,
        "dissolvedO2": 0.2,
        "turbidity": 0.2,
        "tds": 0.2,
        "temp": 0.1,
        "chlorophyll": 0.1
    }

    wqi = 0.0
    for k, w in weights.items():
        v = stats.get(k, 0.0)
        v = 0.0 if np.isnan(v) else v
        wqi += w * min(100, max(0, v * 10))

    return round(wqi, 2)

# =========================
# ML MODEL
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
        X = np.array([[float(stats.get(c, 0.0)) for c in self.features]])
        Xs = self.scaler.transform(X)

        logprob, post = self.hmm.score_samples(Xs)
        logfeat = np.array([[logprob]])

        Xh = np.hstack([Xs, post, logfeat])

        y_raw = self.xgb.predict(Xh)
        if y_raw.ndim > 1:
            y_raw = np.argmax(y_raw, axis=1)

        conf = self.xgb.predict_proba(Xh).max(axis=1)
        cls = self.le.inverse_transform(y_raw.astype(int))[0]

        return cls, float(conf[0])

# =========================
# MAIN
# =========================
def main():
    init_firebase()

    print("📥 Fetching ALL historical data...")
    df = pd.concat([
        fetch_node(PATH_SENSOR),
        fetch_node(PATH_WATERLOGS)
    ], ignore_index=True)

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

        # 🚫 skip windows without valid location
        if stats["lat"] is None or stats["lon"] is None:
            continue

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

if __name__ == "__main__":
    main()
