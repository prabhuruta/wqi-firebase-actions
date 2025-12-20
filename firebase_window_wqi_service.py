#!/usr/bin/env python3
"""
firebase_window_wqi_service.py
Window-level WQI + optional ML (HMM + XGBoost)
Safe for GitHub Actions, Firebase RTDB, and dashboards
"""

import os, time, json, math, warnings
from datetime import datetime
import numpy as np
import pandas as pd

# Firebase
import firebase_admin
from firebase_admin import credentials, db

# ML artifacts
from joblib import load as joblib_load

warnings.filterwarnings("ignore")

# ======================================================
# CONFIGURATION (ENV VARIABLES)
# ======================================================
FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL")
SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

PATH_SENSOR = os.getenv("FB_PATH_SENSOR", "sensor_readings")
PATH_WATERLOGS = os.getenv("FB_PATH_WATERLOGS", "waterLogs")
PATH_OUTPUT = os.getenv("FB_PATH_RES_WINDOWS", "wqi_window_results")

TIME_FIELD = os.getenv("TIME_FIELD", "timestamp")

WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "10"))
STEP_SIZE = int(os.getenv("STEP_SIZE", "3"))
LOOKBACK_MIN = int(os.getenv("LOOKBACK_MIN", "180"))

RUN_CONTINUOUS = os.getenv("RUN_CONTINUOUS", "false").lower() == "true"
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "30"))

USE_MODEL = os.getenv("USE_MODEL", "false").lower() == "true"
ART_DIR = os.getenv("ARTIFACT_DIR", "./artifacts")
ACCEPTABLE_MODEL_CLASSES = set(json.loads(os.getenv("ACCEPTABLE_MODEL_CLASSES", '["low"]')))

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

DRINK_LIMITS = {
    "pH": (6.5, 8.5),
    "dissolvedO2": (5.0, 100),
    "turbidity": (0, 5),
    "tds": (0, 500),
}

FIELD_ALIASES = {
    "chlorophyll_ug_per_L": "chlorophyll",
    "chl_temp_C": "chl_temp",
    "bga_temp_C": "bga_temp",
    "blue_green_algae_cells_per_mL": "bga",
}

SENSOR_FIELDS = [
    "pH", "dissolvedO2", "turbidity", "tds",
    "temp", "chlorophyll", "orp",
    "bga", "bga_temp", "chl_temp"
]

# ======================================================
# FIREBASE INIT
# ======================================================
def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_JSON)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})

# ======================================================
# TIMESTAMP PARSER (NO CONVERSION)
# ======================================================
def parse_ts(val):
    if pd.isna(val):
        return None
    s = str(val).strip()

    try:
        return datetime.strptime(s, "%d-%m-%Y %H:%M:%S")
    except:
        try:
            dt = pd.to_datetime(s, errors="coerce")
            return dt.to_pydatetime() if not pd.isna(dt) else None
        except:
            return None

# ======================================================
# FETCH DATA
# ======================================================
def fetch_node(path, since_ts=None):
    data = db.reference(path).get() or {}
    rows = [v for v in data.values() if isinstance(v, dict)]
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).rename(columns=FIELD_ALIASES)

    if TIME_FIELD not in df.columns:
        return pd.DataFrame()

    df["ts"] = df[TIME_FIELD].apply(parse_ts)
    df = df.dropna(subset=["ts"]).sort_values("ts")

    if since_ts:
        df = df[df["ts"] > since_ts]

    return df.reset_index(drop=True)

# ======================================================
# WQI FUNCTIONS
# ======================================================
def score_param(name, v):
    if v is None or np.isnan(v):
        return 0
    if name == "pH":
        return 100 if 7 <= v <= 8 else 60
    if name == "dissolvedO2":
        return min(100, v * 12.5)
    if name == "turbidity":
        return max(0, 100 - 15 * v)
    if name == "tds":
        return 100 if v <= 300 else max(0, 100 - (v - 300) * 0.1)
    if name == "temp":
        return 100 if 15 <= v <= 25 else 70
    if name == "chlorophyll":
        return max(0, 100 - v * 2)
    return 50

def compute_wqi(stats):
    total, wsum = 0, 0
    for k, w in WQI_WEIGHTS.items():
        s = score_param(k, stats.get(k))
        wsum += s * w
        total += w
    wqi = round(wsum / total, 2)
    cat = "Excellent" if wqi >= 90 else "Good" if wqi >= 70 else "Fair" if wqi >= 50 else "Poor"
    return wqi, cat

# ======================================================
# WINDOW AGGREGATION
# ======================================================
def aggregate(win):
    out = {}
    for c in SENSOR_FIELDS:
        if c in win:
            out[c] = float(pd.to_numeric(win[c], errors="coerce").mean())
    out["n_readings"] = len(win)
    return out

# ======================================================
# OPTIONAL ML
# ======================================================
class OptionalModel:
    def __init__(self, d):
        self.scaler = joblib_load(f"{d}/scaler.joblib")
        self.hmm = joblib_load(f"{d}/hmm_model.joblib")
        self.xgb = joblib_load(f"{d}/xgb_model.joblib")
        self.features = json.load(open(f"{d}/final_feature_order.json"))
        self.le = joblib_load(f"{d}/label_encoder.joblib")

    def predict(self, df):
        X = np.nan_to_num(df[self.features].values)
        Xs = self.scaler.transform(X)
        _, post = self.hmm.score_samples(Xs)
        Xh = np.hstack([Xs, post])
        y = self.xgb.predict(Xh)
        return self.le.inverse_transform(y)

# ======================================================
# MAIN PROCESS
# ======================================================
def process_once(since_ts=None):
    df1 = fetch_node(PATH_SENSOR, since_ts)
    df2 = fetch_node(PATH_WATERLOGS, since_ts)

    df = pd.concat([df1, df2]).sort_values("ts").reset_index(drop=True)
    if len(df) < WINDOW_SIZE:
        return None

    model = OptionalModel(ART_DIR) if USE_MODEL else None
    ref = db.reference(PATH_OUTPUT)

    for i in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        win = df.iloc[i:i + WINDOW_SIZE]

        stats = aggregate(win)
        wqi, cat = compute_wqi(stats)

        payload = {
            **stats,
            "wqi": wqi,
            "wqi_category": cat,
            "window_start": win["ts"].iloc[0].isoformat(),
            "window_end": win["ts"].iloc[-1].isoformat(),
            "window_center": (
                win["ts"].iloc[0] +
                (win["ts"].iloc[-1] - win["ts"].iloc[0]) / 2
            ).isoformat(),
        }

        # CLEAN NaN FOR FIREBASE
        payload = {
            k: (None if isinstance(v, float) and np.isnan(v) else v)
            for k, v in payload.items()
        }

        if model:
            cls = model.predict(pd.DataFrame([payload]))[0]
            payload["model_class"] = cls
            payload["final_drinkable"] = cls in ACCEPTABLE_MODEL_CLASSES
        else:
            payload["final_drinkable"] = True

        ref.push(payload)

    return df["ts"].max()

# ======================================================
# ENTRY POINT
# ======================================================
def main():
    init_firebase()
    since = datetime.now() - pd.Timedelta(minutes=LOOKBACK_MIN)

    if RUN_CONTINUOUS:
        while True:
            since = process_once(since) or since
            time.sleep(POLL_SECONDS)
    else:
        process_once(since)

if __name__ == "__main__":
    main()
