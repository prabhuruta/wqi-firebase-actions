#!/usr/bin/env python3
"""
firebase_window_wqi_service.py
"""

import os, json, time, math, warnings
from datetime import datetime
import numpy as np
import pandas as pd

import firebase_admin
from firebase_admin import credentials, db
from joblib import load as joblib_load

warnings.filterwarnings("ignore")

# ======================
# ENV CONFIG
# ======================
FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL")
SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

PATH_SENSOR = os.getenv("FB_PATH_SENSOR", "sensor_readings")
PATH_WATERLOGS = os.getenv("FB_PATH_WATERLOGS", "waterLogs")
PATH_OUT = os.getenv("FB_PATH_RES_WINDOWS", "wqi_window_results")

TIME_FIELD = os.getenv("TIME_FIELD", "timestamp")
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "10"))
STEP_SIZE = int(os.getenv("STEP_SIZE", "3"))
LOOKBACK_MIN = int(os.getenv("LOOKBACK_MIN", "10"))
RUN_CONTINUOUS = os.getenv("RUN_CONTINUOUS", "false").lower() == "true"

USE_MODEL = os.getenv("USE_MODEL", "true").lower() == "true"
ART_DIR = os.getenv("ARTIFACT_DIR", "./artifacts")
ACCEPTABLE_MODEL_CLASSES = set(json.loads(os.getenv("ACCEPTABLE_MODEL_CLASSES", '["low"]')))

# ======================
# FIELDS
# ======================
FIELD_ALIASES = {
    "chlorophyll_ug_per_L": "chlorophyll",
    "blue_green_algae_cells_per_mL": "bga",
    "bga_temp_C": "bga_temp",
    "chl_temp_C": "chl_temp",
}

SENSOR_FIELDS = [
    "pH","dissolvedO2","turbidity","tds","temp",
    "chlorophyll","orp","bga","bga_temp","chl_temp","lat","lon"
]

# ======================
# FIREBASE
# ======================
def init_firebase():
    cred = credentials.Certificate(SERVICE_ACCOUNT_JSON)
    firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})

# ======================
# TIME PARSER
# ======================
def parse_ts(x):
    if pd.isna(x):
        return None
    try:
        return datetime.strptime(str(x), "%d-%m-%Y %H:%M:%S")
    except:
        try:
            return pd.to_datetime(x).to_pydatetime()
        except:
            return None

# ======================
# FETCH
# ======================
def fetch_node(path, since_ts):
    data = db.reference(path).get() or {}
    rows = []

    for v in data.values():
        if not isinstance(v, dict):
            continue
        ts_raw = v.get(TIME_FIELD)
        ts = parse_ts(ts_raw)
        if ts and ts > since_ts:
            for k, nk in FIELD_ALIASES.items():
                if k in v:
                    v[nk] = v.pop(k)
            v["ts"] = ts
            rows.append(v)

    return pd.DataFrame(rows)

# ======================
# WQI
# ======================
def compute_wqi(stats):
    score = 0
    w = 0
    def s(v): return 0 if v is None else v
    score += s(stats.get("pH")) * 0.2
    score += s(stats.get("dissolvedO2")) * 0.2
    score += s(stats.get("turbidity")) * 0.2
    score += s(stats.get("tds")) * 0.2
    score += s(stats.get("temp")) * 0.1
    score += s(stats.get("chlorophyll")) * 0.1
    return round(score / 1.0, 2)

# ======================
# WINDOW AGG
# ======================
def aggregate_window(df):
    out = {}
    for c in SENSOR_FIELDS:
        if c in df.columns:
            out[c] = float(pd.to_numeric(df[c], errors="coerce").mean())
    out["n_readings"] = len(df)
    return out

# ======================
# ML MODEL
# ======================
class MLModel:
    def __init__(self, path):
        self.scaler = joblib_load(f"{path}/scaler.joblib")
        self.hmm = joblib_load(f"{path}/hmm_model.joblib")
        self.xgb = joblib_load(f"{path}/xgb_model.joblib")
        self.le = joblib_load(f"{path}/label_encoder.joblib")
        self.order = json.load(open(f"{path}/final_feature_order.json"))
        print("✅ ML artifacts loaded")

    def predict(self, df):
        X = []
        for c in self.order:
            if c in df.columns:
                col = pd.to_numeric(df[c], errors="coerce")
            else:
                col = pd.Series([0.0] * len(df))
            col = col.fillna(0.0)
            X.append(col.values)

        X = np.column_stack(X)
        Xs = self.scaler.transform(X)

        try:
            logp, post = self.hmm.score_samples(Xs)
            logf = np.full((len(Xs),1), logp/max(len(Xs),1))
            Xh = np.hstack([Xs, post, logf])
        except:
            Xh = Xs

        y = self.xgb.predict(Xh)
        conf = self.xgb.predict_proba(Xh).max(axis=1)

        return self.le.inverse_transform(y), conf

# ======================
# MAIN
# ======================
def main():
    init_firebase()
    print("✅ Connected to Firebase")

    model = MLModel(ART_DIR) if USE_MODEL else None

    since_ts = datetime.now() - pd.Timedelta(minutes=LOOKBACK_MIN)

    df1 = fetch_node(PATH_SENSOR, since_ts)
    df2 = fetch_node(PATH_WATERLOGS, since_ts)

    df = pd.concat([df1, df2]).sort_values("ts").reset_index(drop=True)

    if len(df) < WINDOW_SIZE:
        print("Not enough data")
        return

    out_ref = db.reference(PATH_OUT)

    for i in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        win = df.iloc[i:i+WINDOW_SIZE]
        stats = aggregate_window(win)

        wqi = compute_wqi(stats)
        stats["wqi"] = wqi

        if model:
            cls, conf = model.predict(pd.DataFrame([stats]))
            stats["model_class"] = cls[0]
            stats["model_conf"] = float(conf[0])
            stats["final_drinkable"] = cls[0] in ACCEPTABLE_MODEL_CLASSES
        else:
            stats["final_drinkable"] = True

        ws = win["ts"].iloc[0]
        we = win["ts"].iloc[-1]

        payload = {
            **{k:(None if pd.isna(v) else v) for k,v in stats.items()},
            "window_start_iso": ws.isoformat(),
            "window_end_iso": we.isoformat(),
            "window_start_epoch": int(ws.timestamp()),
            "window_end_epoch": int(we.timestamp())
        }

        out_ref.push(payload)

    print("✅ Windows pushed successfully")

if __name__ == "__main__":
    main()
