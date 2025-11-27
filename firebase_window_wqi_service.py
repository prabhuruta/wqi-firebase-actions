#!/usr/bin/env python3
"""
firebase_window_wqi_service.py

Reads two RTDB nodes (sensor_readings, waterLogs), aligns them as-of timestamp,
builds sliding windows, computes WQI and rule-based drinkability, optionally
runs a pre-trained HMM+XGBoost model on each window and pushes results to
/path wqi_window_results in RTDB.

Configure behavior with environment variables (see sample GitHub Actions below).
"""
import os
import time
import json
import math
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Firebase Admin
import firebase_admin
from firebase_admin import credentials, db

# Optional ML artifacts loader
from joblib import load as joblib_load

warnings.filterwarnings("ignore")

# ----------------------
# Configuration (env vars)
# ----------------------
FIREBASE_DB_URL = os.getenv(
    "FIREBASE_DB_URL",
    "https://mywaterproject-e6489-default-rtdb.asia-southeast1.firebasedatabase.app/"
)
SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "serviceAccountKey.json")

# DB nodes (exact names in your RTDB)
PATH_SENSOR      = os.getenv("FB_PATH_SENSOR", "sensor_readings")
PATH_WATERLOGS   = os.getenv("FB_PATH_WATERLOGS", "waterLogs")
PATH_RES_WINDOWS = os.getenv("FB_PATH_RES_WINDOWS", "wqi_window_results")

# timestamp field name (if your nodes use different key)
TIME_FIELD = os.getenv("TIME_FIELD", "time")  # sensor uses "timestamp" or waterLogs "time" as fallback

# as-of tolerance (matching two streams)
ASOF_TOL_SEC = int(os.getenv("ASOF_TOL_SEC", "30"))

# windowing
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "10"))
STEP_SIZE = int(os.getenv("STEP_SIZE", "3"))

# polling
RUN_CONTINUOUS = os.getenv("RUN_CONTINUOUS", "true").lower() == "true"
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "20"))
LOOKBACK_MIN = int(os.getenv("LOOKBACK_MIN", "120"))  # used only for initial since_ts if not forced

# artifacts & ML
ART_DIR = os.getenv("ARTIFACT_DIR", "./artifacts")
USE_MODEL = os.getenv("USE_MODEL", "true").lower() == "true"  # default: use ML, change to false to skip
ACCEPTABLE_MODEL_CLASSES = set(json.loads(os.getenv("ACCEPTABLE_MODEL_CLASSES", '["low"]')))

# one-time full fetch (ignore since_ts) on first run
FORCE_FULL_FETCH_ON_FIRST_RUN = os.getenv("FORCE_FULL_FETCH", "false").lower() == "true"

# ----------------------
# WQI and rules
# ----------------------
WQI_WEIGHTS = {
    "pH": 0.20, "dissolvedO2": 0.20, "turbidity": 0.20,
    "tds": 0.20, "temp": 0.10, "chlorophyll": 0.10
}
DRINK_LIMITS = {
    "pH": {"min": 6.5, "max": 8.5},
    "dissolvedO2": {"min": 5.0, "max": float("inf")},
    "turbidity": {"min": 0.0, "max": 5.0},
    "tds": {"min": 0.0, "max": 500.0},
    "temp": {"min": 5.0, "max": 30.0},
    "chlorophyll": {"min": 0.0, "max": 30.0}
}

ALIASES = {
    "dissolved_oxygen": "dissolvedO2",
    "temperature": "temp",
    "do": "dissolvedO2",
    "chlorophyl": "chlorophyll",
}

FIELD_ALIASES_SENSOR = {
    "bga_temp_C": "bga_temp",
    "chl_temp_C": "chl_temp",
    "chlorophyll_ug_per_L": "chlorophyll",
    "blue_green_algae_cells_per_mL": "bga",
}

SENSOR_FIELDS = [
    "pH", "dissolvedO2", "turbidity", "tds", "temp",
    "chlorophyll", "orp", "bga", "bga_temp", "chl_temp", "lat", "lon"
]

# ----------------------
# helpers: scoring & wqi
# ----------------------
def _score_param(name, value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    if name == "pH":
        return 100.0 if 7.0 <= value <= 8.0 else 60.0
    if name == "dissolvedO2":
        return min(100.0, value * 12.5)
    if name == "turbidity":
        return max(0.0, 100.0 - 15.0 * value)
    if name == "tds":
        return 100.0 if value <= 300.0 else max(0.0, 100.0 - (value - 300.0) * 0.1)
    if name == "temp":
        return 100.0 if 15.0 <= value <= 25.0 else 70.0
    if name == "chlorophyll":
        return max(0.0, 100.0 - value * 2.0)
    return 50.0

def compute_wqi_from_stats(stats_row):
    accum = 0.0
    total = 0.0
    param_scores = {}
    for k, w in WQI_WEIGHTS.items():
        v = stats_row.get(k, None)
        if v is None:
            for a, canonical in ALIASES.items():
                if canonical == k and a in stats_row:
                    v = stats_row[a]
                    break
        v = float(v) if v is not None else None
        s = _score_param(k, v)
        param_scores[k] = s
        accum += w * s
        total += w
    wqi = accum / total if total > 0 else 0.0
    category = "Excellent" if wqi >= 90 else "Good" if wqi >= 70 else "Fair" if wqi >= 50 else "Poor"
    return {"wqi": round(wqi, 2), "category": category, "param_scores": param_scores}

def standards_check_from_stats(stats_row):
    violations = []
    for k, rng in DRINK_LIMITS.items():
        v = stats_row.get(k, None)
        if v is None:
            for a, canonical in ALIASES.items():
                if canonical == k and a in stats_row:
                    v = stats_row[a]
                    break
        if v is None:
            violations.append(f"{k}:missing")
            continue
        vf = float(v)
        if vf < rng["min"]:
            violations.append(f"{k}:below({vf}<{rng['min']})")
        if vf > rng["max"]:
            violations.append(f"{k}:above({vf}>{rng['max']})")
    core = ["pH", "dissolvedO2", "turbidity", "tds"]
    hard = [vi for vi in violations if any(vi.startswith(cp + ":") for cp in core)]
    return (len(hard) == 0), violations

# ----------------------
# Firebase helpers
# ----------------------
def init_firebase():
    if not firebase_admin._apps:
        if not os.path.exists(SERVICE_ACCOUNT_JSON):
            raise RuntimeError(f"Service account JSON not found at '{SERVICE_ACCOUNT_JSON}'")
        cred = credentials.Certificate(SERVICE_ACCOUNT_JSON)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
    return db

def ping_write():
    try:
        ref = db.reference(PATH_RES_WINDOWS)
        ref.push({"_ping": True, "ts": pd.Timestamp.now().isoformat()})
        print(f"[DEBUG] Ping write OK → /{PATH_RES_WINDOWS}")
    except Exception as e:
        print("[ERROR] Ping write failed:", e)
        raise

def parse_any_ts(series):
    """
    Parse timestamps exactly as provided (no tz conversion).
    Accepts:
      - epoch seconds or ms (numeric)
      - 'dd-mm-YYYY HH:MM:SS' (ESP32 format used in screenshots)
      - ISO strings (pd.to_datetime fallback)
    Returns pandas Series dtype datetime64[ns] (naive).
    """
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan); continue
        s = str(v).strip()
        # numeric epoch?
        try:
            fv = float(s)
            if fv > 1e12:   # ms
                out.append(datetime.fromtimestamp(fv / 1000.0))
                continue
            if fv > 1e9:    # seconds
                out.append(datetime.fromtimestamp(fv))
                continue
        except Exception:
            pass
        # known dd-mm-YYYY HH:MM:SS
        try:
            if "-" in s and ":" in s and len(s.split(" ")[0].split("-")) == 3:
                try:
                    dt = datetime.strptime(s, "%d-%m-%Y %H:%M:%S")
                    out.append(dt); continue
                except Exception:
                    pass
            # fallback to pandas parsing
            dt = pd.to_datetime(s, errors="coerce")
            if pd.isna(dt):
                out.append(np.nan)
            else:
                # convert to python datetime (naive)
                dt = dt.to_pydatetime()
                if hasattr(dt, "tzinfo") and dt.tzinfo is not None:
                    # drop tz (user asked to keep timestamp as-is; preserve value but drop tz)
                    dt = dt.replace(tzinfo=None)
                out.append(dt)
        except Exception:
            out.append(np.nan)
    return pd.Series(out, dtype="datetime64[ns]")

def fetch_node_since(path, since_ts=None):
    ref = db.reference(path)
    data = ref.get() or {}
    if not isinstance(data, dict):
        print(f"[DEBUG] {path}: returned non-dict (got {type(data)})")
        return pd.DataFrame()
    rows = [p for p in data.values() if isinstance(p, dict)]
    if not rows:
        print(f"[DEBUG] {path}: zero rows in node")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # rename sensor columns if needed
    if path.endswith(PATH_SENSOR):
        df = df.rename(columns=FIELD_ALIASES_SENSOR)
    # detect time field
    if TIME_FIELD not in df.columns:
        for cand in ["timestamp", "time", "createdAt", "timeStamp"]:
            if cand in df.columns:
                df[TIME_FIELD] = df[cand]; break
    if TIME_FIELD not in df.columns:
        print(f"[DEBUG] {path}: no time field found among columns {df.columns.tolist()}")
        return pd.DataFrame()
    df["ts"] = parse_any_ts(df[TIME_FIELD])
    before = len(df)
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"[DEBUG] {path}: dropped {before-after} rows due to invalid timestamps")
    if since_ts is not None:
        try:
            since_ts_pd = pd.Timestamp(since_ts)
        except Exception:
            since_ts_pd = pd.to_datetime(since_ts, errors="coerce")
        if not pd.isna(since_ts_pd):
            # make naive (strip tz) for comparison
            try:
                since_ts_pd = since_ts_pd.tz_localize(None)
            except Exception:
                pass
            df = df[df["ts"] > since_ts_pd]
    return df

def merge_two_streams(df1, df2, tol_sec=5):
    if df1.empty and df2.empty:
        return pd.DataFrame()
    if df1.empty:
        return df2.copy()
    if df2.empty:
        return df1.copy()
    left = df1.sort_values("ts").reset_index(drop=True)
    right = df2.sort_values("ts").reset_index(drop=True)
    merged = pd.merge_asof(
        left, right, on="ts", direction="nearest",
        tolerance=pd.Timedelta(seconds=tol_sec),
        suffixes=("_s1", "_s2")
    )
    return merged

# ----------------------
# window utilities
# ----------------------
def sliding_windows_indices(n, size, step):
    i = 0
    while i + size <= n:
        yield i, i + size
        i += step

def aggregate_window(df_slice):
    stats = {}
    for col in SENSOR_FIELDS:
        if col in df_slice.columns:
            stats[col] = float(pd.to_numeric(df_slice[col], errors="coerce").astype(float).mean())
    for col in ["turbidity", "tds", "dissolvedO2", "pH", "temp"]:
        if col in df_slice.columns:
            stats[f"{col}_std"] = float(pd.to_numeric(df_slice[col], errors="coerce").astype(float).std(ddof=0))
    stats["n_readings"] = int(len(df_slice))
    return stats

# ----------------------
# Optional model wrapper (expects artifacts saved by training pipeline)
# ----------------------
class OptionalModel:
    def __init__(self, art_dir):
        self.ok = False
        try:
            self.scaler = joblib_load(os.path.join(art_dir, "scaler.joblib"))
            self.hmm = joblib_load(os.path.join(art_dir, "hmm_model.joblib"))
            self.xgb = joblib_load(os.path.join(art_dir, "xgb_model.joblib"))
            with open(os.path.join(art_dir, "final_feature_order.json"), "r") as f:
                self.feature_order = json.load(f)
            le_path = os.path.join(art_dir, "label_encoder.joblib")
            self.le = joblib_load(le_path) if os.path.exists(le_path) else None
            self.ok = True
            print("[INFO] Loaded model artifacts from:", art_dir)
        except Exception as e:
            raise RuntimeError(f"Model artifacts missing or incomplete in '{art_dir}': {e}")

    def predict_windows(self, df_win_stats):
        if not self.ok or df_win_stats.empty:
            return pd.Series([None]*len(df_win_stats)), pd.Series([None]*len(df_win_stats))
        X_cols = []
        for col in self.feature_order:
            if col in df_win_stats.columns:
                X_cols.append(pd.to_numeric(df_win_stats[col], errors="coerce").astype(float).values)
            else:
                alias_src = None
                for a, canonical in ALIASES.items():
                    if canonical == col and a in df_win_stats.columns:
                        alias_src = a; break
                if alias_src:
                    X_cols.append(pd.to_numeric(df_win_stats[alias_src], errors="coerce").astype(float).values)
                else:
                    X_cols.append(np.zeros(len(df_win_stats)))
        X = np.column_stack(X_cols).astype(float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Xs = self.scaler.transform(X)
        try:
            logprob, post = self.hmm.score_samples(Xs)
            n = max(1, Xs.shape[0])
            logfeat = np.full((n, 1), logprob / n)
            Xh = np.hstack([Xs, post, logfeat])
        except Exception:
            n_states = getattr(self.hmm, "n_components", 4)
            Xh = np.hstack([Xs, np.zeros((len(Xs), n_states)), np.zeros((len(Xs), 1))])
        y_idx = self.xgb.predict(Xh)
        try:
            y_prob = self.xgb.predict_proba(Xh).max(axis=1)
        except Exception:
            y_prob = np.full(len(Xh), np.nan)
        y_name = pd.Series(self.le.inverse_transform(y_idx)) if self.le is not None else pd.Series(y_idx)
        return y_name, pd.Series(y_prob)

# ----------------------
# core processing
# ----------------------
def process_once_windows(since_ts=None, last_pushed_end_ts=None, model: OptionalModel=None):
    print(f"[DEBUG] since_ts: {since_ts}  last_pushed_end_ts: {last_pushed_end_ts}")
    df1 = fetch_node_since(PATH_SENSOR, since_ts=since_ts)
    df2 = fetch_node_since(PATH_WATERLOGS, since_ts=since_ts)
    print(f"[DEBUG] fetched {PATH_SENSOR}: {len(df1)} rows; {PATH_WATERLOGS}: {len(df2)} rows")
    if not df1.empty:
        print(f"[DEBUG] {PATH_SENSOR} ts: {df1['ts'].min()} → {df1['ts'].max()}")
    if not df2.empty:
        print(f"[DEBUG] {PATH_WATERLOGS} ts: {df2['ts'].min()} → {df2['ts'].max()}")

    # fallback: if both empty and we had a since_ts, try one full fetch (helpful for first-run)
    if df1.empty and df2.empty and since_ts is not None:
        print("[DEBUG] No rows since since_ts -> retrying with full fetch (no watermark).")
        df1 = fetch_node_since(PATH_SENSOR, since_ts=None)
        df2 = fetch_node_since(PATH_WATERLOGS, since_ts=None)
        print(f"[DEBUG] full fetch returned {len(df1)} rows and {len(df2)} rows")

    merged = merge_two_streams(df1, df2, tol_sec=ASOF_TOL_SEC)
    print(f"[DEBUG] merged rows: {len(merged)} (ASOF_TOL_SEC={ASOF_TOL_SEC})")
    if merged.empty:
        max_ts_seen = None
        if not df1.empty:
            max_ts_seen = df1["ts"].max()
        if not df2.empty:
            max_ts_seen = max(df2["ts"].max(), max_ts_seen) if max_ts_seen else df2["ts"].max()
        return max_ts_seen, last_pushed_end_ts, 0

    keep_cols = ["ts"] + [c for c in SENSOR_FIELDS if c in merged.columns]
    dfm = merged[keep_cols].copy().sort_values("ts").reset_index(drop=True)
    n = len(dfm)
    if n < WINDOW_SIZE:
        print(f"[DEBUG] Not enough rows for one window yet (have {n}, need {WINDOW_SIZE}).")
        return dfm["ts"].max(), last_pushed_end_ts, 0

    win_rows = []
    for start, end in sliding_windows_indices(n, WINDOW_SIZE, STEP_SIZE):
        win_df = dfm.iloc[start:end]
        w_start = win_df["ts"].iloc[0]
        w_end = win_df["ts"].iloc[-1]
        if last_pushed_end_ts is not None and w_end <= last_pushed_end_ts:
            continue
        stats = aggregate_window(win_df)
        wqi = compute_wqi_from_stats(stats)
        ok_rules, viols = standards_check_from_stats(stats)
        row_out = {
            "window_start_ts": w_start,
            "window_end_ts": w_end,
            "n_readings": stats.get("n_readings", len(win_df)),
            "wqi": wqi["wqi"],
            "wqi_category": wqi["category"],
            "rule_drinkable": ok_rules,
            "rule_violations": ";".join(viols) if viols else ""
        }
        for k, v in stats.items():
            if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                row_out[k] = float(v)
        win_rows.append(row_out)

    if not win_rows:
        print("[DEBUG] No new windows to push.")
        return dfm["ts"].max(), last_pushed_end_ts, 0

    out_df = pd.DataFrame(win_rows)

    if model is not None:
        y_name, y_conf = model.predict_windows(out_df)
        out_df["model_class"] = y_name
        out_df["model_conf"] = y_conf
        out_df["final_drinkable"] = out_df["rule_drinkable"] & out_df["model_class"].isin(ACCEPTABLE_MODEL_CLASSES)
    else:
        out_df["model_class"] = None
        out_df["model_conf"] = None
        out_df["final_drinkable"] = out_df["rule_drinkable"]

    out_ref = db.reference(PATH_RES_WINDOWS)
    pushed = 0
    max_end_ts = last_pushed_end_ts
    for _, r in out_df.iterrows():
        payload = r.to_dict()
        for key in ["window_start_ts", "window_end_ts"]:
            val = payload.get(key)
            if isinstance(val, pd.Timestamp):
                payload[key.replace("_ts", "_iso")] = val.isoformat()
            elif isinstance(val, datetime):
                payload[key.replace("_ts", "_iso")] = pd.Timestamp(val).isoformat()
            else:
                payload[key.replace("_ts", "_iso")] = str(val)
        payload.pop("window_start_ts", None)
        payload.pop("window_end_ts", None)
        try:
            out_ref.push(payload)
            pushed += 1
        except Exception as e:
            print("[ERROR] push failed:", e, "payload=", payload)
            continue
        cur_end = r.get("window_end_ts")
        if isinstance(cur_end, pd.Timestamp) or isinstance(cur_end, datetime):
            if (max_end_ts is None) or (cur_end > max_end_ts):
                max_end_ts = cur_end
    print(f"Pushed {pushed} window results to /{PATH_RES_WINDOWS}")
    return dfm["ts"].max(), max_end_ts, pushed

# ----------------------
# main
# ----------------------
def main():
    init_firebase()
    print("Connected to Firebase:", FIREBASE_DB_URL)
    print(f"[DEBUG] Paths → sensor: /{PATH_SENSOR}, logs: /{PATH_WATERLOGS}, output: /{PATH_RES_WINDOWS}")
    ping_write()

    # load model if requested
    model = None
    if USE_MODEL:
        try:
            print(f"[INFO] Loading HMM + XGBoost model artifacts from: {ART_DIR}")
            model = OptionalModel(ART_DIR)
        except Exception as e:
            print("[ERROR] Failed to load model artifacts:", e)
            raise

    # initialize watermarks
    since_ts = None if FORCE_FULL_FETCH_ON_FIRST_RUN else (pd.Timestamp.now().tz_localize(None) - pd.Timedelta(minutes=LOOKBACK_MIN))
    last_pushed_end_ts = None

    if not RUN_CONTINUOUS:
        print("[DEBUG] Running single pass...")
        process_once_windows(since_ts=since_ts, last_pushed_end_ts=last_pushed_end_ts, model=model)
        return

    print(f"Running continuous mode; polling every {POLL_SECONDS}s; window size={WINDOW_SIZE}, step={STEP_SIZE}")
    while True:
        try:
            max_ts_seen, last_pushed_end_ts, n = process_once_windows(
                since_ts=since_ts, last_pushed_end_ts=last_pushed_end_ts, model=model
            )
            if max_ts_seen is not None:
                since_ts = max_ts_seen
        except Exception as e:
            print("❌ Error in loop:", e)
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
