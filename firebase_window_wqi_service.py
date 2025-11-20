# full_wqi_hmm_xgb_pipeline_mandatory_ml.py
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

# Optional ML artifacts
from joblib import load as joblib_load

warnings.filterwarnings("ignore")

# ==============================
# CONFIG (override with env vars)
# ==============================
FIREBASE_DB_URL = os.getenv(
    "FIREBASE_DB_URL",
    "https://mywaterproject-e6489-default-rtdb.asia-southeast1.firebasedatabase.app/"
)
SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "serviceAccountKey.json")

# DB paths
PATH_SENSOR      = os.getenv("FB_PATH_SENSOR",      "sensor_readings")
PATH_WATERLOGS   = os.getenv("FB_PATH_WATERLOGS",   "waterLogs")
PATH_RES_WINDOWS = os.getenv("FB_PATH_RES_WINDOWS", "wqi_window_results")

# Time field key present in both nodes
TIME_FIELD   = os.getenv("TIME_FIELD", "time")  # epoch ms/sec or ISO/IST/plain string
ASOF_TOL_SEC = int(os.getenv("ASOF_TOL_SEC", "30"))  # tolerance for merge_asof in seconds

# Windowing (overlap) — default: 10 readings per window, step 3
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "10"))
STEP_SIZE   = int(os.getenv("STEP_SIZE",   "3"))

# Polling / Continuous mode
RUN_CONTINUOUS = os.getenv("RUN_CONTINUOUS", "true").lower() == "true"
POLL_SECONDS   = int(os.getenv("POLL_SECONDS", "20"))
LOOKBACK_MIN   = int(os.getenv("LOOKBACK_MIN", "120"))   # initial lookback window for first pull

# ML artifacts directory (must exist)
ART_DIR   = os.getenv("ARTIFACT_DIR", "./artifacts")
# Note: ML is mandatory now; script will error if artifacts missing/invalid
ACCEPTABLE_MODEL_CLASSES = set(json.loads(os.getenv("ACCEPTABLE_MODEL_CLASSES", '["low"]')))

# Force a one-time full fetch (ignores since_ts) — helpful to verify data flow
FORCE_FULL_FETCH_ON_FIRST_RUN = os.getenv("FORCE_FULL_FETCH", "false").lower() == "true"

# ==============================
# WQI & RULES
# ==============================
WQI_WEIGHTS = {
    "pH": 0.20, "dissolvedO2": 0.20, "turbidity": 0.20,
    "tds": 0.20, "temp": 0.10, "chlorophyll": 0.10
}

DRINK_LIMITS = {
    "pH": {"min": 6.5, "max": 8.5},
    "dissolvedO2": {"min": 5.0, "max": float("inf")},
    "turbidity": {"min": 0.0, "max": 5.0},
    "tds": {"min": 0.0, "max": 500.0},
    # “soft” constraints
    "temp": {"min": 5.0, "max": 30.0},
    "chlorophyll": {"min": 0.0, "max": 30.0}
}

ALIASES = {
    "dissolved_oxygen": "dissolvedO2",
    "temperature": "temp",
    "do": "dissolvedO2",
    "chlorophyl": "chlorophyll",
}
# Extra aliases for your node names
FIELD_ALIASES_SENSOR = {
    "bga_temp_C": "bga_temp",
    "chl_temp_C": "chl_temp",
    "chlorophyll_ug_per_L": "chlorophyll",
    "blue_green_algae_cells_per_mL": "bga",  # treat as bga
}

SENSOR_FIELDS = [
    "pH", "dissolvedO2", "turbidity", "tds", "temp",
    "chlorophyll", "orp", "bga", "bga_temp", "chl_temp", "lat", "lon"
]

# ==============================
# WQI & standards helpers
# ==============================
def _score_param(name, value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    if name == "pH":
        return 100.0 if 7.0 <= value <= 8.0 else 60.0
    if name == "dissolvedO2":
        return min(100.0, value * 12.5)  # ~8 mg/L -> 100
    if name == "turbidity":
        return max(0.0, 100.0 - 15.0 * value)  # 0->100, 5->25
    if name == "tds":
        return 100.0 if value <= 300.0 else max(0.0, 100.0 - (value - 300.0) * 0.1)
    if name == "temp":
        return 100.0 if 15.0 <= value <= 25.0 else 70.0
    if name == "chlorophyll":
        return max(0.0, 100.0 - value * 2.0)
    return 50.0

def compute_wqi_from_stats(stats_row):
    # stats_row: dict of window-aggregated values (e.g., mean)
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

# ==============================
# Firebase helpers
# ==============================
def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_JSON)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
    return db

def ping_write():
    """Write a small ping doc so we know credentials/URL/path are correct."""
    try:
        ref = db.reference(PATH_RES_WINDOWS)
        ref.push({"_ping": True, "ts": pd.Timestamp.now().isoformat()})
        print(f"[DEBUG] Ping write OK → /{PATH_RES_WINDOWS}")
    except Exception as e:
        print("[ERROR] Ping write failed:", e)

def parse_any_ts(series):
    """
    Parse timestamps exactly as provided by the sensor, without timezone or UTC conversion.
    Supports plain strings like '31-08-2025 19:55:19', ISO strings, or numeric epochs.
    Returns pandas Series of naive datetime64 (interpreted exactly as recorded).
    """
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan)
            continue

        s = str(v).strip()

        # Try numeric epoch (as fallback)
        try:
            fv = float(s)
            if fv > 1e12:  # epoch ms
                out.append(datetime.fromtimestamp(fv / 1000.0))
                continue
            elif fv > 1e9:  # epoch sec
                out.append(datetime.fromtimestamp(fv))
                continue
        except Exception:
            pass

        # Try direct datetime parsing
        try:
            if "-" in s and ":" in s:
                try:
                    dt = datetime.strptime(s, "%d-%m-%Y %H:%M:%S")  # Your ESP32 format
                except ValueError:
                    dt = pd.to_datetime(s, errors="coerce")
            else:
                dt = pd.to_datetime(s, errors="coerce")
            out.append(dt if not pd.isna(dt) else np.nan)
        except Exception:
            out.append(np.nan)

    return pd.Series(out, dtype="datetime64[ns]")

def fetch_node_since(path, since_ts=None):
    ref = db.reference(path)
    data = ref.get() or {}
    if not isinstance(data, dict):
        print(f"[DEBUG] {path}: no data dict")
        return pd.DataFrame()

    rows = [p for p in data.values() if isinstance(p, dict)]
    if not rows:
        print(f"[DEBUG] {path}: zero rows")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # rename sensor fields if we are reading sensor_readings
    if path.endswith(PATH_SENSOR):
        df = df.rename(columns=FIELD_ALIASES_SENSOR)

    # pick up TIME_FIELD or common alternates
    if TIME_FIELD not in df.columns:
        for cand in ["timestamp", "time", "createdAt", "timeStamp"]:
            if cand in df.columns:
                df[TIME_FIELD] = df[cand]; break

    if TIME_FIELD not in df.columns:
        print(f"[DEBUG] {path}: no time field found")
        return pd.DataFrame()

    df["ts"] = parse_any_ts(df[TIME_FIELD])
    before = len(df)
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"[DEBUG] {path}: dropped {before - after} rows due to bad timestamps")

    # Coerce since_ts to naive pandas Timestamp before comparing
    if since_ts is not None:
        try:
            since_ts_pd = pd.Timestamp(since_ts)
        except Exception:
            since_ts_pd = pd.to_datetime(since_ts, format="%d-%m-%Y %H:%M:%S", errors="coerce")

        if (since_ts_pd is not pd.NaT) and not pd.isna(since_ts_pd):
            # strip tz if present, keep it naive
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
    left  = df1.sort_values("ts").reset_index(drop=True)
    right = df2.sort_values("ts").reset_index(drop=True)
    merged = pd.merge_asof(
        left, right, on="ts", direction="nearest",
        tolerance=pd.Timedelta(seconds=tol_sec),
        suffixes=("_s1", "_s2")
    )
    return merged

# ==============================
# Windowing
# ==============================
def sliding_windows_indices(n, size, step):
    """Yield (start_idx, end_idx_exclusive) for overlapping windows."""
    i = 0
    while i + size <= n:
        yield i, i + size
        i += step

def aggregate_window(df_slice):
    """
    Window feature aggregation (mean by default) for selected fields.
    You can add median/std/min/max if you want richer window features.
    """
    stats = {}
    # mean stats
    for col in SENSOR_FIELDS:
        if col in df_slice.columns:
            stats[col] = float(pd.to_numeric(df_slice[col], errors="coerce").astype(float).mean())
    # include variability
    for col in ["turbidity", "tds", "dissolvedO2", "pH", "temp"]:
        if col in df_slice.columns:
            stats[f"{col}_std"] = float(pd.to_numeric(df_slice[col], errors="coerce").astype(float).std(ddof=0))
    stats["n_readings"] = int(len(df_slice))
    return stats

# ==============================
# Mandatory ML prediction (window-level)
# ==============================
class OptionalModel:
    """
    Loads artifacts. Name kept for compatibility; but ML is mandatory.
    """
    def __init__(self, art_dir):
        self.ok = False
        try:
            self.scaler = joblib_load(os.path.join(art_dir, "scaler.joblib"))
            self.hmm    = joblib_load(os.path.join(art_dir, "hmm_model.joblib"))
            self.xgb    = joblib_load(os.path.join(art_dir, "xgb_model.joblib"))
            with open(os.path.join(art_dir, "final_feature_order.json"), "r") as f:
                self.feature_order = json.load(f)
            self.le = None
            le_path = os.path.join(art_dir, "label_encoder.joblib")
            if os.path.exists(le_path):
                self.le = joblib_load(le_path)
            self.ok = True
            print("✅ Loaded model artifacts.")
        except Exception as e:
            # Bubble up a clear message but keep ok=False
            print("⚠️ Model artifacts missing/incomplete:", e)
            self.ok = False

    def predict_windows(self, df_win_stats):
        if not self.ok or df_win_stats.empty:
            raise RuntimeError("Model not loaded or input empty")
        # Build feature matrix using training feature order; missing -> 0
        X_cols = []
        for col in self.feature_order:
            if col in df_win_stats.columns:
                X_cols.append(pd.to_numeric(df_win_stats[col], errors="coerce").astype(float).values)
            else:
                # alias resolve if needed
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

        # Scale
        Xs = self.scaler.transform(X)

        # HMM features
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

        if self.le is not None:
            y_name = pd.Series(self.le.inverse_transform(y_idx))
        else:
            y_name = pd.Series(y_idx)
        return y_name, pd.Series(y_prob)

# ==============================
# MAIN PROCESSOR
# ==============================
def process_once_windows(since_ts=None, last_pushed_end_ts=None, model: OptionalModel=None):
    """
    1) Pull new data from both nodes since `since_ts`
    2) Merge as-of on timestamp
    3) Build sliding windows (WINDOW_SIZE, STEP_SIZE)
    4) Aggregate stats per window; compute WQI & rule decision; (mandatory) model predict
    5) Push per-window results to Firebase at PATH_RES_WINDOWS
    Returns:
        max_ts_seen, max_window_end_ts, n_windows_pushed
    """
    print(f"[DEBUG] since_ts: {since_ts}  last_pushed_end_ts: {last_pushed_end_ts}")

    df1 = fetch_node_since(PATH_SENSOR, since_ts=since_ts)
    df2 = fetch_node_since(PATH_WATERLOGS, since_ts=since_ts)

    print(f"[DEBUG] fetched {PATH_SENSOR}: {len(df1)} rows; {PATH_WATERLOGS}: {len(df2)} rows")
    if not df1.empty:
        print(f"[DEBUG] {PATH_SENSOR} ts: {df1['ts'].min()} → {df1['ts'].max()}")
    if not df2.empty:
        print(f"[DEBUG] {PATH_WATERLOGS} ts: {df2['ts'].min()} → {df2['ts'].max()}")

    merged = merge_two_streams(df1, df2, tol_sec=ASOF_TOL_SEC)
    print(f"[DEBUG] merged rows: {len(merged)} (ASOF_TOL_SEC={ASOF_TOL_SEC})")
    if not merged.empty:
        print(f"[DEBUG] merged ts: {merged['ts'].min()} → {merged['ts'].max()}")

    if merged.empty:
        # Still advance watermark by the max of individual streams
        max_ts_seen = None
        if not df1.empty:
            max_ts_seen = df1["ts"].max()
        if not df2.empty:
            max_ts_seen = max(df2["ts"].max(), max_ts_seen) if max_ts_seen else df2["ts"].max()
        return max_ts_seen, last_pushed_end_ts, 0

    # Keep only needed columns + ts
    keep_cols = ["ts"] + [c for c in SENSOR_FIELDS if c in merged.columns]
    dfm = merged[keep_cols].copy()
    dfm = dfm.sort_values("ts").reset_index(drop=True)

    print(f"[DEBUG] dfm rows eligible for windowing: {len(dfm)}  WINDOW_SIZE={WINDOW_SIZE}, STEP={STEP_SIZE}")

    n = len(dfm)
    if n < WINDOW_SIZE:
        print(f"[DEBUG] Not enough rows for one window yet (have {n}, need {WINDOW_SIZE}).")
        return dfm["ts"].max(), last_pushed_end_ts, 0

    # Build windows
    win_rows = []
    for start, end in sliding_windows_indices(n, WINDOW_SIZE, STEP_SIZE):
        win_df = dfm.iloc[start:end]
        w_start = win_df["ts"].iloc[0]
        w_end   = win_df["ts"].iloc[-1]

        # Skip already-pushed windows (by end timestamp watermark)
        if last_pushed_end_ts is not None and w_end <= last_pushed_end_ts:
            continue

        stats = aggregate_window(win_df)  # means + stds
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
        # attach aggregated stats we care about (the means and stds)
        for k, v in stats.items():
            if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                row_out[k] = float(v)
        win_rows.append(row_out)

    print(f"[DEBUG] windows built: {len(win_rows)}")
    if win_rows:
        print(f"[DEBUG] first window: {win_rows[0]['window_start_ts']} → {win_rows[0]['window_end_ts']}")
        print(f"[DEBUG] last  window: {win_rows[-1]['window_start_ts']} → {win_rows[-1]['window_end_ts']}")

    if not win_rows:
        print("[DEBUG] No new windows to push.")
        return dfm["ts"].max(), last_pushed_end_ts, 0

    out_df = pd.DataFrame(win_rows)

    # --- Mandatory ML prediction (window-level) ---
    if model is None or not getattr(model, "ok", False):
        raise RuntimeError("ML model not loaded; cannot proceed (ML is mandatory).")
    try:
        y_name, y_conf = model.predict_windows(out_df)
        out_df["model_class"] = y_name
        out_df["model_conf"]  = y_conf
        out_df["final_drinkable"] = out_df["rule_drinkable"] & out_df["model_class"].isin(ACCEPTABLE_MODEL_CLASSES)
    except Exception as e:
        raise RuntimeError("ML prediction failed: " + str(e))

    # Push to Firebase
    out_ref = db.reference(PATH_RES_WINDOWS)
    pushed = 0
    max_end_ts = last_pushed_end_ts

    print(f"[DEBUG] pushing to RTDB path: /{PATH_RES_WINDOWS}")
    for _, r in out_df.iterrows():
        payload = r.to_dict()

        # Convert timestamps to ISO strings for Firebase payload (human readable, no tz changes)
        for key in ["window_start_ts", "window_end_ts"]:
            val = payload.get(key)
            if isinstance(val, pd.Timestamp):
                payload[key.replace("_ts","_iso")] = val.isoformat()
            elif isinstance(val, datetime):
                payload[key.replace("_ts","_iso")] = pd.Timestamp(val).isoformat()
            else:
                payload[key.replace("_ts","_iso")] = str(val)

        # keep original pd.Timestamp out of payload
        payload.pop("window_start_ts", None)
        payload.pop("window_end_ts", None)

        try:
            out_ref.push(payload)
            pushed += 1
        except Exception as e:
            print("[ERROR] push failed:", e, "payload=", payload)
            continue

        # track watermark by end ts
        cur_end = r.get("window_end_ts")
        if isinstance(cur_end, pd.Timestamp) or isinstance(cur_end, datetime):
            if (max_end_ts is None) or (cur_end > max_end_ts):
                max_end_ts = cur_end

    print(f"Pushed {pushed} window results to /{PATH_RES_WINDOWS}")
    return dfm["ts"].max(), max_end_ts, pushed

# ==============================
# MAIN
# ==============================
def main():
    # Init Firebase
    if not os.path.exists(SERVICE_ACCOUNT_JSON):
        raise RuntimeError("Service account JSON not found. Set GOOGLE_APPLICATION_CREDENTIALS env var.")
    init_firebase()
    print("Connected to Firebase:", FIREBASE_DB_URL)
    print(f"[DEBUG] Paths → sensor: /{PATH_SENSOR}, logs: /{PATH_WATERLOGS}, output: /{PATH_RES_WINDOWS}")

    # Ping write to verify permissions/path
    ping_write()

    # --- Mandatory ML model: ALWAYS load artifacts and fail fast if missing ---
    print("[INFO] Loading HMM + XGBoost model artifacts from:", ART_DIR)
    model = OptionalModel(ART_DIR)
    if not model.ok:
        raise RuntimeError(
            "ML artifacts missing or failed to load from ART_DIR. "
            "Ensure scaler.joblib, hmm_model.joblib, xgb_model.joblib, final_feature_order.json (and label_encoder.joblib optionally) are present."
        )

    # Initialize watermark as NAIVE pandas Timestamp (no tz)
    since_ts = None if FORCE_FULL_FETCH_ON_FIRST_RUN else (pd.Timestamp.now().tz_localize(None) - pd.Timedelta(minutes=LOOKBACK_MIN))
    last_pushed_end_ts = None

    if not RUN_CONTINUOUS:
        print("[DEBUG] Running single pass with ML...")
        process_once_windows(since_ts=since_ts, last_pushed_end_ts=last_pushed_end_ts, model=model)
        return

    print(f"Running continuous mode; polling every {POLL_SECONDS}s; window size={WINDOW_SIZE}, step={STEP_SIZE}")
    while True:
        try:
            max_ts_seen, last_pushed_end_ts, n = process_once_windows(
                since_ts=since_ts, last_pushed_end_ts=last_pushed_end_ts, model=model
            )
            if max_ts_seen is not None:
                since_ts = max_ts_seen  # move pull watermark forward
        except Exception as e:
            print("❌ Error in loop:", e)
            # sleep a bit on error to avoid tight loop
            time.sleep(POLL_SECONDS)
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
