from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import lfilter

st.set_page_config(page_title="GlucoseMonitor", page_icon="🩺", layout="wide")

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = APP_DIR / "data"
MODEL_PATH = APP_DIR / "glucosemonitor_models.joblib"

st.title("🩺 GlucoseMonitor")
st.caption("Research prototype | 30- and 60-minute glucose forecasting and hypoglycemia risk estimation")
st.warning(
    "Research demonstration only. This prototype is not clinically validated, "
    "must not be used for treatment decisions, and is not a medical device."
)

def decay_filter(x, half_life_min, step_min=5):
    a = 0.5 ** (step_min / half_life_min)
    return lfilter([1.0], [1.0, -a], x)

def time_since_event(s, cap=48):
    ev = (s.to_numpy() > 0)
    idx = np.where(ev, np.arange(len(s)), np.nan)
    idx = pd.Series(idx).ffill().to_numpy()
    out = np.arange(len(s)) - idx
    out = np.where(np.isnan(out), cap, out)
    return pd.Series(np.minimum(out, cap), index=s.index)

def add_features(frame):
    frame = frame.copy()
    frame = frame.sort_values(["participant_id", "time"]).reset_index(drop=True)
    g = frame.groupby("participant_id", sort=False)

    for k in [1, 2, 3, 6, 12]:
        frame[f"g_lag{k}"] = g["glucose"].shift(k)
    for k in [1, 3, 6, 12]:
        frame[f"g_diff{k}"] = frame["glucose"] - frame[f"g_lag{k}"]
    frame["g_acc"] = frame["g_diff1"] - (
        frame["g_lag1"] - g["glucose"].shift(2)
    )

    for w in [6, 12, 36]:
        frame[f"g_rmean{w}"] = g["glucose"].transform(
            lambda s: s.rolling(w, min_periods=1).mean()
        )
        frame[f"g_rstd{w}"] = g["glucose"].transform(
            lambda s: s.rolling(w, min_periods=2).std()
        )
    frame["g_min12"] = g["glucose"].transform(
        lambda s: s.rolling(12, min_periods=1).min()
    )
    frame["g_max12"] = g["glucose"].transform(
        lambda s: s.rolling(12, min_periods=1).max()
    )

    hr = frame["time"].dt.hour + frame["time"].dt.minute / 60
    frame["h_sin"] = np.sin(2 * np.pi * hr / 24)
    frame["h_cos"] = np.cos(2 * np.pi * hr / 24)

    for col, short, fn, wins in [
        ("carb_input", "carb", "sum", [6, 12, 24, 36]),
        ("bolus_volume_delivered", "bol", "sum", [6, 12, 24, 36]),
        ("basal_rate", "basal", "mean", [12, 36]),
        ("steps", "steps", "sum", [6, 12]),
        ("calories", "cal", "sum", [6, 12]),
        ("heart_rate", "hr", "mean", [6, 12]),
    ]:
        for w in wins:
            frame[f"{short}_r{w}"] = g[col].transform(
                lambda s, w=w, fn=fn: getattr(s.rolling(w, min_periods=1), fn)()
            )

    frame["carb_tsl"] = g["carb_input"].transform(time_since_event)
    frame["bol_tsl"] = g["bolus_volume_delivered"].transform(time_since_event)
    frame["iob"] = g["bolus_volume_delivered"].transform(
        lambda s: pd.Series(decay_filter(s.to_numpy(), 60), index=s.index)
    )
    frame["cob"] = g["carb_input"].transform(
        lambda s: pd.Series(decay_filter(s.to_numpy(), 30), index=s.index)
    )
    return frame

@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Run the export cell in the corrected notebook and copy the .joblib file here."
        )
    return joblib.load(MODEL_PATH)

@st.cache_data(show_spinner="Loading participant data and calculating features...")
def load_dataset(folder_string):
    folder = Path(folder_string).expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {folder}")

    files = sorted(
        f for f in folder.glob("HUPA*.csv")
        if re.fullmatch(r"HUPA\d+P", f.stem)
    )
    if not files:
        raise FileNotFoundError(f"No participant CSVs found in {folder}")

    frames = []
    for file in files:
        part = pd.read_csv(file, sep=";")
        part["participant_id"] = file.stem
        frames.append(part)

    data = pd.concat(frames, ignore_index=True)
    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    data = data.dropna(subset=["time", "glucose"])
    data = data.sort_values(["participant_id", "time"]).reset_index(drop=True)

    # Actual future outcomes are retained only for retrospective demo display.
    grouped = data.groupby("participant_id")["glucose"]
    data["actual_glucose_30m"] = grouped.shift(-6)
    data["actual_glucose_60m"] = grouped.shift(-12)
    data["actual_min_30m"] = grouped.transform(
        lambda s: s.iloc[::-1].rolling(6, min_periods=6).min().iloc[::-1].shift(-1)
    )
    data["actual_min_60m"] = grouped.transform(
        lambda s: s.iloc[::-1].rolling(12, min_periods=12).min().iloc[::-1].shift(-1)
    )

    data = add_features(data)
    return data, len(files)

try:
    bundle = load_bundle()
except Exception as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.header("Data source")
    data_dir_text = st.text_input(
        "Preprocessed dataset folder",
        value=str(DEFAULT_DATA_DIR),
        help="Folder containing the 25 HUPA participant CSV files."
    )
    load_clicked = st.button("Load / reload dataset", type="primary")

if "dataset_loaded" not in st.session_state:
    st.session_state.dataset_loaded = False

if load_clicked or not st.session_state.dataset_loaded:
    try:
        data, n_participants = load_dataset(data_dir_text)
        st.session_state.data = data
        st.session_state.n_participants = n_participants
        st.session_state.dataset_loaded = True
    except Exception as exc:
        st.error(f"Could not load dataset: {exc}")
        st.stop()

data = st.session_state.data
features = bundle["features"]

missing_features = [c for c in features if c not in data.columns]
if missing_features:
    st.error(f"Feature generation mismatch. Missing features: {missing_features}")
    st.stop()

st.success(
    f"Loaded {st.session_state.n_participants} participants | "
    f"{len(data):,} readings | {len(features)} model features"
)

eligible_data = data[(data["glucose"] >= 70)].copy()
eligible_data = eligible_data.dropna(subset=features)

if eligible_data.empty:
    st.error("No eligible readings with complete feature history were found.")
    st.stop()

left, right = st.columns([1, 1])
with left:
    participant = st.selectbox(
        "Select participant",
        sorted(eligible_data["participant_id"].unique())
    )

participant_rows = eligible_data[
    eligible_data["participant_id"] == participant
].sort_values("time").reset_index(drop=True)

with right:
    selected_idx = st.slider(
        "Select reading (chronological index)",
        min_value=0,
        max_value=len(participant_rows) - 1,
        value=max(0, len(participant_rows) - 13),
        help="Choose a historical reading. Forecasts use data available up to this timestamp."
    )

row = participant_rows.iloc[[selected_idx]]
record = row.iloc[0]
X = row[features]

predictions = {}
risks = {}
for horizon in (30, 60):
    reg = bundle["models"][f"regression_{horizon}m"]
    clf = bundle["models"][f"classifier_{horizon}m"]
    delta = float(reg.predict(X)[0])
    forecast = float(record["glucose"] + delta)
    probability = float(clf.predict_proba(X)[0, 1])
    threshold = float(bundle["thresholds"][f"{horizon}m"])
    predictions[horizon] = forecast
    risks[horizon] = (probability, threshold, probability >= threshold)

st.subheader("Selected reading")
m1, m2, m3 = st.columns(3)
m1.metric("Participant", participant)
m2.metric("Timestamp", record["time"].strftime("%d %b %Y, %H:%M"))
m3.metric("Current glucose", f"{record['glucose']:.1f} mg/dL")

st.subheader("Model predictions")
c1, c2 = st.columns(2)
for col, h in [(c1, 30), (c2, 60)]:
    probability, threshold, flagged = risks[h]
    with col:
        st.markdown(f"#### {h}-minute horizon")
        st.metric("Predicted glucose", f"{predictions[h]:.1f} mg/dL")
        st.metric("Estimated hypoglycemia risk", f"{probability*100:.1f}%")
        st.caption(f"Validation-selected alert threshold: {threshold:.3f}")
        if flagged:
            st.error("Model risk flag: ABOVE THRESHOLD")
        else:
            st.success("Model risk flag: BELOW THRESHOLD")

st.subheader("Recent glucose history")
history = data[
    (data["participant_id"] == participant) &
    (data["time"] <= record["time"])
].tail(72).set_index("time")
st.line_chart(history["glucose"], height=250)
st.caption("Up to the previous 6 hours of readings (5-minute sampling).")

with st.expander("Retrospective ground truth — for demonstration only"):
    st.caption(
        "These values come from later readings in the historical dataset. "
        "They are shown only to compare a past prediction with what subsequently happened; "
        "they are not model inputs."
    )
    actuals = {
        "30-minute actual glucose": record.get("actual_glucose_30m", np.nan),
        "60-minute actual glucose": record.get("actual_glucose_60m", np.nan),
        "Minimum glucose in next 30 min": record.get("actual_min_30m", np.nan),
        "Minimum glucose in next 60 min": record.get("actual_min_60m", np.nan),
    }
    st.dataframe(
        pd.DataFrame([
            {"Measure": key, "Value (mg/dL)": None if pd.isna(value) else round(float(value), 1)}
            for key, value in actuals.items()
        ]),
        hide_index=True,
        use_container_width=True
    )

with st.expander("About this model"):
    st.write(
        "The models use the engineered feature set from the project notebook. "
        "Regression models predict glucose change from the current reading. "
        "Classification models estimate whether glucose will fall below 70 mg/dL "
        "at any point in the next 30 or 60 minutes."
    )
    st.write("Model training metadata:")
    st.json(bundle.get("metadata", {}))

st.caption(
    "HUPA-UCM research dataset: 25 people with Type 1 diabetes. "
    "Performance on this dataset does not establish performance in other populations."
)
