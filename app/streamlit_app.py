import os
import sys
import streamlit as st
import pandas as pd
import joblib
import hopsworks
from dotenv import load_dotenv
import plotly.graph_objects as go

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "feature_pipeline"))
load_dotenv()

st.set_page_config(page_title="AQI Forecast", page_icon="🌫️", layout="centered")

FEATURE_COLS = [
    "hour", "day", "month", "day_of_week",
    "temp", "humidity", "pressure", "wind_speed",
    "pm2_5", "pm10", "no2", "o3", "co", "so2", "nh3",
    "aqi_us", "aqi_change_rate",
]

@st.cache_resource
def get_project():
    return hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT_NAME"),
    )

@st.cache_resource
def load_model():
    project = get_project()
    mr = project.get_model_registry()
    model = mr.get_model("aqi_forecast_model", version=None)  # None = latest
    model_dir = model.download()
    loaded_model = joblib.load(os.path.join(model_dir, "model.pkl"))
    return loaded_model, model

@st.cache_data(ttl=1800)  # refresh every 30 min
def load_latest_features():
    project = get_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=1)
    df = fg.read()
    df = df.sort_values("timestamp", ascending=False)
    return df

def aqi_category(aqi):
    if aqi <= 50:
        return "Good", "#00e400"
    elif aqi <= 100:
        return "Moderate", "#ffff00"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif aqi <= 200:
        return "Unhealthy", "#ff0000"
    elif aqi <= 300:
        return "Very Unhealthy", "#8f3f97"
    else:
        return "Hazardous", "#7e0023"

    st.title("🌫️ AQI Forecast Dashboard")
st.caption("Predicting Air Quality Index for the next 3 days")

with st.spinner("Loading model and latest data..."):
    model, model_meta = load_model()
    df = load_latest_features()

latest = df.iloc[0]
current_aqi = latest["aqi_us"]
category, color = aqi_category(current_aqi)

col1, col2 = st.columns(2)
with col1:
    st.metric("Current AQI", f"{current_aqi:.0f}")
with col2:
    st.markdown(
        f"<div style='background-color:{color};padding:10px;border-radius:8px;text-align:center;'>"
        f"<b>{category}</b></div>",
        unsafe_allow_html=True,
    )

if current_aqi > 150:
    st.error("⚠️ Hazardous air quality detected — consider limiting outdoor activity.")
elif current_aqi > 100:
    st.warning("⚠️ Air quality may affect sensitive groups.")

X_latest = latest[FEATURE_COLS].to_frame().T
prediction = model.predict(X_latest)[0]
pred_category, pred_color = aqi_category(prediction)

st.subheader("Forecast: AQI in 3 days")
col3, col4 = st.columns(2)
with col3:
    st.metric("Predicted AQI (3d)", f"{prediction:.0f}", delta=f"{prediction - current_aqi:+.0f}")
with col4:
    st.markdown(
        f"<div style='background-color:{pred_color};padding:10px;border-radius:8px;text-align:center;'>"
        f"<b>{pred_category}</b></div>",
        unsafe_allow_html=True,
    )

    st.subheader("Recent AQI Trend")
trend_df = df.sort_values("timestamp").tail(72)  # last 72 hourly readings
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=pd.to_datetime(trend_df["timestamp"]),
    y=trend_df["aqi_us"],
    mode="lines+markers",
    name="AQI",
    line=dict(color="#1f77b4"),
))
fig.update_layout(xaxis_title="Time", yaxis_title="AQI", height=350)
st.plotly_chart(fig, use_container_width=True)

st.caption(f"Model: {model_meta.name} v{model_meta.version} | Last data update: {latest['timestamp']}")

