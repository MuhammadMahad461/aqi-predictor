import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT, LON = 34.0083, 71.5375

def fetch_pollution_history(lat, lon, start_ts, end_ts):
    url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {"lat": lat, "lon": lon, "start": start_ts, "end": end_ts, "appid": API_KEY}
    return requests.get(url, params=params).json()

def backfill_pollution(days_back=60):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    data = fetch_pollution_history(LAT, LON, int(start.timestamp()), int(end.timestamp()))
    rows = []
    for entry in data["list"]:
        c = entry["components"]
        rows.append({
            "timestamp": datetime.fromtimestamp(entry["dt"], tz=timezone.utc).isoformat(),
            "pm2_5": c["pm2_5"], "pm10": c["pm10"], "no2": c["no2"],
            "o3": c["o3"], "co": c["co"], "so2": c["so2"], "nh3": c["nh3"],
        })
    return pd.DataFrame(rows)

def backfill_weather(lat, lon, days_back=60):
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days_back)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "timezone": "UTC",
    }
    resp = requests.get(url, params=params).json()
    df = pd.DataFrame({
        "timestamp": resp["hourly"]["time"],
        "temp": resp["hourly"]["temperature_2m"],
        "humidity": resp["hourly"]["relative_humidity_2m"],
        "pressure": resp["hourly"]["surface_pressure"],
        "wind_speed": resp["hourly"]["wind_speed_10m"],
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df

def pm25_to_aqi(pm25):
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= pm25 <= c_hi:
            return round((i_hi - i_lo) / (c_hi - c_lo) * (pm25 - c_lo) + i_lo)
    return 500

def merge_and_engineer(weather_df, pollution_df):
    pollution_df["timestamp"] = pd.to_datetime(pollution_df["timestamp"], utc=True)
    weather_df = weather_df.sort_values("timestamp")
    pollution_df = pollution_df.sort_values("timestamp")

    merged = pd.merge_asof(pollution_df, weather_df, on="timestamp", direction="nearest")

    merged["hour"] = merged["timestamp"].dt.hour
    merged["day"] = merged["timestamp"].dt.day
    merged["month"] = merged["timestamp"].dt.month
    merged["day_of_week"] = merged["timestamp"].dt.dayofweek

    merged["aqi_us"] = merged["pm2_5"].apply(pm25_to_aqi)  # reuse from Step 2
    merged["aqi_change_rate"] = merged["aqi_us"].diff().fillna(0)

    return merged

def add_future_target(df, horizon_hours=72):
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["target_aqi_3d"] = df["aqi_us"].shift(-horizon_hours)
    df = df.dropna(subset=["target_aqi_3d"])  # drop rows with no future value yet
    return df

if __name__ == "__main__":

    weather_df = backfill_weather(LAT, LON, days_back=60)
    pollution_df = backfill_pollution(days_back=60)
    merged = merge_and_engineer(weather_df, pollution_df)
    final_df = add_future_target(merged, horizon_hours=72)

    final_df.to_csv("backfill/historical_features.csv", index=False)
    print(final_df.shape)
    print(final_df.head())

