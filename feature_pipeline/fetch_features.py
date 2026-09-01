import os
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

LAT, LON = 33.699619, 73.036187


def fetch_current_weather(lat, lon):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY,
        "units": "metric"
    }

    response = requests.get(url, params=params)
    return response.json()


def fetch_air_pollution(lat, lon):
    url = "https://api.openweathermap.org/data/2.5/air_pollution"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY
    }

    response = requests.get(url, params=params)
    return response.json()


weather = fetch_current_weather(LAT, LON)
pollution = fetch_air_pollution(LAT, LON)

print("Weather data:")
print(weather)

print("\nAir pollution data:")
print(pollution)

def build_feature_row(weather_json, pollution_json):
    dt = datetime.now(timezone.utc)
    components = pollution_json["list"][0]["components"]
    aqi_category = pollution_json["list"][0]["main"]["aqi"]  # 1-5 OpenWeather scale

    row = {
        "timestamp": dt.isoformat(),
        "hour": dt.hour,
        "day": dt.day,
        "month": dt.month,
        "day_of_week": dt.weekday(),
        "temp": weather_json["main"]["temp"],
        "humidity": weather_json["main"]["humidity"],
        "pressure": weather_json["main"]["pressure"],
        "wind_speed": weather_json["wind"]["speed"],
        "pm2_5": components["pm2_5"],
        "pm10": components["pm10"],
        "no2": components["no2"],
        "o3": components["o3"],
        "co": components["co"],
        "so2": components["so2"],
        "nh3": components["nh3"],
        "aqi_category": aqi_category,
    }
    return row

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
    return 500  # cap

def add_change_rate(current_row, previous_row=None):
    if previous_row is not None:
        current_row["aqi_change_rate"] = current_row["aqi_us"] - previous_row["aqi_us"]
    else:
        current_row["aqi_change_rate"] = 0
    return current_row

if __name__ == "__main__":
    weather = fetch_current_weather(LAT, LON)
    pollution = fetch_air_pollution(LAT, LON)
    row = build_feature_row(weather, pollution)
    row["aqi_us"] = pm25_to_aqi(row["pm2_5"])
    df = pd.DataFrame([row])
    print(df)

    