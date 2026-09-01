import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "feature_pipeline"))
from hopsworks_utils import get_feature_store

def upload_backfill():
    df = pd.read_csv("backfill/historical_features.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Hopsworks needs a unique primary key column
    df["id"] = df["timestamp"].astype("int64") // 10**9  # unix seconds as int PK

    fs = get_feature_store()

    aqi_fg = fs.get_or_create_feature_group(
    name="aqi_features",
    version=1,
    description="Weather and pollution features for AQI forecasting",
    primary_key=["id"],
    event_time="timestamp",
    time_travel_format="HUDI",
)

    aqi_fg.insert(df)
    print(f"Uploaded {len(df)} rows to feature group 'aqi_features'.")

if __name__ == "__main__":
    upload_backfill()