# Pearls AQI Predictor 🌫️

An end-to-end, 100% serverless machine learning system that predicts the Air Quality Index (AQI) for the next 3 days, built with a feature store, automated pipelines, and a live interactive dashboard.

**🔗 Live App:** [Add your Streamlit Cloud URL here]
**📦 Repository:** [Add your GitHub repo URL here]

---

## 📋 Overview

This project implements a full MLOps pipeline for AQI forecasting:

1. **Feature Pipeline** — fetches live weather and pollution data, engineers time-based and derived features
2. **Historical Backfill** — builds a training dataset from ~60 days of historical weather/pollution data
3. **Feature Store** — all features are stored and versioned in Hopsworks
4. **Training Pipeline** — trains and evaluates multiple ML models, registers the best one
5. **Automation** — GitHub Actions runs the feature pipeline hourly and the training pipeline daily
6. **Dashboard** — a Streamlit web app shows current AQI, a 3-day forecast, and recent trends, with hazard alerts

---

## 🏗️ Architecture

```
Weather & Pollution APIs (OpenWeather, Open-Meteo)
            │
            ▼
   Feature Pipeline (Python)
            │
            ▼
  Hopsworks Feature Store  ◄──── runs hourly via GitHub Actions
            │
            ▼
   Training Pipeline (Ridge / Random Forest)
            │
            ▼
  Hopsworks Model Registry  ◄──── runs daily via GitHub Actions
            │
            ▼
  Streamlit Dashboard (live forecast + alerts)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Weather/Pollution Data | OpenWeather API, Open-Meteo |
| Feature Store & Model Registry | Hopsworks (free tier) |
| ML Models | Scikit-learn (Ridge Regression, Random Forest) |
| Automation / CI-CD | GitHub Actions |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Deployment | Streamlit Community Cloud |

---

## 📁 Project Structure

```
aqi-predictor/
├── feature_pipeline/
│   ├── fetch_features.py       # Fetches live data, engineers features, pushes to Hopsworks
│   └── hopsworks_utils.py      # Shared Hopsworks connection helper
├── backfill/
│   ├── backfill_historical.py  # Builds historical training dataset
│   └── upload_to_hopsworks.py  # Uploads backfilled data to the feature store
├── training_pipeline/
│   └── train_model.py          # Trains, evaluates, and registers the best model
├── app/
│   └── streamlit_app.py        # Live dashboard: current AQI, 3-day forecast, trend chart, alerts
├── .github/workflows/
│   ├── feature_pipeline.yml    # Runs feature pipeline every hour
│   └── training_pipeline.yml   # Runs training pipeline daily
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd aqi-predictor
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the project root:
```
OPENWEATHER_API_KEY=your_key_here
HOPSWORKS_API_KEY=your_key_here
HOPSWORKS_PROJECT_NAME=your_project_name
```

### 4. Run the pipeline locally
```bash
# Fetch current features and push to Hopsworks
python feature_pipeline/fetch_features.py

# Backfill historical data (one-time)
python backfill/backfill_historical.py
python backfill/upload_to_hopsworks.py

# Train and register a model
python training_pipeline/train_model.py

# Launch the dashboard
streamlit run app/streamlit_app.py
```

---

## 🤖 Automation

Two GitHub Actions workflows keep the system running without manual intervention:

- **`feature_pipeline.yml`** — runs every hour, fetching the latest weather/pollution data and pushing new features to the feature store
- **`training_pipeline.yml`** — runs once daily, retraining models on the latest data and updating the model registry with the best performer

Required GitHub repository secrets: `OPENWEATHER_API_KEY`, `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT_NAME`.

---

## 📊 Model

Two models are trained and compared on a chronological train/test split (not random, to respect the time-series nature of the data):

- **Ridge Regression**
- **Random Forest Regressor**

The best-performing model (by RMSE) is automatically registered in the Hopsworks Model Registry and used by the dashboard for live 3-day-ahead predictions.

**Features used:** hour, day, month, day of week, temperature, humidity, pressure, wind speed, PM2.5, PM10, NO2, O3, CO, SO2, NH3, current AQI, AQI change rate.

**Target:** US AQI value 72 hours ahead, computed from PM2.5 concentration using EPA breakpoints.

---

## 🌐 Dashboard Features

- Current AQI with color-coded category (Good → Hazardous)
- 3-day-ahead AQI forecast
- Recent 72-hour AQI trend chart
- Automatic hazard alerts for unhealthy/hazardous air quality levels

---

## 🔮 Future Improvements

- Extend historical backfill window for improved model accuracy
- Add SHAP-based feature importance explanations
- Support multiple cities
- Add deep learning models (LSTM/TensorFlow) for comparison
- Email/push notifications for hazard alerts

---

## 🙏 Acknowledgments

Built as part of the Pearls AQI Predictor project — a serverless ML pipeline exercise covering feature engineering, MLOps, and automated deployment.
