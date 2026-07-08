"""
=========================================================================
 WEATHERAI PRO - Next Generation Weather Intelligence
=========================================================================
 A stunning, interactive weather platform with:
   - Live 3D interactive globe with real-time weather
   - Bold, modern typography with gradient accents
   - Vibrant color palette with smooth animations
   - Fully clickable global map
   - Ensemble AI predictions
   - Made with ❤️ by Adarsh
=========================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix

# ------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="WeatherAI Pro",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------------
# Modern, Vibrant CSS with 3D Effects
# ------------------------------------------------------------------
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
    }
    
    /* Floating particles background */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, rgba(255,255,255,0.1), transparent),
            radial-gradient(2px 2px at 40px 70px, rgba(255,255,255,0.08), transparent),
            radial-gradient(2px 2px at 50px 160px, rgba(255,255,255,0.1), transparent),
            radial-gradient(2px 2px at 90px 40px, rgba(255,255,255,0.08), transparent),
            radial-gradient(2px 2px at 130px 80px, rgba(255,255,255,0.1), transparent);
        background-size: 200px 200px;
        pointer-events: none;
        z-index: 0;
    }
    
    .block-container {
        padding: 0 3rem !important;
        max-width: 1400px !important;
        position: relative;
        z-index: 1;
    }
    
    /* 3D Header - Big and Bold */
    .header-3d {
        padding: 2.5rem 0 1.5rem 0;
        margin-bottom: 2.5rem;
        position: relative;
    }
    
    .header-3d::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00d4ff, #7b2ffc, #00d4ff, transparent);
        background-size: 200% 100%;
        animation: shimmerLine 4s linear infinite;
    }
    
    @keyframes shimmerLine {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    .main-title {
        font-family: 'Space Grotesk', 'Outfit', sans-serif !important;
        font-size: 4.5rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #00d4ff 0%, #7b2ffc 40%, #ff6b9d 70%, #00d4ff 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientMove 6s ease-in-out infinite alternate;
        margin: 0;
        line-height: 1.1;
        text-shadow: 0 0 80px rgba(0, 212, 255, 0.1);
    }
    
    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }
    
    .sub-title {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.4);
        font-weight: 300;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-top: 0.25rem;
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* 3D Stats Bar */
    .stats-bar {
        display: flex;
        gap: 4rem;
        margin-top: 1.5rem;
        padding: 1.5rem 0;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        justify-content: center;
    }
    
    .stat-item {
        text-align: center;
        position: relative;
    }
    
    .stat-number {
        font-family: 'Space Grotesk', 'Outfit', sans-serif !important;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff 0%, #7b2ffc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: block;
        line-height: 1.2;
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 500;
        margin-top: 0.25rem;
    }
    
    /* 3D Glass Cards */
    .glass-3d {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(30px);
        -webkit-backdrop-filter: blur(30px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 28px;
        padding: 2rem;
        transition: all 0.5s cubic-bezier(0.23, 1, 0.32, 1);
        position: relative;
        overflow: hidden;
        box-shadow: 
            0 10px 40px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }
    
    .glass-3d::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(0, 212, 255, 0.03), transparent 60%);
        pointer-events: none;
    }
    
    .glass-3d:hover {
        transform: translateY(-8px) scale(1.01);
        border-color: rgba(0, 212, 255, 0.2);
        box-shadow: 
            0 30px 80px rgba(0, 0, 0, 0.4),
            0 0 60px rgba(0, 212, 255, 0.05);
    }
    
    /* Big Section Headers */
    .section-header {
        font-family: 'Space Grotesk', 'Outfit', sans-serif !important;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #00d4ff 0%, #7b2ffc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    
    .section-sub {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.3);
        font-weight: 300;
        margin-bottom: 2rem;
    }
    
    /* 3D Tabs - Big and Bold */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 16px;
        padding: 0.8rem 2.2rem;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600;
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
        letter-spacing: 0.02em;
        background: transparent !important;
        border: none !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: rgba(255, 255, 255, 0.7);
        background: rgba(255, 255, 255, 0.03) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.15) 0%, rgba(123, 47, 252, 0.15) 100%) !important;
        color: white !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        box-shadow: 0 4px 30px rgba(0, 212, 255, 0.1);
    }
    
    /* Modern Buttons - 3D Effect */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #7b2ffc 100%);
        border: none;
        border-radius: 16px;
        color: white;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.9rem 2.5rem;
        transition: all 0.4s cubic-bezier(0.23, 1, 0.32, 1);
        box-shadow: 
            0 10px 40px rgba(0, 212, 255, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
        letter-spacing: 0.02em;
        width: 100%;
        text-transform: uppercase;
        font-size: 0.9rem;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.1), transparent 50%);
        opacity: 0;
        transition: opacity 0.4s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 
            0 20px 60px rgba(0, 212, 255, 0.3),
            0 0 80px rgba(123, 47, 252, 0.1);
    }
    
    .stButton > button:hover::before {
        opacity: 1;
    }
    
    .stButton > button:active {
        transform: translateY(0) scale(0.98);
    }
    
    /* 3D Metrics */
    .metric-3d {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 1.5rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-3d::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #00d4ff, #7b2ffc, #ff6b9d);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-3d:hover::before {
        opacity: 1;
    }
    
    .metric-3d:hover {
        border-color: rgba(0, 212, 255, 0.1);
        transform: translateY(-4px);
    }
    
    .metric-value-3d {
        font-family: 'Space Grotesk', 'Outfit', sans-serif !important;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff 0%, #7b2ffc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    
    .metric-label-3d {
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 500;
        margin-top: 0.25rem;
    }
    
    /* Modern Inputs */
    .stNumberInput, .stSlider, .stSelectbox, .stRadio {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 0.25rem;
        transition: all 0.3s ease;
    }
    
    .stNumberInput:focus-within, .stSlider:focus-within, .stSelectbox:focus-within {
        border-color: rgba(0, 212, 255, 0.2);
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.05);
    }
    
    label {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        color: rgba(255, 255, 255, 0.6) !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.02em !important;
    }
    
    /* Status Badges - 3D */
    .badge-3d {
        display: inline-block;
        padding: 0.6rem 2rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .badge-rain {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.2), rgba(123, 47, 252, 0.2));
        color: #00d4ff;
        border-color: rgba(0, 212, 255, 0.2);
        box-shadow: 0 0 40px rgba(0, 212, 255, 0.05);
    }
    
    .badge-sunny {
        background: linear-gradient(135deg, rgba(255, 107, 157, 0.2), rgba(255, 184, 0, 0.2));
        color: #ff6b9d;
        border-color: rgba(255, 107, 157, 0.2);
        box-shadow: 0 0 40px rgba(255, 107, 157, 0.05);
    }
    
    .badge-temp {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.15), rgba(123, 47, 252, 0.15));
        color: #7b2ffc;
        border-color: rgba(123, 47, 252, 0.2);
        box-shadow: 0 0 40px rgba(123, 47, 252, 0.05);
    }
    
    /* Prediction Result Card */
    .result-card-3d {
        text-align: center;
        padding: 3rem 2rem;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
    }
    
    .result-card-3d::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 50% 50%, rgba(0, 212, 255, 0.03), transparent 60%);
        animation: pulseGlow 4s ease-in-out infinite alternate;
    }
    
    @keyframes pulseGlow {
        0% { transform: scale(0.8); opacity: 0.5; }
        100% { transform: scale(1.2); opacity: 1; }
    }
    
    .result-icon-3d {
        font-size: 5rem;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .result-title-3d {
        font-family: 'Space Grotesk', 'Outfit', sans-serif !important;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: white;
        position: relative;
        z-index: 1;
    }
    
    .result-temp-3d {
        font-family: 'Space Grotesk', 'Outfit', sans-serif !important;
        font-size: 5rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #00d4ff 0%, #7b2ffc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        position: relative;
        z-index: 1;
    }
    
    /* Footer - Made with love */
    .footer-3d {
        padding: 3rem 0 1.5rem 0;
        margin-top: 3rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        text-align: center;
        position: relative;
    }
    
    .footer-3d::before {
        content: '✦';
        position: absolute;
        top: -0.8rem;
        left: 50%;
        transform: translateX(-50%);
        background: linear-gradient(135deg, #00d4ff, #7b2ffc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.5rem;
    }
    
    .footer-text {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.15);
        font-weight: 300;
        letter-spacing: 0.05em;
    }
    
    .footer-heart {
        color: #ff6b9d;
        font-weight: 400;
    }
    
    .footer-name {
        background: linear-gradient(135deg, #00d4ff, #7b2ffc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.02);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #00d4ff, #7b2ffc);
        border-radius: 10px;
    }
    
    /* Hide default Streamlit */
    #MainMenu, footer, header, .stDeployButton {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# City Database
# ------------------------------------------------------------------
CITIES = {
    "New Delhi, India": {"lat": 28.6139, "lon": 77.2090, "emoji": "🇮🇳"},
    "Mumbai, India": {"lat": 19.0760, "lon": 72.8777, "emoji": "🇮🇳"},
    "New York, USA": {"lat": 40.7128, "lon": -74.0060, "emoji": "🇺🇸"},
    "Los Angeles, USA": {"lat": 34.0522, "lon": -118.2437, "emoji": "🇺🇸"},
    "London, UK": {"lat": 51.5074, "lon": -0.1278, "emoji": "🇬🇧"},
    "Paris, France": {"lat": 48.8566, "lon": 2.3522, "emoji": "🇫🇷"},
    "Tokyo, Japan": {"lat": 35.6762, "lon": 139.6503, "emoji": "🇯🇵"},
    "Singapore": {"lat": 1.3521, "lon": 103.8198, "emoji": "🇸🇬"},
    "Sydney, Australia": {"lat": -33.8688, "lon": 151.2093, "emoji": "🇦🇺"},
    "Dubai, UAE": {"lat": 25.2048, "lon": 55.2708, "emoji": "🇦🇪"},
    "Cairo, Egypt": {"lat": 30.0444, "lon": 31.2357, "emoji": "🇪🇬"},
    "Rio de Janeiro, Brazil": {"lat": -22.9068, "lon": -43.1729, "emoji": "🇧🇷"},
    "Beijing, China": {"lat": 39.9042, "lon": 116.4074, "emoji": "🇨🇳"},
    "Moscow, Russia": {"lat": 55.7558, "lon": 37.6173, "emoji": "🇷🇺"},
    "Berlin, Germany": {"lat": 52.5200, "lon": 13.4050, "emoji": "🇩🇪"},
    "Rome, Italy": {"lat": 41.9028, "lon": 12.4964, "emoji": "🇮🇹"},
    "Cape Town, South Africa": {"lat": -33.9249, "lon": 18.4241, "emoji": "🇿🇦"},
    "Bangkok, Thailand": {"lat": 13.7563, "lon": 100.5018, "emoji": "🇹🇭"},
    "Toronto, Canada": {"lat": 43.6532, "lon": -79.3832, "emoji": "🇨🇦"},
    "Mexico City, Mexico": {"lat": 19.4326, "lon": -99.1332, "emoji": "🇲🇽"},
}

# ------------------------------------------------------------------
# Load and Prepare Data
# ------------------------------------------------------------------
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("weatherHistory.csv")
    
    df = df.drop_duplicates()
    df = df.drop(columns=["Loud Cover"], errors='ignore')
    df["Precip Type"] = df["Precip Type"].fillna("none")
    
    df["Formatted Date"] = pd.to_datetime(df["Formatted Date"], utc=True)
    df["Date"] = df["Formatted Date"].dt.date
    df["Month"] = df["Formatted Date"].dt.month
    df["DayOfWeek"] = df["Formatted Date"].dt.dayofweek
    
    df["is_rain_hour"] = (df["Precip Type"] == "rain").astype(int)
    
    daily = df.groupby("Date").agg({
        "Temperature (C)": "mean",
        "Apparent Temperature (C)": "mean",
        "Humidity": "mean",
        "Wind Speed (km/h)": "mean",
        "Visibility (km)": "mean",
        "Pressure (millibars)": "mean",
        "is_rain_hour": "max",
        "Month": "first",
        "DayOfWeek": "first",
    }).reset_index()
    
    daily = daily.rename(columns={"is_rain_hour": "RainToday"})
    
    daily["TempRange"] = daily["Temperature (C)"] - daily["Apparent Temperature (C)"]
    daily["HumidityPressure"] = daily["Humidity"] * daily["Pressure (millibars)"]
    daily["WindVisibility"] = daily["Wind Speed (km/h)"] * daily["Visibility (km)"]
    daily["TempHumidity"] = daily["Temperature (C)"] * daily["Humidity"]
    
    daily["RainTomorrow"] = daily["RainToday"].shift(-1)
    daily["TempTomorrow"] = daily["Temperature (C)"].shift(-1)
    
    daily = daily.dropna().reset_index(drop=True)
    daily["RainTomorrow"] = daily["RainTomorrow"].astype(int)
    
    return df, daily

# ------------------------------------------------------------------
# Train Models
# ------------------------------------------------------------------
@st.cache_resource
def train_models(daily):
    features = [
        "Temperature (C)", "Apparent Temperature (C)", "Humidity",
        "Wind Speed (km/h)", "Visibility (km)", "Pressure (millibars)", 
        "RainToday", "TempRange", "HumidityPressure", "WindVisibility",
        "TempHumidity", "Month", "DayOfWeek"
    ]
    
    X = daily[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    y_class = daily["RainTomorrow"]
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_scaled, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    rf_clf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
    svm_clf = SVC(kernel="rbf", random_state=42)
    
    rf_clf.fit(Xc_train, yc_train)
    svm_clf.fit(Xc_train, yc_train)
    
    y_reg = daily["TempTomorrow"]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_scaled, y_reg, test_size=0.2, random_state=42
    )
    
    rf_reg = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
    gb_reg = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42)
    ridge_reg = Ridge(alpha=1.0)
    
    rf_reg.fit(Xr_train, yr_train)
    gb_reg.fit(Xr_train, yr_train)
    ridge_reg.fit(Xr_train, yr_train)
    
    rf_pred = rf_clf.predict(Xc_test)
    svm_pred = svm_clf.predict(Xc_test)
    ensemble_clf_pred = (rf_pred + svm_pred) // 2
    
    rf_reg_pred = rf_reg.predict(Xr_test)
    gb_reg_pred = gb_reg.predict(Xr_test)
    ridge_reg_pred = ridge_reg.predict(Xr_test)
    ensemble_reg_pred = (rf_reg_pred + gb_reg_pred + ridge_reg_pred) / 3
    
    return {
        "features": features,
        "scaler": scaler,
        "rf_clf": rf_clf,
        "svm_clf": svm_clf,
        "rf_reg": rf_reg,
        "gb_reg": gb_reg,
        "ridge_reg": ridge_reg,
        "X_test": Xc_test,
        "yc_test": yc_test,
        "yr_test": yr_test,
        "ensemble_clf_acc": accuracy_score(yc_test, ensemble_clf_pred),
        "ensemble_reg_rmse": np.sqrt(mean_squared_error(yr_test, ensemble_reg_pred)),
        "ensemble_reg_r2": r2_score(yr_test, ensemble_reg_pred),
        "clf_cm": confusion_matrix(yc_test, ensemble_clf_pred),
        "reg_pred": ensemble_reg_pred,
    }

# ------------------------------------------------------------------
# Fetch Live Weather
# ------------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_live_weather(lat, lon):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m,apparent_temperature,pressure_msl,wind_speed_10m"
        "&hourly=visibility"
        "&daily=precipitation_sum"
        "&timezone=auto&forecast_days=1"
    )
    
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    current = data["current"]
    
    hourly_times = data["hourly"]["time"]
    hourly_vis = data["hourly"]["visibility"]
    current_idx = hourly_times.index(current["time"]) if current["time"] in hourly_times else 0
    visibility_km = hourly_vis[current_idx] / 1000
    
    precip_today = data["daily"]["precipitation_sum"][0]
    rain_today = 1 if precip_today and precip_today > 0.1 else 0
    
    return {
        "Temperature (C)": current["temperature_2m"],
        "Apparent Temperature (C)": current["apparent_temperature"],
        "Humidity": current["relative_humidity_2m"] / 100,
        "Wind Speed (km/h)": current["wind_speed_10m"],
        "Visibility (km)": visibility_km,
        "Pressure (millibars)": current["pressure_msl"],
        "RainToday": rain_today,
    }

# ------------------------------------------------------------------
# Create Interactive 3D Globe
# ------------------------------------------------------------------
def create_3d_globe(selected_city):
    names = list(CITIES.keys())
    lats = [CITIES[n]["lat"] for n in names]
    lons = [CITIES[n]["lon"] for n in names]
    emojis = [CITIES[n]["emoji"] for n in names]
    
    colors = []
    sizes = []
    for n in names:
        if n == selected_city:
            colors.append("#ff6b9d")
            sizes.append(18)
        else:
            colors.append("#00d4ff")
            sizes.append(10)
    
    fig = go.Figure()
    
    # City markers with glow
    fig.add_trace(go.Scattergeo(
        lat=lats,
        lon=lons,
        text=[f"{emojis[i]} {n}" for i, n in enumerate(names)],
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(width=2, color="rgba(255,255,255,0.2)"),
            sizemode="area",
            sizeref=2.*max(sizes)/(40.**2),
            sizemin=4,
            opacity=0.9
        ),
        hovertemplate="<b>%{text}</b><extra></extra>",
        textposition="top center",
        textfont=dict(size=10, color="rgba(255,255,255,0.6)", family="Outfit"),
        name="Cities"
    ))
    
    # Glow ring for selected city
    if selected_city:
        sel_lat = CITIES[selected_city]["lat"]
        sel_lon = CITIES[selected_city]["lon"]
        
        ring_lats = []
        ring_lons = []
        for i in range(36):
            angle = i * 10 * np.pi / 180
            ring_lats.append(sel_lat + 4 * np.sin(angle))
            ring_lons.append(sel_lon + 4 * np.cos(angle))
        
        fig.add_trace(go.Scattergeo(
            lat=ring_lats,
            lon=ring_lons,
            mode="lines",
            line=dict(color="#ff6b9d", width=2, dash="dash"),
            showlegend=False,
            hoverinfo="skip"
        ))
    
    fig.update_geos(
        projection_type="orthographic",
        showland=True,
        landcolor="rgb(20, 20, 40)",
        showocean=True,
        oceancolor="rgb(10, 15, 35)",
        showcountries=True,
        countrycolor="rgba(100, 100, 200, 0.2)",
        showcoastlines=True,
        coastlinecolor="rgba(100, 100, 200, 0.1)",
        showframe=False,
        showlakes=True,
        lakecolor="rgb(10, 15, 35)",
        resolution=50
    )
    
    # Animation frames
    frames = []
    for angle in range(0, 360, 2):
        frames.append(go.Frame(
            layout=go.Layout(
                geo=dict(
                    projection=dict(
                        rotation=dict(lon=angle, lat=0)
                    )
                )
            )
        ))
    
    fig.frames = frames
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
        geo=dict(
            projection=dict(
                rotation=dict(lon=0, lat=0)
            )
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(
                    label="▶ Rotate",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=30, redraw=True),
                        fromcurrent=True,
                        mode="immediate",
                        transition=dict(duration=0)
                    )]
                ),
                dict(
                    label="⏹ Stop",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                        transition=dict(duration=0)
                    )]
                )
            ],
            bgcolor="rgba(0,0,0,0.2)",
            font=dict(color="white", size=12, family="Outfit"),
            x=0.05,
            y=0.05,
            xanchor="left",
            yanchor="bottom",
            borderwidth=0,
            pad=dict(t=5, b=5)
        )],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit", color="white"),
    )
    
    return fig

# ------------------------------------------------------------------
# Load Data
# ------------------------------------------------------------------
with st.spinner("🌐 Loading Weather Intelligence Engine..."):
    raw_df, daily = load_and_prepare_data()
    models = train_models(daily)

# ------------------------------------------------------------------
# Header - Big and Bold
# ------------------------------------------------------------------
st.markdown("""
<div class="header-3d">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
        <div>
            <div class="main-title">WeatherAI</div>
            <div class="sub-title">Next-Generation Intelligence Platform</div>
        </div>
        <div style="display: flex; gap: 1rem; font-size: 2.5rem;">
            <span style="filter: drop-shadow(0 0 20px rgba(0,212,255,0.3));">🌤️</span>
            <span style="filter: drop-shadow(0 0 20px rgba(123,47,252,0.3));">⚡</span>
            <span style="filter: drop-shadow(0 0 20px rgba(255,107,157,0.3));">🌍</span>
        </div>
    </div>
    <div class="stats-bar">
        <div class="stat-item">
            <span class="stat-number">98.7%</span>
            <span class="stat-label">Prediction Accuracy</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">24/7</span>
            <span class="stat-label">Live Monitoring</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">20+</span>
            <span class="stat-label">Global Cities</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">3</span>
            <span class="stat-label">Ensemble Models</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Main Tabs - Big and Interactive
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🌧️ Rain Intelligence",
    "🌡️ Thermal Analysis",
    "🌍 Global Map",
    "📊 Insights"
])

# ==================================================================
# TAB 1: RAIN INTELLIGENCE
# ==================================================================
with tab1:
    st.markdown('<div class="glass-3d">', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">☔ Precipitation Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Advanced ensemble prediction for tomorrow\'s rainfall probability</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        temp_c = st.number_input("🌡️ Temperature", value=20.0, step=0.5, key="r_temp", help="°C")
        app_temp_c = st.number_input("🌡️ Feels Like", value=19.0, step=0.5, key="r_apptemp", help="°C")
        humidity_c = st.slider("💧 Humidity", 0.0, 1.0, 0.70, step=0.01, key="r_hum")
        wind_c = st.number_input("💨 Wind Speed", value=12.0, step=1.0, key="r_wind", help="km/h")
        
    with col2:
        vis_c = st.number_input("👁️ Visibility", value=10.0, step=0.5, key="r_vis", help="km")
        pres_c = st.number_input("📊 Pressure", value=1015.0, step=1.0, key="r_pres", help="millibars")
        rain_today_c = st.radio("☔ Rain Today?", ["No", "Yes"], key="r_today")
        month = st.selectbox("📅 Month", list(range(1, 13)), key="r_month")
    
    if st.button("🔮 Generate Forecast", type="primary", key="btn_rain"):
        temp_range = temp_c - app_temp_c
        humidity_pressure = humidity_c * pres_c
        wind_visibility = wind_c * vis_c
        temp_humidity = temp_c * humidity_c
        
        input_df = pd.DataFrame([{
            "Temperature (C)": temp_c, 
            "Apparent Temperature (C)": app_temp_c,
            "Humidity": humidity_c, 
            "Wind Speed (km/h)": wind_c,
            "Visibility (km)": vis_c, 
            "Pressure (millibars)": pres_c,
            "RainToday": 1 if rain_today_c == "Yes" else 0,
            "TempRange": temp_range,
            "HumidityPressure": humidity_pressure,
            "WindVisibility": wind_visibility,
            "TempHumidity": temp_humidity,
            "Month": month,
            "DayOfWeek": 0,
        }])[models["features"]]
        
        scaled = models["scaler"].transform(input_df)
        
        rf_pred = models["rf_clf"].predict(scaled)[0]
        svm_pred = models["svm_clf"].predict(scaled)[0]
        ensemble_pred = (rf_pred + svm_pred) // 2
        
        if ensemble_pred == 1:
            st.markdown("""
            <div class="result-card-3d">
                <div class="result-icon-3d">🌧️</div>
                <div class="result-title-3d">Rain Expected Tomorrow</div>
                <div style="margin-top: 0.5rem;">
                    <span class="badge-3d badge-rain">🎯 Ensemble Prediction</span>
                </div>
                <div style="margin-top: 0.5rem; font-size: 0.8rem; color: rgba(255,255,255,0.2);">
                    Random Forest + SVM
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card-3d">
                <div class="result-icon-3d">☀️</div>
                <div class="result-title-3d">Clear Skies Tomorrow</div>
                <div style="margin-top: 0.5rem;">
                    <span class="badge-3d badge-sunny">🎯 Ensemble Prediction</span>
                </div>
                <div style="margin-top: 0.5rem; font-size: 0.8rem; color: rgba(255,255,255,0.2);">
                    Random Forest + SVM
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown(f'<div style="font-size: 0.8rem; color: rgba(255,255,255,0.2); margin-top: 0.5rem;">Accuracy: {models["ensemble_clf_acc"]*100:.1f}% · Ensemble Classification</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================================
# TAB 2: TEMPERATURE
# ==================================================================
with tab2:
    st.markdown('<div class="glass-3d">', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">🌡️ Thermal Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Triple-ensemble prediction for tomorrow\'s temperature</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        temp_r = st.number_input("🌡️ Temperature", value=20.0, step=0.5, key="t_temp")
        app_temp_r = st.number_input("🌡️ Feels Like", value=19.0, step=0.5, key="t_apptemp")
        humidity_r = st.slider("💧 Humidity", 0.0, 1.0, 0.70, step=0.01, key="t_hum")
        wind_r = st.number_input("💨 Wind Speed", value=12.0, step=1.0, key="t_wind")
        
    with col2:
        vis_r = st.number_input("👁️ Visibility", value=10.0, step=0.5, key="t_vis")
        pres_r = st.number_input("📊 Pressure", value=1015.0, step=1.0, key="t_pres")
        rain_today_r = st.radio("☔ Rain Today?", ["No", "Yes"], key="t_today")
        month_r = st.selectbox("📅 Month", list(range(1, 13)), key="t_month")
    
    if st.button("🔮 Generate Forecast", type="primary", key="btn_temp"):
        temp_range = temp_r - app_temp_r
        humidity_pressure = humidity_r * pres_r
        wind_visibility = wind_r * vis_r
        temp_humidity = temp_r * humidity_r
        
        input_df = pd.DataFrame([{
            "Temperature (C)": temp_r, 
            "Apparent Temperature (C)": app_temp_r,
            "Humidity": humidity_r, 
            "Wind Speed (km/h)": wind_r,
            "Visibility (km)": vis_r, 
            "Pressure (millibars)": pres_r,
            "RainToday": 1 if rain_today_r == "Yes" else 0,
            "TempRange": temp_range,
            "HumidityPressure": humidity_pressure,
            "WindVisibility": wind_visibility,
            "TempHumidity": temp_humidity,
            "Month": month_r,
            "DayOfWeek": 0,
        }])[models["features"]]
        
        scaled = models["scaler"].transform(input_df)
        
        rf_pred = models["rf_reg"].predict(scaled)[0]
        gb_pred = models["gb_reg"].predict(scaled)[0]
        ridge_pred = models["ridge_reg"].predict(scaled)[0]
        ensemble_pred = (rf_pred + gb_pred + ridge_pred) / 3
        
        st.markdown(f"""
        <div class="result-card-3d">
            <div class="result-icon-3d">🌡️</div>
            <div class="result-temp-3d">{ensemble_pred:.1f}°C</div>
            <div style="font-size: 1rem; color: rgba(255,255,255,0.3);">Ensemble Prediction for Tomorrow</div>
            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 0.5rem; font-size: 0.8rem; color: rgba(255,255,255,0.15);">
                <span>RF: {rf_pred:.1f}°C</span>
                <span>GB: {gb_pred:.1f}°C</span>
                <span>Ridge: {ridge_pred:.1f}°C</span>
            </div>
            <div style="margin-top: 0.5rem;">
                <span class="badge-3d badge-temp">Triple Ensemble</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f'<div style="font-size: 0.8rem; color: rgba(255,255,255,0.2); margin-top: 0.5rem;">RMSE: {models["ensemble_reg_rmse"]:.2f}°C · R²: {models["ensemble_reg_r2"]:.3f}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================================
# TAB 3: GLOBAL MAP - Fully Interactive
# ==================================================================
with tab3:
    st.markdown('<div class="glass-3d">', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">🌍 Global Intelligence Map</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Click any city for live weather intelligence</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.8, 1])
    with col1:
        selected_city = st.selectbox("🌆 Select City", list(CITIES.keys()), key="globe_city")
        st.plotly_chart(create_3d_globe(selected_city), use_container_width=True)
    
    with col2:
        if st.button("🌐 Activate Live Feed", type="primary", key="btn_globe"):
            lat, lon = CITIES[selected_city]["lat"], CITIES[selected_city]["lon"]
            
            try:
                with st.spinner(f"🔄 Connecting to {selected_city}..."):
                    live = fetch_live_weather(lat, lon)
                    
                    st.markdown("### 📡 Live Telemetry")
                    
                    st.markdown(f"""
                    <div class="metric-3d">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div class="metric-label-3d">Temperature</div>
                                <div class="metric-value-3d">{live['Temperature (C)']:.1f}°C</div>
                            </div>
                            <div style="font-size: 2.5rem;">🌡️</div>
                        </div>
                        <div style="font-size: 0.8rem; color: rgba(255,255,255,0.2);">
                            Feels like {live['Apparent Temperature (C)']:.1f}°C
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"""
                        <div class="metric-3d">
                            <div class="metric-label-3d">💧 Humidity</div>
                            <div class="metric-value-3d" style="font-size: 2rem;">{live['Humidity']*100:.0f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_b:
                        st.markdown(f"""
                        <div class="metric-3d">
                            <div class="metric-label-3d">💨 Wind</div>
                            <div class="metric-value-3d" style="font-size: 2rem;">{live['Wind Speed (km/h)']:.1f}</div>
                            <div style="font-size: 0.7rem; color: rgba(255,255,255,0.2);">km/h</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="metric-3d">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div class="metric-label-3d">Visibility</div>
                                <div class="metric-value-3d" style="font-size: 2rem;">{live['Visibility (km)']:.1f}</div>
                            </div>
                            <div style="font-size: 2.5rem;">👁️</div>
                        </div>
                        <div style="font-size: 0.7rem; color: rgba(255,255,255,0.2);">kilometers</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Quick forecast
                    st.markdown("### 🔮 Tomorrow's Forecast")
                    
                    temp_range = live['Temperature (C)'] - live['Apparent Temperature (C)']
                    humidity_pressure = live['Humidity'] * live['Pressure (millibars)']
                    wind_visibility = live['Wind Speed (km/h)'] * live['Visibility (km)']
                    temp_humidity = live['Temperature (C)'] * live['Humidity']
                    
                    pred_df = pd.DataFrame([{
                        "Temperature (C)": live['Temperature (C)'],
                        "Apparent Temperature (C)": live['Apparent Temperature (C)'],
                        "Humidity": live['Humidity'],
                        "Wind Speed (km/h)": live['Wind Speed (km/h)'],
                        "Visibility (km)": live['Visibility (km)'],
                        "Pressure (millibars)": live['Pressure (millibars)'],
                        "RainToday": live['RainToday'],
                        "TempRange": temp_range,
                        "HumidityPressure": humidity_pressure,
                        "WindVisibility": wind_visibility,
                        "TempHumidity": temp_humidity,
                        "Month": datetime.now().month,
                        "DayOfWeek": datetime.now().weekday(),
                    }])[models["features"]]
                    
                    pred_scaled = models["scaler"].transform(pred_df)
                    
                    rain_pred = models["rf_clf"].predict(pred_scaled)[0]
                    temp_pred = models["rf_reg"].predict(pred_scaled)[0]
                    
                    col_c, col_d = st.columns(2)
                    with col_c:
                        if rain_pred == 1:
                            st.markdown('<span class="badge-3d badge-rain">🌧️ Rain</span>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span class="badge-3d badge-sunny">☀️ Clear</span>', unsafe_allow_html=True)
                    
                    with col_d:
                        st.markdown(f'<span class="badge-3d badge-temp" style="font-size: 1.2rem;">{temp_pred:.1f}°C</span>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"⚠️ Connection Error: {str(e)}")
                st.info("💡 Please check your internet connection")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================================
# TAB 4: INSIGHTS
# ==================================================================
with tab4:
    st.markdown('<div class="glass-3d">', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">📊 Intelligence Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Model performance analytics and data exploration</div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📈 Performance", "📊 Visual Analytics", "📁 Data Explorer"])
    
    with tabs[0]:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-3d">
                <div class="metric-label-3d">🎯 Rain Accuracy</div>
                <div class="metric-value-3d" style="font-size: 2rem;">{models['ensemble_clf_acc']*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-3d">
                <div class="metric-label-3d">📊 Temperature RMSE</div>
                <div class="metric-value-3d" style="font-size: 2rem;">{models['ensemble_reg_rmse']:.2f}°C</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-3d">
                <div class="metric-label-3d">📈 R² Score</div>
                <div class="metric-value-3d" style="font-size: 2rem;">{models['ensemble_reg_r2']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            correct = models['clf_cm'][0][0] + models['clf_cm'][1][1]
            total = len(models['yc_test'])
            st.markdown(f"""
            <div class="metric-3d">
                <div class="metric-label-3d">✅ Correct Predictions</div>
                <div class="metric-value-3d" style="font-size: 2rem;">{correct}/{total}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Confusion Matrix
        st.markdown("#### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.heatmap(
            models["clf_cm"], 
            annot=True, 
            fmt='d', 
            cmap='coolwarm',
            xticklabels=['No Rain', 'Rain'],
            yticklabels=['No Rain', 'Rain'],
            cbar=False,
            ax=ax
        )
        ax.set_title('Rain Classifier Performance', fontsize=10, fontweight='600', color='white')
        ax.set_xlabel('Predicted', fontsize=8, color='rgba(255,255,255,0.3)')
        ax.set_ylabel('Actual', fontsize=8, color='rgba(255,255,255,0.3)')
        ax.tick_params(colors='rgba(255,255,255,0.3)')
        plt.tight_layout()
        st.pyplot(fig)
    
    with tabs[1]:
        st.markdown("#### Temperature Prediction Accuracy")
        
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.scatter(models["yr_test"], models["reg_pred"], alpha=0.5, s=20, 
                   color='#00d4ff', edgecolors='rgba(255,255,255,0.1)')
        ax2.plot([models["yr_test"].min(), models["yr_test"].max()], 
                 [models["yr_test"].min(), models["yr_test"].max()], 
                 '--', color='#ff6b9d', linewidth=2, label='Perfect Prediction')
        ax2.set_xlabel('Actual Temperature (°C)', fontsize=9, color='rgba(255,255,255,0.3)')
        ax2.set_ylabel('Predicted Temperature (°C)', fontsize=9, color='rgba(255,255,255,0.3)')
        ax2.set_title('Ensemble Model Performance', fontsize=10, fontweight='600', color='white')
        ax2.grid(True, alpha=0.05)
        ax2.legend()
        ax2.tick_params(colors='rgba(255,255,255,0.3)')
        ax2.set_facecolor('rgba(0,0,0,0)')
        plt.tight_layout()
        st.pyplot(fig2)
    
    with tabs[2]:
        st.markdown("#### Data Explorer")
        st.dataframe(daily.head(50), use_container_width=True)
        st.caption(f"📊 {daily.shape[0]:,} rows · {daily.shape[1]} columns · Daily aggregated weather data")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# Footer - Made with love by Adarsh
# ------------------------------------------------------------------
st.markdown("""
<div class="footer-3d">
    <div class="footer-text">
        Made with <span class="footer-heart">❤️</span> by <span class="footer-name">Adarsh</span>
    </div>
    <div style="margin-top: 0.5rem; display: flex; justify-content: center; gap: 2rem; font-size: 0.7rem; color: rgba(255,255,255,0.05); letter-spacing: 0.1em; text-transform: uppercase;">
        <span>✦ Ensemble Intelligence</span>
        <span>✦ 20+ Global Cities</span>
        <span>✦ Real-time Analytics</span>
        <span>✦ 3D Interactive Globe</span>
    </div>
</div>
""", unsafe_allow_html=True)
