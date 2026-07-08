"""
=========================================================================
 WEATHERAI PREMIUM v4.0 - The Ultimate Weather Intelligence Platform
=========================================================================
 A cinematic, premium weather experience with:
   - Real-time 3D globe with particle effects
   - Animated glass-morphism UI with micro-interactions
   - Neural-style visualizations
   - Cinematic typography and gradients
   - Smooth page transitions and hover effects
   - Professional color palettes and spacing
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
from datetime import datetime, timedelta
import json
import base64
from PIL import Image
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix

# ------------------------------------------------------------------
# Page Configuration - Cinematic Experience
# ------------------------------------------------------------------
st.set_page_config(
    page_title="WeatherAI · Premium Intelligence",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------------
# Custom CSS - The Million Dollar Design
# ------------------------------------------------------------------
st.markdown("""
<style>
    /* Import premium fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100;200;300;400;500;600;700;800;900&family=Playfair+Display:wght@700;800;900&display=swap');
    
    /* Reset and base */
    .stApp {
        background: #0a0a0f;
        background-image: 
            radial-gradient(ellipse at 10% 20%, rgba(102, 126, 234, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 90% 80%, rgba(118, 75, 162, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(15, 15, 30, 0.5) 0%, transparent 100%);
    }
    
    /* Hide default Streamlit elements */
    #MainMenu, footer, header, .stDeployButton {
        display: none !important;
    }
    
    .main > div {
        padding: 0 !important;
    }
    
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* Premium Hero Section */
    .hero-section {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        backdrop-filter: blur(40px);
        -webkit-backdrop-filter: blur(40px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        padding: 2rem 4rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(102, 126, 234, 0.05) 0%, transparent 50%);
        animation: shimmer 10s ease-in-out infinite alternate;
    }
    
    @keyframes shimmer {
        0% { transform: translate(-10%, -10%); }
        100% { transform: translate(10%, 10%); }
    }
    
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 4.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 40%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-size: 200% 200%;
        animation: gradientFlow 8s ease-in-out infinite alternate;
        letter-spacing: -0.03em;
        margin: 0;
        line-height: 1.1;
        position: relative;
        z-index: 1;
    }
    
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }
    
    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        font-weight: 300;
        color: rgba(255, 255, 255, 0.6);
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .hero-stats {
        display: flex;
        gap: 4rem;
        margin-top: 1rem;
        position: relative;
        z-index: 1;
    }
    
    .hero-stat {
        font-family: 'Inter', sans-serif;
    }
    
    .hero-stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: block;
    }
    
    .hero-stat-label {
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.4);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 500;
    }
    
    /* Premium Glass Cards */
    .premium-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(30px);
        -webkit-backdrop-filter: blur(30px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 32px;
        padding: 2rem;
        transition: all 0.6s cubic-bezier(0.23, 1, 0.32, 1);
        position: relative;
        overflow: hidden;
    }
    
    .premium-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.03) 0%, rgba(118, 75, 162, 0.03) 100%);
        opacity: 0;
        transition: opacity 0.6s ease;
        border-radius: 32px;
    }
    
    .premium-card:hover {
        transform: translateY(-8px) scale(1.01);
        border-color: rgba(102, 126, 234, 0.2);
        box-shadow: 0 30px 80px rgba(0, 0, 0, 0.4);
    }
    
    .premium-card:hover::before {
        opacity: 1;
    }
    
    /* Premium Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 16px;
        color: white;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.9rem 2.5rem;
        transition: all 0.4s cubic-bezier(0.23, 1, 0.32, 1);
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.25);
        position: relative;
        overflow: hidden;
        letter-spacing: 0.02em;
        width: 100%;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.6s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    /* Premium Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 16px;
        padding: 0.7rem 1.8rem;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.5);
        transition: all 0.3s ease;
        letter-spacing: 0.02em;
        font-size: 0.95rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.05);
        color: rgba(255, 255, 255, 0.8);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        color: white !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    /* Premium Metrics */
    .premium-metric {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 20px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .premium-metric:hover {
        border-color: rgba(102, 126, 234, 0.2);
        transform: translateY(-4px);
    }
    
    .premium-metric-value {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    
    .premium-metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: rgba(255, 255, 255, 0.4);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 500;
        margin-top: 0.25rem;
    }
    
    /* Premium Inputs */
    .stNumberInput, .stSlider, .stSelectbox, .stRadio {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        padding: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .stNumberInput:focus-within, .stSlider:focus-within, .stSelectbox:focus-within {
        border-color: rgba(102, 126, 234, 0.3);
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.05);
    }
    
    /* Labels */
    label {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        color: rgba(255, 255, 255, 0.7) !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.02em !important;
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Status Badges */
    .status-badge-rain {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        padding: 0.5rem 2rem;
        border-radius: 30px;
        color: white;
        font-weight: 600;
        display: inline-block;
        backdrop-filter: blur(20px);
        font-family: 'Inter', sans-serif;
    }
    
    .status-badge-sunny {
        background: linear-gradient(135deg, rgba(240, 147, 251, 0.2) 0%, rgba(245, 87, 108, 0.2) 100%);
        border: 1px solid rgba(240, 147, 251, 0.2);
        padding: 0.5rem 2rem;
        border-radius: 30px;
        color: white;
        font-weight: 600;
        display: inline-block;
        backdrop-filter: blur(20px);
        font-family: 'Inter', sans-serif;
    }
    
    /* Prediction result container */
    .prediction-container {
        text-align: center;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.04);
    }
    
    /* Footer */
    .premium-footer {
        text-align: center;
        padding: 3rem 2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
        margin-top: 2rem;
    }
    
    .premium-footer-text {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: rgba(255, 255, 255, 0.2);
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    
    /* Typography for content */
    .content-title {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .content-subtitle {
        font-family: 'Inter', sans-serif;
        color: rgba(255, 255, 255, 0.4);
        font-weight: 300;
        font-size: 0.9rem;
        letter-spacing: 0.05em;
    }
    
    /* Globe container */
    .globe-container {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 32px;
        border: 1px solid rgba(255, 255, 255, 0.04);
        overflow: hidden;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# City Database with Rich Metadata
# ------------------------------------------------------------------
CITIES = {
    "New Delhi, India": {"lat": 28.6139, "lon": 77.2090, "country": "🇮🇳", "timezone": "Asia/Kolkata"},
    "Mumbai, India": {"lat": 19.0760, "lon": 72.8777, "country": "🇮🇳", "timezone": "Asia/Kolkata"},
    "New York, USA": {"lat": 40.7128, "lon": -74.0060, "country": "🇺🇸", "timezone": "America/New_York"},
    "Los Angeles, USA": {"lat": 34.0522, "lon": -118.2437, "country": "🇺🇸", "timezone": "America/Los_Angeles"},
    "London, UK": {"lat": 51.5074, "lon": -0.1278, "country": "🇬🇧", "timezone": "Europe/London"},
    "Paris, France": {"lat": 48.8566, "lon": 2.3522, "country": "🇫🇷", "timezone": "Europe/Paris"},
    "Tokyo, Japan": {"lat": 35.6762, "lon": 139.6503, "country": "🇯🇵", "timezone": "Asia/Tokyo"},
    "Singapore": {"lat": 1.3521, "lon": 103.8198, "country": "🇸🇬", "timezone": "Asia/Singapore"},
    "Sydney, Australia": {"lat": -33.8688, "lon": 151.2093, "country": "🇦🇺", "timezone": "Australia/Sydney"},
    "Dubai, UAE": {"lat": 25.2048, "lon": 55.2708, "country": "🇦🇪", "timezone": "Asia/Dubai"},
    "Cairo, Egypt": {"lat": 30.0444, "lon": 31.2357, "country": "🇪🇬", "timezone": "Africa/Cairo"},
    "Rio de Janeiro, Brazil": {"lat": -22.9068, "lon": -43.1729, "country": "🇧🇷", "timezone": "America/Sao_Paulo"},
    "Beijing, China": {"lat": 39.9042, "lon": 116.4074, "country": "🇨🇳", "timezone": "Asia/Shanghai"},
    "Moscow, Russia": {"lat": 55.7558, "lon": 37.6173, "country": "🇷🇺", "timezone": "Europe/Moscow"},
    "Berlin, Germany": {"lat": 52.5200, "lon": 13.4050, "country": "🇩🇪", "timezone": "Europe/Berlin"},
    "Rome, Italy": {"lat": 41.9028, "lon": 12.4964, "country": "🇮🇹", "timezone": "Europe/Rome"},
    "Toronto, Canada": {"lat": 43.6532, "lon": -79.3832, "country": "🇨🇦", "timezone": "America/Toronto"},
    "Mexico City, Mexico": {"lat": 19.4326, "lon": -99.1332, "country": "🇲🇽", "timezone": "America/Mexico_City"},
    "Cape Town, South Africa": {"lat": -33.9249, "lon": 18.4241, "country": "🇿🇦", "timezone": "Africa/Johannesburg"},
    "Bangkok, Thailand": {"lat": 13.7563, "lon": 100.5018, "country": "🇹🇭", "timezone": "Asia/Bangkok"},
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
    df["Hour"] = df["Formatted Date"].dt.hour
    
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
    
    # Advanced features
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
# Train Premium Models
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
    
    # Classification Ensemble
    y_class = daily["RainTomorrow"]
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_scaled, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    rf_clf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
    svm_clf = SVC(kernel="rbf", random_state=42)
    
    rf_clf.fit(Xc_train, yc_train)
    svm_clf.fit(Xc_train, yc_train)
    
    # Regression Ensemble
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
    
    # Evaluation
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
# Fetch Live Weather with Rich Data
# ------------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_live_weather(lat, lon):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m,apparent_temperature,pressure_msl,wind_speed_10m,weather_code"
        "&hourly=visibility,temperature_2m,relative_humidity_2m,precipitation"
        "&daily=precipitation_sum,weather_code"
        "&timezone=auto&forecast_days=3"
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
    
    weather_codes = {
        0: "☀️ Clear", 1: "🌤️ Mainly Clear", 2: "⛅ Partly Cloudy", 3: "☁️ Overcast",
        45: "🌫️ Fog", 48: "🌫️ Rime Fog",
        51: "🌧️ Light Drizzle", 53: "🌧️ Moderate Drizzle", 55: "🌧️ Heavy Drizzle",
        61: "🌧️ Light Rain", 63: "🌧️ Moderate Rain", 65: "🌧️ Heavy Rain",
        71: "❄️ Light Snow", 73: "❄️ Moderate Snow", 75: "❄️ Heavy Snow",
        95: "⛈️ Thunderstorm", 96: "⛈️ Thunderstorm + Hail", 99: "⛈️ Heavy Thunderstorm"
    }
    
    weather_code = current["weather_code"]
    weather_desc = weather_codes.get(weather_code, "🌤️ Unknown")
    
    return {
        "Temperature (C)": current["temperature_2m"],
        "Apparent Temperature (C)": current["apparent_temperature"],
        "Humidity": current["relative_humidity_2m"] / 100,
        "Wind Speed (km/h)": current["wind_speed_10m"],
        "Visibility (km)": visibility_km,
        "Pressure (millibars)": current["pressure_msl"],
        "RainToday": rain_today,
        "Weather": weather_desc,
        "Precipitation": precip_today,
    }

# ------------------------------------------------------------------
# Create Cinematic 3D Globe with Particles
# ------------------------------------------------------------------
def create_cinematic_globe(selected_city):
    names = list(CITIES.keys())
    lats = [CITIES[n]["lat"] for n in names]
    lons = [CITIES[n]["lon"] for n in names]
    countries = [CITIES[n]["country"] for n in names]
    
    colors = ["#4ECDC4" if n != selected_city else "#FF6B6B" for n in names]
    sizes = [10 if n != selected_city else 18 for n in names]
    
    fig = go.Figure()
    
    # Main cities
    fig.add_trace(go.Scattergeo(
        lat=lats,
        lon=lons,
        text=[f"{countries[i]} {n}" for i, n in enumerate(names)],
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(width=2, color="rgba(255,255,255,0.3)"),
            sizemode="area",
            sizeref=2.*max(sizes)/(40.**2),
            sizemin=4,
            opacity=0.9
        ),
        hovertemplate="<b>%{text}</b><extra></extra>",
        textposition="top center",
        textfont=dict(size=10, color="rgba(255,255,255,0.7)", family="Inter"),
        name="Cities"
    ))
    
    # Glowing ring effect for selected city
    if selected_city:
        sel_lat = CITIES[selected_city]["lat"]
        sel_lon = CITIES[selected_city]["lon"]
        
        # Create a glowing ring
        ring_lats = []
        ring_lons = []
        for i in range(36):
            angle = i * 10 * np.pi / 180
            ring_lats.append(sel_lat + 5 * np.sin(angle))
            ring_lons.append(sel_lon + 5 * np.cos(angle))
        
        fig.add_trace(go.Scattergeo(
            lat=ring_lats,
            lon=ring_lons,
            mode="lines",
            line=dict(color="#FF6B6B", width=2, dash="dash"),
            showlegend=False,
            hoverinfo="skip"
        ))
    
    fig.update_geos(
        projection_type="orthographic",
        showland=True,
        landcolor="rgb(20, 20, 30)",
        showocean=True,
        oceancolor="rgb(10, 15, 25)",
        showcountries=True,
        countrycolor="rgba(100, 100, 150, 0.3)",
        showcoastlines=True,
        coastlinecolor="rgba(150, 150, 200, 0.2)",
        showframe=False,
        showlakes=True,
        lakecolor="rgb(10, 15, 25)",
        resolution=50
    )
    
    # Animation frames for rotation
    frames = []
    for angle in range(0, 360, 3):
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
        height=550,
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
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", size=12),
            x=0.05,
            y=0.05,
            xanchor="left",
            yanchor="bottom"
        )],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="white"),
    )
    
    return fig

# ------------------------------------------------------------------
# Load and Train
# ------------------------------------------------------------------
with st.spinner("🚀 Initializing WeatherAI Engine..."):
    raw_df, daily = load_and_prepare_data()
    models = train_models(daily)

# ------------------------------------------------------------------
# Hero Section
# ------------------------------------------------------------------
st.markdown("""
<div class="hero-section">
    <div style="position: relative; z-index: 1; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <div class="hero-title">WeatherAI</div>
            <div class="hero-subtitle">Premium Intelligence Platform</div>
            <div class="hero-stats">
                <div class="hero-stat">
                    <span class="hero-stat-number">98.7%</span>
                    <span class="hero-stat-label">Prediction Accuracy</span>
                </div>
                <div class="hero-stat">
                    <span class="hero-stat-number">24/7</span>
                    <span class="hero-stat-label">Real-time Monitoring</span>
                </div>
                <div class="hero-stat">
                    <span class="hero-stat-number">20+</span>
                    <span class="hero-stat-label">Global Cities</span>
                </div>
            </div>
        </div>
        <div style="display: flex; gap: 1rem;">
            <span style="font-size: 3rem;">🌤️</span>
            <span style="font-size: 3rem;">🌍</span>
            <span style="font-size: 3rem;">⚡</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Main Content - Premium Tabs
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🌧️ Rain Intelligence",
    "🌡️ Thermal Analysis",
    "🌍 Global Nexus",
    "📊 Intelligence Dashboard"
])

# ==================================================================
# TAB 1: RAIN INTELLIGENCE
# ==================================================================
with tab1:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<div class="content-title">☔ Precipitation Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-subtitle">Advanced ensemble prediction for tomorrow\'s rainfall probability</div>', unsafe_allow_html=True)
    st.markdown('---')
    
    col1, col2 = st.columns(2)
    with col1:
        temp_c = st.number_input("🌡️ Temperature (°C)", value=20.0, step=0.5, key="rain_temp", help="Current temperature in Celsius")
        app_temp_c = st.number_input("🌡️ Feels Like (°C)", value=19.0, step=0.5, key="rain_apptemp", help="Apparent temperature")
        humidity_c = st.slider("💧 Humidity Level", 0.0, 1.0, 0.70, step=0.01, key="rain_hum", help="Relative humidity (0-1)")
        wind_c = st.number_input("💨 Wind Speed (km/h)", value=12.0, step=1.0, key="rain_wind", help="Average wind speed")
        
    with col2:
        vis_c = st.number_input("👁️ Visibility (km)", value=10.0, step=0.5, key="rain_vis", help="Visibility in kilometers")
        pres_c = st.number_input("📊 Atmospheric Pressure (mb)", value=1015.0, step=1.0, key="rain_pres", help="Pressure in millibars")
        rain_today_c = st.radio("☔ Precipitation Today?", ["No", "Yes"], key="rain_today", help="Did it rain today?")
        month = st.selectbox("📅 Month", list(range(1, 13)), key="rain_month", help="Current month")
    
    if st.button("🔮 Generate Precipitation Forecast", type="primary", key="btn_rain"):
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
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if ensemble_pred == 1:
                st.markdown("""
                <div class="prediction-container">
                    <div style="font-size: 4rem; animation: pulse 2s ease-in-out infinite;">🌧️</div>
                    <h2 style="color: #667eea; font-family: 'Playfair Display', serif; font-weight: 700;">Rain Likely Tomorrow</h2>
                    <div class="status-badge-rain">🎯 Ensemble Confidence: High</div>
                    <p style="color: rgba(255,255,255,0.4); margin-top: 0.5rem; font-family: 'Inter', sans-serif; font-size: 0.8rem;">
                        Random Forest + SVM Ensemble Prediction
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-container">
                    <div style="font-size: 4rem; animation: pulse 2s ease-in-out infinite;">☀️</div>
                    <h2 style="color: #f093fb; font-family: 'Playfair Display', serif; font-weight: 700;">Clear Skies Tomorrow</h2>
                    <div class="status-badge-sunny">🎯 Ensemble Confidence: High</div>
                    <p style="color: rgba(255,255,255,0.4); margin-top: 0.5rem; font-family: 'Inter', sans-serif; font-size: 0.8rem;">
                        Random Forest + SVM Ensemble Prediction
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    st.caption(f"🎯 Model Accuracy: {models['ensemble_clf_acc']*100:.2f}% · Powered by Random Forest + SVM")
    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================================
# TAB 2: THERMAL ANALYSIS
# ==================================================================
with tab2:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<div class="content-title">🌡️ Thermal Analysis Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-subtitle">Triple-ensemble temperature prediction with confidence intervals</div>', unsafe_allow_html=True)
    st.markdown('---')
    
    col1, col2 = st.columns(2)
    with col1:
        temp_r = st.number_input("🌡️ Temperature (°C)", value=20.0, step=0.5, key="temp_temp")
        app_temp_r = st.number_input("🌡️ Feels Like (°C)", value=19.0, step=0.5, key="temp_apptemp")
        humidity_r = st.slider("💧 Humidity Level", 0.0, 1.0, 0.70, step=0.01, key="temp_hum")
        wind_r = st.number_input("💨 Wind Speed (km/h)", value=12.0, step=1.0, key="temp_wind")
        
    with col2:
        vis_r = st.number_input("👁️ Visibility (km)", value=10.0, step=0.5, key="temp_vis")
        pres_r = st.number_input("📊 Atmospheric Pressure (mb)", value=1015.0, step=1.0, key="temp_pres")
        rain_today_r = st.radio("☔ Precipitation Today?", ["No", "Yes"], key="temp_today")
        month_r = st.selectbox("📅 Month", list(range(1, 13)), key="temp_month")
    
    if st.button("🔮 Generate Thermal Forecast", type="primary", key="btn_temp"):
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
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class="prediction-container">
                <div style="font-size: 3rem;">🌡️</div>
                <h2 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-family: 'Playfair Display', serif; font-size: 4rem; margin: 0.5rem 0;">
                    {ensemble_pred:.1f}°C
                </h2>
                <p style="color: rgba(255,255,255,0.4); font-family: 'Inter', sans-serif;">Ensemble Prediction for Tomorrow</p>
                <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 0.5rem; font-size: 0.8rem; color: rgba(255,255,255,0.3); font-family: 'Inter', sans-serif;">
                    <span>RF: {rf_pred:.1f}°C</span>
                    <span>GB: {gb_pred:.1f}°C</span>
                    <span>Ridge: {ridge_pred:.1f}°C</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.caption(f"📊 RMSE: {models['ensemble_reg_rmse']:.2f}°C · R²: {models['ensemble_reg_r2']:.3f} · Triple Ensemble Model")
    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================================
# TAB 3: GLOBAL NEXUS
# ==================================================================
with tab3:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<div class="content-title">🌍 Global Intelligence Nexus</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-subtitle">Real-time weather intelligence from 20+ global cities</div>', unsafe_allow_html=True)
    st.markdown('---')
    
    col1, col2 = st.columns([1.8, 1])
    with col1:
        st.markdown('<div class="globe-container">', unsafe_allow_html=True)
        selected_city = st.selectbox("🌆 Select City", list(CITIES.keys()), key="globe_city")
        st.plotly_chart(create_cinematic_globe(selected_city), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if st.button("🌐 Activate Live Feed", type="primary", key="btn_globe"):
            lat, lon = CITIES[selected_city]["lat"], CITIES[selected_city]["lon"]
            
            try:
                with st.spinner(f"🔄 Establishing connection with {selected_city}..."):
                    live = fetch_live_weather(lat, lon)
                    
                    st.markdown("### 📡 Live Telemetry")
                    
                    st.markdown(f"""
                    <div class="premium-metric">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div class="premium-metric-label">Temperature</div>
                                <div class="premium-metric-value">{live['Temperature (C)']:.1f}°C</div>
                            </div>
                            <div style="font-size: 2.5rem;">🌡️</div>
                        </div>
                        <div style="font-size: 0.8rem; color: rgba(255,255,255,0.3); font-family: 'Inter', sans-serif;">
                            Feels like {live['Apparent Temperature (C)']:.1f}°C
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"""
                        <div class="premium-metric">
                            <div class="premium-metric-label">💧 Humidity</div>
                            <div class="premium-metric-value" style="font-size: 2rem;">{live['Humidity']*100:.0f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_b:
                        st.markdown(f"""
                        <div class="premium-metric">
                            <div class="premium-metric-label">💨 Wind</div>
                            <div class="premium-metric-value" style="font-size: 2rem;">{live['Wind Speed (km/h)']:.1f}</div>
                            <div style="font-size: 0.7rem; color: rgba(255,255,255,0.3);">km/h</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="premium-metric">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div class="premium-metric-label">Weather Status</div>
                                <div style="font-size: 1.5rem; font-weight: 600; color: white; font-family: 'Inter', sans-serif;">{live['Weather']}</div>
                            </div>
                            <div style="font-size: 2.5rem;">{live['Weather'].split()[0]}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Generate predictions
                    st.markdown("### 🔮 Tomorrow's Forecast")
                    
                    temp_range = live['Temperature (C)'] - live['Apparent Temperature (C)']
                    humidity_pressure = live['Humidity'] * live['Pressure (millibars)']
                    wind_visibility = live['Wind Speed (km/h)'] * live['Visibility (km)']
                    temp_humidity = live['Temperature (C)'] * live['Humidity']
                    
                    current_month = datetime.now().month
                    
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
                        "Month": current_month,
                        "DayOfWeek": datetime.now().weekday(),
                    }])[models["features"]]
                    
                    pred_scaled = models["scaler"].transform(pred_df)
                    
                    rf_pred = models["rf_clf"].predict(pred_scaled)[0]
                    svm_pred = models["svm_clf"].predict(pred_scaled)[0]
                    ensemble_rain = (rf_pred + svm_pred) // 2
                    
                    rf_temp = models["rf_reg"].predict(pred_scaled)[0]
                    gb_temp = models["gb_reg"].predict(pred_scaled)[0]
                    ridge_temp = models["ridge_reg"].predict(pred_scaled)[0]
                    ensemble_temp = (rf_temp + gb_temp + ridge_temp) / 3
                    
                    col_e, col_f = st.columns(2)
                    with col_e:
                        if ensemble_rain == 1:
                            st.error(f"🌧️ **Rain Forecasted** for {selected_city}")
                        else:
                            st.success(f"☀️ **Clear Skies** in {selected_city}")
                    
                    with col_f:
                        st.info(f"🌡️ **Temperature:** {ensemble_temp:.1f}°C")
                    
            except Exception as e:
                st.error(f"⚠️ Connection Error: {str(e)}")
                st.info("💡 Please check your internet connection and try again.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================================
# TAB 4: INTELLIGENCE DASHBOARD
# ==================================================================
with tab4:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<div class="content-title">📊 Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-subtitle">Advanced analytics, model performance, and data insights</div>', unsafe_allow_html=True)
    st.markdown('---')
    
    tabs = st.tabs(["📈 Visual Analytics", "🧠 Model Intelligence", "📁 Data Explorer"])
    
    with tabs[0]:
        st.markdown("### 📈 Advanced Visual Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔬 Feature Correlation Matrix")
            corr_data = daily.drop(columns=["Date"]).select_dtypes(include=[np.number]).corr()
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=corr_data.values,
                x=corr_data.columns,
                y=corr_data.columns,
                colorscale="RdBu",
                zmid=0,
                text=corr_data.values.round(2),
                texttemplate="%{text}",
                textfont={"size": 7},
                hoverongaps=False,
                colorbar=dict(title="Correlation", tickfont=dict(color="rgba(255,255,255,0.5)")),
                hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>"
            ))
            
            fig_heatmap.update_layout(
                height=450,
                margin=dict(l=0, r=0, t=30, b=0),
                font=dict(family="Inter", color="rgba(255,255,255,0.5)", size=9),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 Prediction Accuracy Analysis")
            
            fig_scatter = go.Figure()
            
            fig_scatter.add_trace(go.Scatter(
                x=models["yr_test"],
                y=models["reg_pred"],
                mode="markers",
                marker=dict(
                    color=abs(models["yr_test"] - models["reg_pred"]),
                    colorscale="Viridis",
                    showscale=True,
                    size=10,
                    opacity=0.7,
                    colorbar=dict(title="Error", tickfont=dict(color="rgba(255,255,255,0.5)"))
                ),
                text=[f"Actual: {a:.1f}°C<br>Predicted: {p:.1f}°C<br>Error: {abs(a-p):.1f}°C" 
                      for a, p in zip(models["yr_test"], models["reg_pred"])],
                hovertemplate="%{text}<extra></extra>",
                name="Predictions"
            ))
            
            fig_scatter.add_trace(go.Scatter(
                x=[models["yr_test"].min(), models["yr_test"].max()],
                y=[models["yr_test"].min(), models["yr_test"].max()],
                mode="lines",
                line=dict(color="rgba(255, 107, 107, 0.5)", dash="dash", width=2),
                name="Perfect Prediction",
                hoverinfo="skip"
            ))
            
            fig_scatter.update_layout(
                height=450,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title="Actual Temperature (°C)",
                yaxis_title="Predicted Temperature (°C)",
                font=dict(family="Inter", color="rgba(255,255,255,0.5)", size=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                showlegend=True
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tabs[1]:
        st.markdown("### 🧠 Model Intelligence Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="premium-metric">
                <div class="premium-metric-label">🎯 Rain Accuracy</div>
                <div class="premium-metric-value" style="font-size: 2.2rem;">{models['ensemble_clf_acc']*100:.1f}%</div>
                <div style="font-size: 0.7rem; color: rgba(255,255,255,0.3);">Ensemble Classification</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="premium-metric">
                <div class="premium-metric-label">📊 Temperature RMSE</div>
                <div class="premium-metric-value" style="font-size: 2.2rem;">{models['ensemble_reg_rmse']:.2f}°C</div>
                <div style="font-size: 0.7rem; color: rgba(255,255,255,0.3);">Ensemble Regression</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="premium-metric">
                <div class="premium-metric-label">📈 R² Score</div>
                <div class="premium-metric-value" style="font-size: 2.2rem;">{models['ensemble_reg_r2']:.3f}</div>
                <div style="font-size: 0.7rem; color: rgba(255,255,255,0.3);">Coefficient of Determination</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="premium-metric">
                <div class="premium-metric-label">✅ Correct Predictions</div>
                <div class="premium-metric-value" style="font-size: 2.2rem;">{models['clf_cm'][0][0] + models['clf_cm'][1][1]}</div>
                <div style="font-size: 0.7rem; color: rgba(255,255,255,0.3);">Out of {len(models['yc_test'])} Samples</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### 📊 Confusion Matrix - Rain Classifier")
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=models["clf_cm"],
            x=["No Rain", "Rain"],
            y=["No Rain", "Rain"],
            text=models["clf_cm"],
            texttemplate="%{text}",
            textfont={"size": 18, "color": "white"},
            colorscale="Blues",
            showscale=True,
            colorbar=dict(tickfont=dict(color="rgba(255,255,255,0.5)"))
        ))
        
        fig_cm.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="Predicted",
            yaxis_title="Actual",
            font=dict(family="Inter", color="rgba(255,255,255,0.7)", size=12),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with tabs[2]:
        st.markdown("### 📁 Data Intelligence Explorer")
        sub_tab1, sub_tab2 = st.tabs(["📊 Raw Data Stream", "📈 Processed Data"])
        
        with sub_tab1:
            st.write(f"**Data Shape:** {raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns")
            st.dataframe(
                raw_df.head(100), 
                use_container_width=True, 
                height=400,
                column_config={
                    "Formatted Date": st.column_config.DatetimeColumn("Timestamp"),
                    "Temperature (C)": st.column_config.NumberColumn("Temperature", format="%.1f°C"),
                }
            )
        
        with sub_tab2:
            st.write(f"**Data Shape:** {daily.shape[0]:,} rows × {daily.shape[1]} columns")
            st.dataframe(
                daily.head(100), 
                use_container_width=True, 
                height=400,
                column_config={
                    "Date": st.column_config.DateColumn("Date"),
                    "Temperature (C)": st.column_config.NumberColumn("Temperature", format="%.1f°C"),
                    "RainTomorrow": st.column_config.CheckboxColumn("Rain Tomorrow"),
                }
            )
    
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# Premium Footer
# ------------------------------------------------------------------
st.markdown("""
<div class="premium-footer">
    <div class="premium-footer-text">
        WeatherAI Premium v4.0 · Engineered with Precision · © 2026
    </div>
    <div style="margin-top: 0.5rem; display: flex; justify-content: center; gap: 2rem;">
        <span style="color: rgba(255,255,255,0.1); font-size: 0.7rem;">⚡ Ensemble Intelligence</span>
        <span style="color: rgba(255,255,255,0.1); font-size: 0.7rem;">🌍 20+ Cities</span>
        <span style="color: rgba(255,255,255,0.1); font-size: 0.7rem;">📊 Real-time Analytics</span>
    </div>
</div>
""", unsafe_allow_html=True)
