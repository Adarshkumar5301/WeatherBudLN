"""
=========================================================================
 WEATHER PREDICTION WEB APP (Streamlit) - ULTRA PREMIUM v3.0
=========================================================================
 The most advanced weather prediction app with:
   - Stunning 3D globe with animated rotation
   - Modern glass-morphism UI design
   - Advanced visualizations with Plotly
   - Real-time weather fetching
   - Dual-model predictions with confidence scores
   - Animated transitions and hover effects
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
import time
from datetime import datetime, timedelta
import json

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------------
# Page configuration - Ultra Premium
# ------------------------------------------------------------------
st.set_page_config(
    page_title="WeatherAI - Premium Predictor", 
    page_icon="🌤️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------------
# Custom CSS - Glassmorphism + Modern Design
# ------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0 !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }
    
    /* Glass cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 24px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.15);
    }
    
    /* Premium buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Premium metrics */
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 16px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        color: #4b5563;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Input fields */
    .stNumberInput, .stSlider, .stSelectbox {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .stNumberInput:focus, .stSlider:focus, .stSelectbox:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header {
        display: none !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Status indicators */
    .status-rain {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.5rem 1.5rem;
        border-radius: 30px;
        color: white;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .status-sunny {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 0.5rem 1.5rem;
        border-radius: 30px;
        color: white;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# City database with more details
# ------------------------------------------------------------------
CITIES = {
    "New Delhi, India": {"lat": 28.6139, "lon": 77.2090, "country": "India", "timezone": "Asia/Kolkata"},
    "Mumbai, India": {"lat": 19.0760, "lon": 72.8777, "country": "India", "timezone": "Asia/Kolkata"},
    "New York, USA": {"lat": 40.7128, "lon": -74.0060, "country": "USA", "timezone": "America/New_York"},
    "Los Angeles, USA": {"lat": 34.0522, "lon": -118.2437, "country": "USA", "timezone": "America/Los_Angeles"},
    "London, UK": {"lat": 51.5074, "lon": -0.1278, "country": "UK", "timezone": "Europe/London"},
    "Paris, France": {"lat": 48.8566, "lon": 2.3522, "country": "France", "timezone": "Europe/Paris"},
    "Tokyo, Japan": {"lat": 35.6762, "lon": 139.6503, "country": "Japan", "timezone": "Asia/Tokyo"},
    "Singapore": {"lat": 1.3521, "lon": 103.8198, "country": "Singapore", "timezone": "Asia/Singapore"},
    "Sydney, Australia": {"lat": -33.8688, "lon": 151.2093, "country": "Australia", "timezone": "Australia/Sydney"},
    "Dubai, UAE": {"lat": 25.2048, "lon": 55.2708, "country": "UAE", "timezone": "Asia/Dubai"},
    "Cairo, Egypt": {"lat": 30.0444, "lon": 31.2357, "country": "Egypt", "timezone": "Africa/Cairo"},
    "Rio de Janeiro, Brazil": {"lat": -22.9068, "lon": -43.1729, "country": "Brazil", "timezone": "America/Sao_Paulo"},
    "Beijing, China": {"lat": 39.9042, "lon": 116.4074, "country": "China", "timezone": "Asia/Shanghai"},
    "Moscow, Russia": {"lat": 55.7558, "lon": 37.6173, "country": "Russia", "timezone": "Europe/Moscow"},
    "Berlin, Germany": {"lat": 52.5200, "lon": 13.4050, "country": "Germany", "timezone": "Europe/Berlin"},
    "Rome, Italy": {"lat": 41.9028, "lon": 12.4964, "country": "Italy", "timezone": "Europe/Rome"},
    "Toronto, Canada": {"lat": 43.6532, "lon": -79.3832, "country": "Canada", "timezone": "America/Toronto"},
    "Mexico City, Mexico": {"lat": 19.4326, "lon": -99.1332, "country": "Mexico", "timezone": "America/Mexico_City"},
    "Cape Town, South Africa": {"lat": -33.9249, "lon": 18.4241, "country": "South Africa", "timezone": "Africa/Johannesburg"},
    "Bangkok, Thailand": {"lat": 13.7563, "lon": 100.5018, "country": "Thailand", "timezone": "Asia/Bangkok"},
}

# ------------------------------------------------------------------
# Load and prepare data with advanced features
# ------------------------------------------------------------------
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("weatherHistory.csv")
    
    # Advanced preprocessing
    df = df.drop_duplicates()
    df = df.drop(columns=["Loud Cover"], errors='ignore')
    df["Precip Type"] = df["Precip Type"].fillna("none")
    
    df["Formatted Date"] = pd.to_datetime(df["Formatted Date"], utc=True)
    df["Date"] = df["Formatted Date"].dt.date
    df["Hour"] = df["Formatted Date"].dt.hour
    df["DayOfYear"] = df["Formatted Date"].dt.dayofyear
    df["Month"] = df["Formatted Date"].dt.month
    df["DayOfWeek"] = df["Formatted Date"].dt.dayofweek
    
    # Create cyclical features for time
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    
    df["is_rain_hour"] = (df["Precip Type"] == "rain").astype(int)
    df["is_snow_hour"] = (df["Precip Type"] == "snow").astype(int)
    
    # Daily aggregation
    daily = df.groupby("Date").agg({
        "Temperature (C)": "mean",
        "Apparent Temperature (C)": "mean",
        "Humidity": "mean",
        "Wind Speed (km/h)": "mean",
        "Visibility (km)": "mean",
        "Pressure (millibars)": "mean",
        "is_rain_hour": "max",
        "is_snow_hour": "max",
        "DayOfYear": "first",
        "Month": "first",
        "DayOfWeek": "first",
        "Hour_sin": "mean",
        "Hour_cos": "mean",
        "Month_sin": "mean",
        "Month_cos": "mean",
    }).reset_index()
    
    daily = daily.rename(columns={"is_rain_hour": "RainToday"})
    
    # Create more advanced features
    daily["TempRange"] = daily["Temperature (C)"] - daily["Apparent Temperature (C)"]
    daily["HumidityPressure"] = daily["Humidity"] * daily["Pressure (millibars)"]
    daily["WindVisibility"] = daily["Wind Speed (km/h)"] * daily["Visibility (km)"]
    
    # Target variables
    daily["RainTomorrow"] = daily["RainToday"].shift(-1)
    daily["TempTomorrow"] = daily["Temperature (C)"].shift(-1)
    daily["TempChange"] = daily["Temperature (C)"].shift(-1) - daily["Temperature (C)"]
    
    daily = daily.dropna().reset_index(drop=True)
    daily["RainTomorrow"] = daily["RainTomorrow"].astype(int)
    
    return df, daily

# ------------------------------------------------------------------
# Train multiple models with ensemble approach
# ------------------------------------------------------------------
@st.cache_resource
def train_models(daily):
    # Expanded feature set
    features = [
        "Temperature (C)", "Apparent Temperature (C)", "Humidity",
        "Wind Speed (km/h)", "Visibility (km)", "Pressure (millibars)", 
        "RainToday", "TempRange", "HumidityPressure", "WindVisibility",
        "Month", "DayOfWeek", "Month_sin", "Month_cos"
    ]
    
    X = daily[features]
    X_scaled = StandardScaler().fit_transform(X)
    
    # Classification - Ensemble
    y_class = daily["RainTomorrow"]
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_scaled, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # Multiple models for ensemble
    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    svm_clf = SVC(kernel="rbf", probability=True, random_state=42)
    
    rf_clf.fit(Xc_train, yc_train)
    svm_clf.fit(Xc_train, yc_train)
    
    # Regression - Ensemble
    y_reg = daily["TempTomorrow"]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_scaled, y_reg, test_size=0.2, random_state=42
    )
    
    rf_reg = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    gb_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
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
        "scaler": StandardScaler().fit(X),
        "rf_clf": rf_clf,
        "svm_clf": svm_clf,
        "rf_reg": rf_reg,
        "gb_reg": gb_reg,
        "ridge_reg": ridge_reg,
        "X_train": Xc_train,
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
# Fetch live weather with more data
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
    
    # Get hourly data
    hourly_times = data["hourly"]["time"]
    hourly_vis = data["hourly"]["visibility"]
    current_idx = hourly_times.index(current["time"]) if current["time"] in hourly_times else 0
    visibility_km = hourly_vis[current_idx] / 1000
    
    # Weather code mapping
    weather_codes = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall",
        95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
    }
    
    weather_code = current["weather_code"]
    weather_desc = weather_codes.get(weather_code, "Unknown")
    
    # Check if it rained today
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
        "Weather": weather_desc,
        "Precipitation": precip_today,
    }

# ------------------------------------------------------------------
# Create stunning 3D globe with rotation
# ------------------------------------------------------------------
def create_animated_globe(selected_city):
    names = list(CITIES.keys())
    lats = [CITIES[n]["lat"] for n in names]
    lons = [CITIES[n]["lon"] for n in names]
    countries = [CITIES[n]["country"] for n in names]
    
    # Color gradient
    colors = []
    for n in names:
        if n == selected_city:
            colors.append("#FF6B6B")
        else:
            colors.append("#4ECDC4")
    
    sizes = [12 if n == selected_city else 8 for n in names]
    
    # Create figure with multiple traces
    fig = go.Figure()
    
    # Add globe base
    fig.add_trace(go.Scattergeo(
        lat=lats,
        lon=lons,
        text=[f"<b>{n}</b><br>{countries[i]}<br>📍 Click to select" for i, n in enumerate(names)],
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(width=2, color="white"),
            sizemode="area",
            sizeref=2.*max(sizes)/(40.**2),
            sizemin=4
        ),
        hovertemplate="<b>%{text}</b><extra></extra>",
        textposition="top center",
        textfont=dict(size=9, color="white", family="Inter"),
        name="Cities"
    ))
    
    # Add rotation animation frames
    frames = []
    for angle in range(0, 360, 5):
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
    
    fig.update_geos(
        projection_type="orthographic",
        showland=True,
        landcolor="rgb(40, 40, 60)",
        showocean=True,
        oceancolor="rgb(20, 30, 50)",
        showcountries=True,
        countrycolor="rgb(100, 100, 150)",
        showcoastlines=True,
        coastlinecolor="rgb(150, 150, 200)",
        showframe=False,
        showlakes=True,
        lakecolor="rgb(30, 50, 80)",
    )
    
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
            buttons=[dict(
                label="▶ Rotate",
                method="animate",
                args=[None, dict(
                    frame=dict(duration=50, redraw=True),
                    fromcurrent=True,
                    mode="immediate",
                    transition=dict(duration=0)
                )]
            ), dict(
                label="⏹ Stop",
                method="animate",
                args=[[None], dict(
                    frame=dict(duration=0, redraw=False),
                    mode="immediate",
                    transition=dict(duration=0)
                )]
            )]
        )],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    
    return fig

# ------------------------------------------------------------------
# Load and train
# ------------------------------------------------------------------
with st.spinner("🌐 Loading data and training ensemble models..."):
    raw_df, daily = load_and_prepare_data()
    models = train_models(daily)

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3.5rem; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🌤️ WeatherAI
        </h1>
        <p style="font-size: 1.1rem; color: #6b7280; margin-top: 0.5rem; font-weight: 500;">
            Next-Generation Weather Intelligence · Powered by Ensemble Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🌧️ Rain Prediction", 
    "🌡️ Temperature Prediction", 
    "🌍 Live Global", 
    "📊 Advanced Analytics"
])

# ==================================================================
# TAB 1: RAIN PREDICTION with confidence
# ==================================================================
with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("☔ Will it rain tomorrow?")
    st.caption("Powered by Random Forest + SVM ensemble with confidence scoring")
    
    col1, col2 = st.columns(2)
    with col1:
        temp_c = st.number_input("🌡️ Temperature (°C)", value=20.0, step=0.5, key="rain_temp")
        app_temp_c = st.number_input("🌡️ Apparent Temperature (°C)", value=19.0, step=0.5, key="rain_apptemp")
        humidity_c = st.slider("💧 Humidity", 0.0, 1.0, 0.70, step=0.01, key="rain_hum")
        wind_c = st.number_input("💨 Wind Speed (km/h)", value=12.0, step=1.0, key="rain_wind")
        
    with col2:
        vis_c = st.number_input("👁️ Visibility (km)", value=10.0, step=0.5, key="rain_vis")
        pres_c = st.number_input("📊 Pressure (millibars)", value=1015.0, step=1.0, key="rain_pres")
        rain_today_c = st.radio("☔ Did it rain TODAY?", ["No", "Yes"], key="rain_today")
        
        # Advanced features
        month = st.selectbox("📅 Month", list(range(1, 13)), key="rain_month")
        day_of_week = st.selectbox("📆 Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], key="rain_dow")
    
    if st.button("🔮 Predict Rain Tomorrow", type="primary", key="btn_rain"):
        # Calculate advanced features
        temp_range = temp_c - app_temp_c
        humidity_pressure = humidity_c * pres_c
        wind_visibility = wind_c * vis_c
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        dow_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
        dow = dow_map[day_of_week]
        
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
            "Month": month,
            "DayOfWeek": dow,
            "Month_sin": month_sin,
            "Month_cos": month_cos,
        }])[models["features"]]
        
        scaled = models["scaler"].transform(input_df)
        
        # Get predictions from both models
        rf_pred = models["rf_clf"].predict(scaled)[0]
        svm_pred = models["svm_clf"].predict(scaled)[0]
        rf_prob = models["rf_clf"].predict_proba(scaled)[0]
        svm_prob = models["svm_clf"].predict_proba(scaled)[0]
        
        # Ensemble prediction (majority vote)
        ensemble_pred = (rf_pred + svm_pred) // 2
        
        # Confidence score (average of probabilities)
        confidence = np.mean([rf_prob[1 if ensemble_pred == 1 else 0], svm_prob[1 if ensemble_pred == 1 else 0]])
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if ensemble_pred == 1:
                st.markdown(f"""
                <div style="text-align: center; padding: 2rem;">
                    <div style="font-size: 4rem;">🌧️</div>
                    <h2 style="color: #667eea; margin: 0.5rem 0;">Rain Likely Tomorrow</h2>
                    <div class="status-rain">Confidence: {confidence:.1%}</div>
                    <p style="color: #6b7280; margin-top: 1rem;">
                        Ensemble prediction from Random Forest and SVM models
                    </p>
                    <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;">
                        <div><small>RF: {rf_prob[1]:.1%}</small></div>
                        <div><small>SVM: {svm_prob[1]:.1%}</small></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="text-align: center; padding: 2rem;">
                    <div style="font-size: 4rem;">☀️</div>
                    <h2 style="color: #f093fb; margin: 0.5rem 0;">No Rain Tomorrow</h2>
                    <div class="status-sunny">Confidence: {confidence:.1%}</div>
                    <p style="color: #6b7280; margin-top: 1rem;">
                        Ensemble prediction from Random Forest and SVM models
                    </p>
                    <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;">
                        <div><small>RF: {rf_prob[0]:.1%}</small></div>
                        <div><small>SVM: {svm_prob[0]:.1%}</small></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.caption(f"🎯 Model Accuracy: {models['ensemble_clf_acc']*100:.2f}% · Ensemble: Random Forest + SVM")
    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================================
# TAB 2: TEMPERATURE PREDICTION with ensemble
# ==================================================================
with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🌡️ Tomorrow's Temperature Prediction")
    st.caption("Ensemble of Random Forest, Gradient Boosting, and Ridge Regression")
    
    col1, col2 = st.columns(2)
    with col1:
        temp_r = st.number_input("🌡️ Temperature (°C)", value=20.0, step=0.5, key="temp_temp")
        app_temp_r = st.number_input("🌡️ Apparent Temperature (°C)", value=19.0, step=0.5, key="temp_apptemp")
        humidity_r = st.slider("💧 Humidity", 0.0, 1.0, 0.70, step=0.01, key="temp_hum")
        wind_r = st.number_input("💨 Wind Speed (km/h)", value=12.0, step=1.0, key="temp_wind")
        
    with col2:
        vis_r = st.number_input("👁️ Visibility (km)", value=10.0, step=0.5, key="temp_vis")
        pres_r = st.number_input("📊 Pressure (millibars)", value=1015.0, step=1.0, key="temp_pres")
        rain_today_r = st.radio("☔ Did it rain TODAY?", ["No", "Yes"], key="temp_today")
        month_r = st.selectbox("📅 Month", list(range(1, 13)), key="temp_month")
    
    if st.button("🔮 Predict Temperature Tomorrow", type="primary", key="btn_temp"):
        temp_range = temp_r - app_temp_r
        humidity_pressure = humidity_r * pres_r
        wind_visibility = wind_r * vis_r
        month_sin = np.sin(2 * np.pi * month_r / 12)
        month_cos = np.cos(2 * np.pi * month_r / 12)
        
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
            "Month": month_r,
            "DayOfWeek": 0,  # Default
            "Month_sin": month_sin,
            "Month_cos": month_cos,
        }])[models["features"]]
        
        scaled = models["scaler"].transform(input_df)
        
        # Get predictions from all three models
        rf_pred = models["rf_reg"].predict(scaled)[0]
        gb_pred = models["gb_reg"].predict(scaled)[0]
        ridge_pred = models["ridge_reg"].predict(scaled)[0]
        
        # Ensemble average
        ensemble_pred = (rf_pred + gb_pred + ridge_pred) / 3
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem;">
                <div style="font-size: 4rem;">🌡️</div>
                <h2 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3rem; margin: 0.5rem 0;">
                    {ensemble_pred:.1f}°C
                </h2>
                <p style="color: #6b7280;">Ensemble prediction for tomorrow</p>
                <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;">
                    <div><small>RF: {rf_pred:.1f}°C</small></div>
                    <div><small>GB: {gb_pred:.1f}°C</small></div>
                    <div><small>Ridge: {ridge_pred:.1f}°C</small></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.caption(f"📊 RMSE: {models['ensemble_reg_rmse']:.2f}°C · R²: {models['ensemble_reg_r2']:.3f}")
    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================================
# TAB 3: LIVE GLOBAL with 3D globe
# ==================================================================
with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🌍 Live Global Weather Intelligence")
    st.caption("Real-time weather data from Open-Meteo · 3D interactive globe")
    
    # Globe and selector layout
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_city = st.selectbox("🌆 Select a city", list(CITIES.keys()), key="globe_city")
        st.plotly_chart(create_animated_globe(selected_city), use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Quick Actions")
        if st.button("🌐 Fetch Live Weather", type="primary", key="btn_globe"):
            lat, lon = CITIES[selected_city]["lat"], CITIES[selected_city]["lon"]
            
            try:
                with st.spinner(f"🔄 Fetching live data for {selected_city}..."):
                    live = fetch_live_weather(lat, lon)
                    
                    # Display live metrics
                    st.markdown("#### Current Conditions")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">🌡️ Temperature</div>
                            <div class="metric-value">{live['Temperature (C)']:.1f}°C</div>
                            <div style="font-size: 0.8rem; color: #6b7280;">Feels like {live['Apparent Temperature (C)']:.1f}°C</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_b:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">💧 Humidity</div>
                            <div class="metric-value">{live['Humidity']*100:.0f}%</div>
                            <div style="font-size: 0.8rem; color: #6b7280;">{live['Weather']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    col_c, col_d = st.columns(2)
                    with col_c:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">💨 Wind Speed</div>
                            <div class="metric-value">{live['Wind Speed (km/h)']:.1f}</div>
                            <div style="font-size: 0.8rem; color: #6b7280;">km/h</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_d:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">📊 Pressure</div>
                            <div class="metric-value">{live['Pressure (millibars)']:.0f}</div>
                            <div style="font-size: 0.8rem; color: #6b7280;">millibars</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Generate predictions
                    st.markdown("#### 🔮 Tomorrow's Forecast")
                    
                    # Create input for prediction
                    temp_range = live['Temperature (C)'] - live['Apparent Temperature (C)']
                    humidity_pressure = live['Humidity'] * live['Pressure (millibars)']
                    wind_visibility = live['Wind Speed (km/h)'] * live['Visibility (km)']
                    
                    current_month = datetime.now().month
                    month_sin = np.sin(2 * np.pi * current_month / 12)
                    month_cos = np.cos(2 * np.pi * current_month / 12)
                    
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
                        "Month": current_month,
                        "DayOfWeek": datetime.now().weekday(),
                        "Month_sin": month_sin,
                        "Month_cos": month_cos,
                    }])[models["features"]]
                    
                    pred_scaled = models["scaler"].transform(pred_df)
                    
                    # Rain prediction
                    rf_pred = models["rf_clf"].predict(pred_scaled)[0]
                    svm_pred = models["svm_clf"].predict(pred_scaled)[0]
                    ensemble_rain = (rf_pred + svm_pred) // 2
                    
                    # Temp prediction
                    rf_temp = models["rf_reg"].predict(pred_scaled)[0]
                    gb_temp = models["gb_reg"].predict(pred_scaled)[0]
                    ridge_temp = models["ridge_reg"].predict(pred_scaled)[0]
                    ensemble_temp = (rf_temp + gb_temp + ridge_temp) / 3
                    
                    col_e, col_f = st.columns(2)
                    with col_e:
                        if ensemble_rain == 1:
                            st.error(f"🌧️ **Rain likely** in {selected_city}")
                        else:
                            st.success(f"☀️ **No rain** in {selected_city}")
                    
                    with col_f:
                        st.info(f"🌡️ **Temperature:** {ensemble_temp:.1f}°C")
                    
            except Exception as e:
                st.error(f"⚠️ Couldn't fetch weather data: {str(e)}")
                st.info("💡 Make sure you have an internet connection and try again.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================================
# TAB 4: ADVANCED ANALYTICS
# ==================================================================
with tab4:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📊 Advanced Analytics & Insights")
    
    tabs = st.tabs(["📈 Visualizations", "🧠 Model Performance", "📁 Data Explorer", "💻 Source Code"])
    
    with tabs[0]:
        st.markdown("### Interactive Analysis Dashboard")
        
        # Two rows of visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Correlation heatmap with Plotly (interactive)
            st.markdown("#### Feature Correlation Matrix")
            corr_data = daily.drop(columns=["Date"]).select_dtypes(include=[np.number]).corr()
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=corr_data.values,
                x=corr_data.columns,
                y=corr_data.columns,
                colorscale="RdBu",
                zmid=0,
                text=corr_data.values.round(2),
                texttemplate="%{text}",
                textfont={"size": 8},
                hoverongaps=False,
                colorbar=dict(title="Correlation")
            ))
            
            fig_heatmap.update_layout(
                height=400,
                margin=dict(l=0, r=0, t=30, b=0),
                font=dict(family="Inter", size=10)
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with col2:
            # Feature importance
            st.markdown("#### Feature Importance (Random Forest)")
            feature_importance = pd.DataFrame({
                'Feature': models["features"],
                'Importance': models["rf_reg"].feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig_imp = go.Figure(data=go.Bar(
                x=feature_importance['Importance'],
                y=feature_importance['Feature'],
                orientation='h',
                marker=dict(
                    color=feature_importance['Importance'],
                    colorscale="Viridis",
                    showscale=True
                ),
                text=feature_importance['Importance'].round(3),
                textposition='outside'
            ))
            
            fig_imp.update_layout(
                height=400,
                margin=dict(l=0, r=30, t=30, b=0),
                xaxis_title="Importance Score",
                font=dict(family="Inter", size=10)
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        
        # Second row
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("#### Actual vs Predicted Temperature")
            fig_scatter = go.Figure(data=go.Scatter(
                x=models["yr_test"],
                y=models["reg_pred"],
                mode="markers",
                marker=dict(
                    color=models["yr_test"] - models["reg_pred"],
                    colorscale="RdBu",
                    showscale=True,
                    size=8,
                    opacity=0.6
                ),
                text=[f"Actual: {a:.1f}°C<br>Predicted: {p:.1f}°C" for a, p in zip(models["yr_test"], models["reg_pred"])],
                hovertemplate="%{text}<extra></extra>"
            ))
            
            fig_scatter.add_trace(go.Scatter(
                x=[models["yr_test"].min(), models["yr_test"].max()],
                y=[models["yr_test"].min(), models["yr_test"].max()],
                mode="lines",
                line=dict(color="red", dash="dash", width=2),
                name="Perfect Prediction"
            ))
            
            fig_scatter.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title="Actual Temperature (°C)",
                yaxis_title="Predicted Temperature (°C)",
                font=dict(family="Inter", size=10),
                showlegend=False
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col4:
            st.markdown("#### Rain Prediction Confusion Matrix")
            fig_cm = go.Figure(data=go.Heatmap(
                z=models["clf_cm"],
                x=["No Rain", "Rain"],
                y=["No Rain", "Rain"],
                text=models["clf_cm"],
                texttemplate="%{text}",
                textfont={"size": 16},
                colorscale="Blues",
                showscale=True,
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
            ))
            
            fig_cm.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title="Predicted",
                yaxis_title="Actual",
                font=dict(family="Inter", size=12),
            )
            st.plotly_chart(fig_cm, use_container_width=True)
    
    with tabs[1]:
        st.markdown("### 🧠 Model Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Rain Prediction</div>
                <div class="metric-value">{models['ensemble_clf_acc']*100:.1f}%</div>
                <div style="font-size: 0.7rem; color: #6b7280;">Ensemble Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Temp Prediction RMSE</div>
                <div class="metric-value">{models['ensemble_reg_rmse']:.2f}°C</div>
                <div style="font-size: 0.7rem; color: #6b7280;">Root Mean Square Error</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Temp Prediction R²</div>
                <div class="metric-value">{models['ensemble_reg_r2']:.3f}</div>
                <div style="font-size: 0.7rem; color: #6b7280;">Coefficient of Determination</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Calculate cross-validation score
            cv_scores = cross_val_score(
                models["rf_clf"], 
                models["X_train"], 
                daily["RainTomorrow"][:len(models["X_train"])],
                cv=5
            )
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Cross-Validation</div>
                <div class="metric-value">{cv_scores.mean()*100:.1f}%</div>
                <div style="font-size: 0.7rem; color: #6b7280;">5-Fold CV Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Model comparison
        st.markdown("#### Model Performance Comparison")
        
        comparison_data = pd.DataFrame({
            'Model': ['Random Forest', 'SVM', 'Ensemble'],
            'Accuracy': [
                accuracy_score(models["yc_test"], models["rf_clf"].predict(models["X_test"])),
                accuracy_score(models["yc_test"], models["svm_clf"].predict(models["X_test"])),
                models['ensemble_clf_acc']
            ]
        })
        
        fig_comparison = go.Figure(data=go.Bar(
            x=comparison_data['Model'],
            y=comparison_data['Accuracy'],
            text=comparison_data['Accuracy'].round(3),
            textposition='auto',
            marker=dict(
                color=['#667eea', '#764ba2', '#f093fb'],
                line=dict(width=2, color='white')
            )
        ))
        
        fig_comparison.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis_title="Accuracy Score",
            yaxis=dict(range=[0.7, 1.0]),
            font=dict(family="Inter", size=12),
            showlegend=False
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    with tabs[2]:
        st.markdown("### 📁 Data Explorer")
        sub_tab1, sub_tab2 = st.tabs(["Raw Data", "Processed Data", "Statistics"])
        
        with sub_tab1:
            st.write(f"**Shape:** {raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns")
            st.dataframe(raw_df.head(100), use_container_width=True, height=400)
        
        with sub_tab2:
            st.write(f"**Shape:** {daily.shape[0]:,} rows × {daily.shape[1]} columns")
            st.dataframe(daily.head(100), use_container_width=True, height=400)
        
        with sub_tab3:
            st.markdown("#### Dataset Statistics")
            
            numeric_cols = daily.select_dtypes(include=[np.number]).columns
            stats_df = daily[numeric_cols].describe()
            st.dataframe(stats_df, use_container_width=True)
            
            # Missing values
            missing = daily.isnull().sum()
            if missing.sum() > 0:
                st.warning(f"⚠️ Found {missing.sum()} missing values in the dataset")
                st.dataframe(missing[missing > 0].to_frame('Missing Count'), use_container_width=True)
    
    with tabs[3]:
        st.markdown("### 💻 Source Code & Methodology")
        
        with st.expander("📦 Data Preprocessing Pipeline", expanded=False):
            st.code('''
def load_and_prepare_data():
    df = pd.read_csv("weatherHistory.csv")
    
    # Clean and preprocess
    df = df.drop_duplicates()
    df = df.drop(columns=["Loud Cover"], errors='ignore')
    df["Precip Type"] = df["Precip Type"].fillna("none")
    
    # Create time features
    df["Formatted Date"] = pd.to_datetime(df["Formatted Date"], utc=True)
    df["Hour"] = df["Formatted Date"].dt.hour
    df["Month"] = df["Formatted Date"].dt.month
    df["DayOfWeek"] = df["Formatted Date"].dt.dayofweek
    
    # Cyclical encoding for time features
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    
    # Daily aggregation
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
        "Month_sin": "mean",
        "Month_cos": "mean",
    }).reset_index()
    
    # Create derived features
    daily["TempRange"] = daily["Temperature (C)"] - daily["Apparent Temperature (C)"]
    daily["HumidityPressure"] = daily["Humidity"] * daily["Pressure (millibars)"]
    daily["WindVisibility"] = daily["Wind Speed (km/h)"] * daily["Visibility (km)"]
    
    return df, daily
            ''', language="python")
        
        with st.expander("🤖 Model Training (Ensemble Approach)", expanded=False):
            st.code('''
# Features for training
features = [
    "Temperature (C)", "Apparent Temperature (C)", "Humidity",
    "Wind Speed (km/h)", "Visibility (km)", "Pressure (millibars)", 
    "RainToday", "TempRange", "HumidityPressure", "WindVisibility",
    "Month", "DayOfWeek", "Month_sin", "Month_cos"
]

# Classification Ensemble
rf_clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
svm_clf = SVC(kernel="rbf", probability=True, random_state=42)

# Regression Ensemble
rf_reg = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
gb_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
ridge_reg = Ridge(alpha=1.0)

# Train all models
rf_clf.fit(X_train_scaled, y_train_class)
svm_clf.fit(X_train_scaled, y_train_class)
rf_reg.fit(X_train_scaled, y_train_reg)
gb_reg.fit(X_train_scaled, y_train_reg)
ridge_reg.fit(X_train_scaled, y_train_reg)

# Ensemble predictions
ensemble_clf_pred = (rf_clf_pred + svm_clf_pred) // 2
ensemble_reg_pred = (rf_reg_pred + gb_reg_pred + ridge_reg_pred) / 3
            ''', language="python")
        
        with st.expander("🌐 Live Weather Integration", expanded=False):
            st.code('''
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
    data = response.json()
    
    # Extract current conditions
    return {
        "Temperature (C)": data["current"]["temperature_2m"],
        "Apparent Temperature (C)": data["current"]["apparent_temperature"],
        "Humidity": data["current"]["relative_humidity_2m"] / 100,
        "Wind Speed (km/h)": data["current"]["wind_speed_10m"],
        "Visibility (km)": visibility_km,
        "Pressure (millibars)": data["current"]["pressure_msl"],
        "RainToday": rain_today,
    }
            ''', language="python")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown("""
<div style="text-align: center; padding: 2rem 0; opacity: 0.6; font-size: 0.8rem;">
    <p>WeatherAI v3.0 — Built with ❤️ using Streamlit, Scikit-learn, and Plotly</p>
    <p style="font-size: 0.7rem;">Powered by Open-Meteo API · Ensemble learning models</p>
</div>
""", unsafe_allow_html=True)
