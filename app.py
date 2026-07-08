"""
=========================================================================
 WEATHERAI · MINIMALIST PREMIUM
=========================================================================
 Inspired by Nike's design philosophy:
   - Clean, uncluttered layout
   - Bold typography with hierarchy
   - Generous whitespace
   - Subtle interactions
   - Premium but approachable
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
from datetime import datetime

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
    page_title="WeatherAI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------------
# Nike-Inspired Minimalist CSS
# ------------------------------------------------------------------
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Reset */
    .stApp {
        background: #ffffff;
    }
    
    .block-container {
        padding: 0 4rem !important;
        max-width: 1280px !important;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Bold minimal header */
    .brand-header {
        padding: 2rem 0 1rem 0;
        border-bottom: 1px solid #f0f0f0;
        margin-bottom: 3rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .brand-logo {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        color: #1a1a1a;
        text-decoration: none;
    }
    
    .brand-logo span {
        color: #e63946;
    }
    
    .brand-tagline {
        font-size: 0.75rem;
        font-weight: 500;
        color: #999;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    
    /* Clean cards */
    .clean-card {
        background: #ffffff;
        border: 1px solid #f0f0f0;
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.2s ease;
    }
    
    .clean-card:hover {
        border-color: #1a1a1a;
    }
    
    /* Bold section titles */
    .section-title {
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #1a1a1a;
        margin-bottom: 0.25rem;
    }
    
    .section-subtitle {
        font-size: 0.9rem;
        color: #999;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Minimal buttons */
    .stButton > button {
        background: #1a1a1a;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.2s ease;
        width: 100%;
        letter-spacing: 0.02em;
    }
    
    .stButton > button:hover {
        background: #e63946;
        transform: translateY(-1px);
    }
    
    /* Clean metrics */
    .metric-block {
        padding: 1.5rem 0;
        border-bottom: 1px solid #f5f5f5;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #1a1a1a;
        line-height: 1.1;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #999;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }
    
    /* Clean tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid #f0f0f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem 1rem 0;
        font-weight: 500;
        color: #999;
        font-size: 0.9rem;
        transition: all 0.2s ease;
        background: transparent !important;
        border: none !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #1a1a1a;
    }
    
    .stTabs [aria-selected="true"] {
        color: #1a1a1a !important;
        border-bottom: 2px solid #1a1a1a !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    
    /* Minimal inputs */
    .stNumberInput, .stSlider, .stSelectbox {
        background: #fafafa;
        border-radius: 8px;
        border: 1px solid #f0f0f0;
        padding: 0.25rem;
    }
    
    .stNumberInput:focus-within, .stSlider:focus-within, .stSelectbox:focus-within {
        border-color: #1a1a1a;
    }
    
    label {
        font-weight: 500 !important;
        color: #1a1a1a !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.02em !important;
    }
    
    /* Status badges - minimal */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1.5rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.02em;
    }
    
    .status-badge-rain {
        background: #e63946;
        color: white;
    }
    
    .status-badge-sunny {
        background: #f4a261;
        color: white;
    }
    
    .status-badge-neutral {
        background: #f0f0f0;
        color: #1a1a1a;
    }
    
    /* Prediction result */
    .result-block {
        text-align: center;
        padding: 3rem 2rem;
        background: #fafafa;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .result-icon {
        font-size: 4rem;
        margin-bottom: 0.5rem;
    }
    
    .result-title {
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #1a1a1a;
    }
    
    .result-temp {
        font-size: 4rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #1a1a1a;
    }
    
    /* Divider */
    .divider {
        border: none;
        border-top: 1px solid #f0f0f0;
        margin: 2rem 0;
    }
    
    /* Footer */
    .footer {
        padding: 3rem 0 2rem 0;
        border-top: 1px solid #f0f0f0;
        margin-top: 3rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .footer-text {
        font-size: 0.75rem;
        color: #ccc;
        font-weight: 400;
        letter-spacing: 0.05em;
    }
    
    .footer-links {
        display: flex;
        gap: 2rem;
    }
    
    .footer-links span {
        font-size: 0.7rem;
        color: #ccc;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Hide default Streamlit */
    #MainMenu, footer, header, .stDeployButton {
        display: none !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 4px;
        height: 4px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f5f5f5;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #ccc;
        border-radius: 10px;
    }
    
    /* Dataframe */
    .stDataFrame {
        border: 1px solid #f0f0f0 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# City Database
# ------------------------------------------------------------------
CITIES = {
    "New Delhi, India": {"lat": 28.6139, "lon": 77.2090},
    "Mumbai, India": {"lat": 19.0760, "lon": 72.8777},
    "New York, USA": {"lat": 40.7128, "lon": -74.0060},
    "Los Angeles, USA": {"lat": 34.0522, "lon": -118.2437},
    "London, UK": {"lat": 51.5074, "lon": -0.1278},
    "Paris, France": {"lat": 48.8566, "lon": 2.3522},
    "Tokyo, Japan": {"lat": 35.6762, "lon": 139.6503},
    "Singapore": {"lat": 1.3521, "lon": 103.8198},
    "Sydney, Australia": {"lat": -33.8688, "lon": 151.2093},
    "Dubai, UAE": {"lat": 25.2048, "lon": 55.2708},
    "Cairo, Egypt": {"lat": 30.0444, "lon": 31.2357},
    "Rio de Janeiro, Brazil": {"lat": -22.9068, "lon": -43.1729},
    "Beijing, China": {"lat": 39.9042, "lon": 116.4074},
    "Moscow, Russia": {"lat": 55.7558, "lon": 37.6173},
    "Berlin, Germany": {"lat": 52.5200, "lon": 13.4050},
    "Rome, Italy": {"lat": 41.9028, "lon": 12.4964},
    "Cape Town, South Africa": {"lat": -33.9249, "lon": 18.4241},
    "Bangkok, Thailand": {"lat": 13.7563, "lon": 100.5018},
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
    
    # Simple derived features
    daily["TempRange"] = daily["Temperature (C)"] - daily["Apparent Temperature (C)"]
    daily["HumidityPressure"] = daily["Humidity"] * daily["Pressure (millibars)"]
    daily["WindVisibility"] = daily["Wind Speed (km/h)"] * daily["Visibility (km)"]
    
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
        "Month", "DayOfWeek"
    ]
    
    X = daily[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Classification
    y_class = daily["RainTomorrow"]
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_scaled, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    svm_clf = SVC(kernel="rbf", random_state=42)
    
    rf_clf.fit(Xc_train, yc_train)
    svm_clf.fit(Xc_train, yc_train)
    
    # Regression
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
# Minimal Globe
# ------------------------------------------------------------------
def create_minimal_globe(selected_city):
    names = list(CITIES.keys())
    lats = [CITIES[n]["lat"] for n in names]
    lons = [CITIES[n]["lon"] for n in names]
    
    colors = ["#1a1a1a" if n != selected_city else "#e63946" for n in names]
    sizes = [6 if n != selected_city else 12 for n in names]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scattergeo(
        lat=lats,
        lon=lons,
        text=names,
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(width=1, color="white"),
            sizemode="area",
            sizeref=2.*max(sizes)/(40.**2),
            sizemin=3
        ),
        hovertemplate="<b>%{text}</b><extra></extra>",
        textposition="top center",
        textfont=dict(size=8, color="#1a1a1a", family="Inter"),
    ))
    
    fig.update_geos(
        projection_type="orthographic",
        showland=True,
        landcolor="#f5f5f5",
        showocean=True,
        oceancolor="#fafafa",
        showcountries=True,
        countrycolor="#e5e5e5",
        showcoastlines=True,
        coastlinecolor="#e5e5e5",
        showframe=False,
    )
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=400,
        geo=dict(
            projection=dict(
                rotation=dict(lon=0, lat=0)
            )
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter", color="#1a1a1a"),
    )
    
    return fig

# ------------------------------------------------------------------
# Load Data
# ------------------------------------------------------------------
with st.spinner("Loading..."):
    raw_df, daily = load_and_prepare_data()
    models = train_models(daily)

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown("""
<div class="brand-header">
    <div>
        <span class="brand-logo">✦ Weather<span>AI</span></span>
    </div>
    <div>
        <span class="brand-tagline">Intelligence · Precision · Clarity</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Rain", "Temperature", "Global", "Insights"
])

# ==================================================================
# TAB 1: RAIN
# ==================================================================
with tab1:
    st.markdown('<div class="clean-card">', unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">Rain Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Will it rain tomorrow? Enter today\'s conditions.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        temp_c = st.number_input("Temperature", value=20.0, step=0.5, key="r_temp", help="°C")
        app_temp_c = st.number_input("Feels like", value=19.0, step=0.5, key="r_apptemp", help="°C")
        humidity_c = st.slider("Humidity", 0.0, 1.0, 0.70, step=0.01, key="r_hum")
        wind_c = st.number_input("Wind speed", value=12.0, step=1.0, key="r_wind", help="km/h")
        
    with col2:
        vis_c = st.number_input("Visibility", value=10.0, step=0.5, key="r_vis", help="km")
        pres_c = st.number_input("Pressure", value=1015.0, step=1.0, key="r_pres", help="millibars")
        rain_today_c = st.radio("Rain today?", ["No", "Yes"], key="r_today")
        month = st.selectbox("Month", list(range(1, 13)), key="r_month")
    
    if st.button("Predict", type="primary", key="btn_rain"):
        temp_range = temp_c - app_temp_c
        humidity_pressure = humidity_c * pres_c
        wind_visibility = wind_c * vis_c
        
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
            "DayOfWeek": 0,
        }])[models["features"]]
        
        scaled = models["scaler"].transform(input_df)
        
        rf_pred = models["rf_clf"].predict(scaled)[0]
        svm_pred = models["svm_clf"].predict(scaled)[0]
        ensemble_pred = (rf_pred + svm_pred) // 2
        
        if ensemble_pred == 1:
            st.markdown("""
            <div class="result-block">
                <div class="result-icon">🌧️</div>
                <div class="result-title">Rain expected</div>
                <div style="margin-top: 0.5rem;">
                    <span class="status-badge status-badge-rain">Ensemble prediction</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-block">
                <div class="result-icon">☀️</div>
                <div class="result-title">Clear skies</div>
                <div style="margin-top: 0.5rem;">
                    <span class="status-badge status-badge-sunny">Ensemble prediction</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown(f'<div style="font-size: 0.8rem; color: #999; margin-top: 0.5rem;">Accuracy: {models["ensemble_clf_acc"]*100:.1f}% · Random Forest + SVM</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================================
# TAB 2: TEMPERATURE
# ==================================================================
with tab2:
    st.markdown('<div class="clean-card">', unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">Temperature Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Predict tomorrow\'s temperature with ensemble models.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        temp_r = st.number_input("Temperature", value=20.0, step=0.5, key="t_temp")
        app_temp_r = st.number_input("Feels like", value=19.0, step=0.5, key="t_apptemp")
        humidity_r = st.slider("Humidity", 0.0, 1.0, 0.70, step=0.01, key="t_hum")
        wind_r = st.number_input("Wind speed", value=12.0, step=1.0, key="t_wind")
        
    with col2:
        vis_r = st.number_input("Visibility", value=10.0, step=0.5, key="t_vis")
        pres_r = st.number_input("Pressure", value=1015.0, step=1.0, key="t_pres")
        rain_today_r = st.radio("Rain today?", ["No", "Yes"], key="t_today")
        month_r = st.selectbox("Month", list(range(1, 13)), key="t_month")
    
    if st.button("Predict", type="primary", key="btn_temp"):
        temp_range = temp_r - app_temp_r
        humidity_pressure = humidity_r * pres_r
        wind_visibility = wind_r * vis_r
        
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
            "DayOfWeek": 0,
        }])[models["features"]]
        
        scaled = models["scaler"].transform(input_df)
        
        rf_pred = models["rf_reg"].predict(scaled)[0]
        gb_pred = models["gb_reg"].predict(scaled)[0]
        ridge_pred = models["ridge_reg"].predict(scaled)[0]
        ensemble_pred = (rf_pred + gb_pred + ridge_pred) / 3
        
        st.markdown(f"""
        <div class="result-block">
            <div class="result-icon">🌡️</div>
            <div class="result-temp">{ensemble_pred:.1f}°C</div>
            <div style="font-size: 0.9rem; color: #999; margin-top: 0.25rem;">Ensemble prediction for tomorrow</div>
            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 0.5rem; font-size: 0.8rem; color: #ccc;">
                <span>RF: {rf_pred:.1f}°C</span>
                <span>GB: {gb_pred:.1f}°C</span>
                <span>Ridge: {ridge_pred:.1f}°C</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f'<div style="font-size: 0.8rem; color: #999; margin-top: 0.5rem;">RMSE: {models["ensemble_reg_rmse"]:.2f}°C · R²: {models["ensemble_reg_r2"]:.3f}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================================
# TAB 3: GLOBAL
# ==================================================================
with tab3:
    st.markdown('<div class="clean-card">', unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">Global Live</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Real-time weather from cities worldwide.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.5, 1])
    with col1:
        selected_city = st.selectbox("City", list(CITIES.keys()), key="globe_city")
        st.plotly_chart(create_minimal_globe(selected_city), use_container_width=True)
    
    with col2:
        if st.button("Get live data", type="primary", key="btn_globe"):
            lat, lon = CITIES[selected_city]["lat"], CITIES[selected_city]["lon"]
            
            try:
                with st.spinner(f"Fetching {selected_city}..."):
                    live = fetch_live_weather(lat, lon)
                    
                    st.markdown(f"""
                    <div class="metric-block">
                        <div class="metric-value">{live['Temperature (C)']:.1f}°C</div>
                        <div class="metric-label">Temperature</div>
                        <div style="font-size: 0.8rem; color: #ccc;">Feels like {live['Apparent Temperature (C)']:.1f}°C</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"""
                        <div class="metric-block">
                            <div class="metric-value" style="font-size: 1.8rem;">{live['Humidity']*100:.0f}%</div>
                            <div class="metric-label">Humidity</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_b:
                        st.markdown(f"""
                        <div class="metric-block">
                            <div class="metric-value" style="font-size: 1.8rem;">{live['Wind Speed (km/h)']:.1f}</div>
                            <div class="metric-label">Wind · km/h</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Quick prediction
                    st.markdown('<hr class="divider">', unsafe_allow_html=True)
                    st.markdown('<div style="font-weight: 600; font-size: 0.9rem;">Tomorrow\'s forecast</div>', unsafe_allow_html=True)
                    
                    temp_range = live['Temperature (C)'] - live['Apparent Temperature (C)']
                    humidity_pressure = live['Humidity'] * live['Pressure (millibars)']
                    wind_visibility = live['Wind Speed (km/h)'] * live['Visibility (km)']
                    
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
                        "Month": datetime.now().month,
                        "DayOfWeek": datetime.now().weekday(),
                    }])[models["features"]]
                    
                    pred_scaled = models["scaler"].transform(pred_df)
                    
                    rain_pred = models["rf_clf"].predict(pred_scaled)[0]
                    temp_pred = models["rf_reg"].predict(pred_scaled)[0]
                    
                    col_c, col_d = st.columns(2)
                    with col_c:
                        if rain_pred == 1:
                            st.markdown('<span class="status-badge status-badge-rain" style="font-size: 0.8rem;">🌧️ Rain</span>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span class="status-badge status-badge-sunny" style="font-size: 0.8rem;">☀️ Clear</span>', unsafe_allow_html=True)
                    
                    with col_d:
                        st.markdown(f'<span style="font-weight: 600; font-size: 1.1rem;">{temp_pred:.1f}°C</span>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Couldn't fetch data: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================================
# TAB 4: INSIGHTS
# ==================================================================
with tab4:
    st.markdown('<div class="clean-card">', unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Model performance and data exploration.</div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["Performance", "Data"])
    
    with tabs[0]:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-block">
                <div class="metric-value" style="font-size: 2rem;">{models['ensemble_clf_acc']*100:.1f}%</div>
                <div class="metric-label">Rain accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-block">
                <div class="metric-value" style="font-size: 2rem;">{models['ensemble_reg_rmse']:.2f}°C</div>
                <div class="metric-label">Temperature RMSE</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-block">
                <div class="metric-value" style="font-size: 2rem;">{models['ensemble_reg_r2']:.3f}</div>
                <div class="metric-label">R² score</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        
        # Confusion matrix with matplotlib
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(
            models["clf_cm"], 
            annot=True, 
            fmt='d', 
            cmap='Greys',
            xticklabels=['No Rain', 'Rain'],
            yticklabels=['No Rain', 'Rain'],
            cbar=False,
            ax=ax
        )
        ax.set_title('Confusion Matrix', fontsize=10, fontweight='600')
        ax.set_xlabel('Predicted', fontsize=8)
        ax.set_ylabel('Actual', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Scatter plot
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.scatter(models["yr_test"], models["reg_pred"], alpha=0.5, s=15, color='#1a1a1a')
        ax2.plot([models["yr_test"].min(), models["yr_test"].max()], 
                 [models["yr_test"].min(), models["yr_test"].max()], 
                 '--', color='#e63946', linewidth=1)
        ax2.set_xlabel('Actual °C', fontsize=9)
        ax2.set_ylabel('Predicted °C', fontsize=9)
        ax2.set_title('Temperature Predictions', fontsize=10, fontweight='600')
        ax2.grid(True, alpha=0.1)
        st.pyplot(fig2)
    
    with tabs[1]:
        st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.dataframe(daily.head(20), use_container_width=True)
        st.caption(f"{daily.shape[0]} rows · {daily.shape[1]} columns")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown("""
<div class="footer">
    <div class="footer-text">✦ WeatherAI · Built with clarity</div>
    <div class="footer-links">
        <span>Ensemble models</span>
        <span>Real-time</span>
        <span>20+ cities</span>
    </div>
</div>
""", unsafe_allow_html=True)
