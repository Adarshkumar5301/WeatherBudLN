import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------------
# 1. CYBERPUNK MINIMALIST THEME CONFIGURATION
# ------------------------------------------------------------------
st.set_page_config(page_title="NEBULA // Weather Intelligence Engine", page_icon="⚡", layout="wide")

# Custom injection for a true Cyberpunk Dark Mode experience
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Inter:wght@300;400;600&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0a0c10 !important;
        font-family: 'Inter', sans-serif;
        color: #c9d1d9;
    }
    
    /* Headers & Text */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 600 !important;
        letter-spacing: -0.5px;
    }
    .neon-text-teal { color: #00f2fe; text-shadow: 0 0 10px rgba(0, 242, 254, 0.3); }
    .neon-text-orange { color: #ff5e62; text-shadow: 0 0 10px rgba(255, 94, 98, 0.3); }
    
    /* Custom Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%) !important;
        border: 1px solid #30363d !important;
        border-radius: 12px !important;
        padding: 20px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #00f2fe !important;
        box-shadow: 0 4px 25px rgba(0, 242, 254, 0.15);
    }
    div[data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem !important;
        text-transform: uppercase;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        color: #f0f6fc !important;
        font-size: 1.8rem !important;
    }
    
    /* Clean Input Styling */
    .stNumberInput input, .stSlider div {
        background-color: #0d1117 !important;
        border: 1px solid #30363d !important;
        color: #f0f6fc !important;
        border-radius: 8px !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%) !important;
        color: #0a0c10 !important;
        border: none !important;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.4) !important;
    }
    
    /* Clean Tabs */
    button[data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #8b949e !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #00f2fe !important;
        border-bottom-color: #00f2fe !important;
    }
    
    /* Hide Default UI elements */
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 2. DATA PROCESSING & MODEL TRAINING ENGINE (CACHED)
# ------------------------------------------------------------------
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("weatherHistory.csv")
    df = df.drop_duplicates()
    if "Loud Cover" in df.columns:
        df = df.drop(columns=["Loud Cover"])
    df["Precip Type"] = df["Precip Type"].fillna("none")
    df["Formatted Date"] = pd.to_datetime(df["Formatted Date"], utc=True)
    df["Date"] = df["Formatted Date"].dt.date
    df["is_rain_hour"] = (df["Precip Type"] == "rain").astype(int)

    daily = df.groupby("Date").agg({
        "Temperature (C)": "mean",
        "Apparent Temperature (C)": "mean",
        "Humidity": "mean",
        "Wind Speed (km/h)": "mean",
        "Visibility (km)": "mean",
        "Pressure (millibars)": "mean",
        "is_rain_hour": "max"
    }).reset_index()
    daily = daily.rename(columns={"is_rain_hour": "RainToday"})

    daily["RainTomorrow"] = daily["RainToday"].shift(-1)
    daily["TempTomorrow"] = daily["Temperature (C)"].shift(-1)
    daily = daily.dropna().reset_index(drop=True)
    daily["RainTomorrow"] = daily["RainTomorrow"].astype(int)
    return daily

@st.cache_resource
def train_models(daily):
    features = ["Temperature (C)", "Apparent Temperature (C)", "Humidity", "Wind Speed (km/h)", "Visibility (km)", "Pressure (millibars)", "RainToday"]
    X = daily[features]
    
    # Classification (Rain)
    y_class = daily["RainTomorrow"]
    Xc_train, Xc_test, yc_train, _ = train_test_split(X, y_class, test_size=0.2, random_state=42)
    scaler_clf = StandardScaler()
    Xc_train_scaled = scaler_clf.fit_transform(Xc_train)
    clf_model = SVC(kernel="rbf", probability=True)
    clf_model.fit(Xc_train_scaled, yc_train)
    
    # Regression (Temperature)
    y_reg = daily["TempTomorrow"]
    Xr_train, Xr_test, yr_train, _ = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    scaler_reg = StandardScaler()
    Xr_train_scaled = scaler_reg.fit_transform(Xr_train)
    reg_model = LinearRegression()
    reg_model.fit(Xr_train_scaled, yr_train)
    
    return features, clf_model, scaler_clf, reg_model, scaler_reg

# Initialize data and models
try:
    daily_data = load_and_prepare_data()
    features, clf_model, scaler_clf, reg_model, scaler_reg = train_models(daily_data)
except Exception as e:
    st.error(f"Initialization Failed. Make sure 'weatherHistory.csv' is in the root directory. Error: {e}")
    st.stop()

# ------------------------------------------------------------------
# 3. LIVE GEOLOCATION & GLOBE UTILITIES
# ------------------------------------------------------------------
CITIES = {
    "New Delhi, India": (28.6139, 77.2090), "Mumbai, India": (19.0760, 72.8777),
    "Kolkata, India": (22.5726, 88.3639), "London, UK": (51.5074, -0.1278),
    "New York, USA": (40.7128, -74.0060), "Tokyo, Japan": (35.6762, 139.6503),
    "Paris, France": (48.8566, 2.3522), "Dubai, UAE": (25.2048, 55.2708),
    "Sydney, Australia": (-33.8688, 151.2093), "Singapore": (1.3521, 103.8198)
}

@st.cache_data(ttl=300)
def fetch_live_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,apparent_temperature,pressure_msl,wind_speed_10m&hourly=visibility&daily=precipitation_sum&timezone=auto&forecast_days=1"
    res = requests.get(url, timeout=10).json()
    current = res["current"]
    
    # Extracting current visibility window safely
    try: visibility_km = res["hourly"]["visibility"][0] / 1000
    except: visibility_km = 10.0
        
    precip = res["daily"]["precipitation_sum"][0]
    return {
        "Temperature (C)": current["temperature_2m"],
        "Apparent Temperature (C)": current["apparent_temperature"],
        "Humidity": current["relative_humidity_2m"] / 100,
        "Wind Speed (km/h)": current["wind_speed_10m"],
        "Visibility (km)": visibility_km,
        "Pressure (millibars)": current["pressure_msl"],
        "RainToday": 1 if (precip and precip > 0.1) else 0
    }

def render_cyber_globe(selected_city):
    names = list(CITIES.keys())
    lats = [CITIES[n][0] for n in names]
    lons = [CITIES[n][1] for n in names]
    
    # Active nodes styling
    colors = ["#00f2fe" if n != selected_city else "#ff5e62" for n in names]
    sizes = [8 if n != selected_city else 16 for n in names]
    
    fig = go.Figure(go.Scattergeo(
        lat=lats, lon=lons, text=names, mode="markers",
        marker=dict(size=sizes, color=colors, line=dict(width=1.5, color="#0a0c10")),
        hovertemplate="%{text}<extra></extra>"
    ))
    
    fig.update_geos(
        projection_type="orthographic",
        showland=True, landcolor="#161b22",
        showocean=True, oceancolor="#0d1117",
        showcountries=True, countrycolor="#30363d",
        showcoastlines=True, coastlinecolor="#21262d",
        bgcolor="rgba(0,0,0,0)"
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# ------------------------------------------------------------------
# 4. SITE LAYOUT & INTERFACE WRAPPING
# ------------------------------------------------------------------
st.markdown('# <span class="neon-text-teal">// NEBULA</span> WEATHER INTELLIGENCE', unsafe_allow_html=True)
st.caption("Predictive Engine Architecture Engine utilizing High-Dimensional Vector Spaces.")
st.markdown("---")

tab_global, tab_manual = st.st.tabs(["🌍 GLOBAL TELEMETRY ENGINE", "🎛️ CORE MATRIX MANUAL RUN"])

# --- TAB 1: HERO VIEW (LIVE GLOBAL GLOBE PREDICTION) ---
with tab_global:
    col_globe, col_metrics = st.columns([1.2, 1])
    
    with col_globe:
        st.markdown("### SYSTEM NODE MAP")
        selected_node = st.selectbox("Select Target Telemetry Node:", list(CITIES.keys()))
        st.plotly_chart(render_cyber_globe(selected_node), use_container_width=True, config={'displayModeBar': False})
        
    with col_metrics:
        st.markdown("### LIVE METRIC TELEMETRY")
        if st.button("EXECUTE QUANTUM PREDICTION MATRIX"):
            lat, lon = CITIES[selected_node]
            with st.spinner("Intercepting atmosphere data signals..."):
                live_data = fetch_live_weather(lat, lon)
                
            # Render Live Inputs
            m1, m2 = st.columns(2)
            m1.metric("Node Air Temp", f"{live_data['Temperature (C)']:.1f} °C")
            m2.metric("Relative Humidity", f"{int(live_data['Humidity']*100)} %")
            
            # Predict Pipeline
            input_df = pd.DataFrame([live_data])[features]
            rain_pred = clf_model.predict(scaler_clf.transform(input_df))[0]
            temp_pred = reg_model.predict(scaler_reg.transform(input_df))[0]
            
            st.markdown("#### PROJECTION MATRIX TARGETS (TOMORROW)")
            
            if rain_pred == 1:
                st.markdown('<div style="border: 1px solid #ff5e62; padding: 15px; border-radius: 8px; background: rgba(255,94,98,0.05);">'
                            '⚡ <strong class="neon-text-orange">PRECIPITATION EVENT DETECTED:</strong> Atmosphere vector matrices indicate Rain Tomorrow.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="border: 1px solid #00f2fe; padding: 15px; border-radius: 8px; background: rgba(0,242,254,0.05);">'
                            '☀️ <strong class="neon-text-teal">CLEAR ATMOSPHERE PROJECTED:</strong> Low probability of precipitation tomorrow.</div>', unsafe_allow_html=True)
                
            st.metric("Predicted Node Temperature", f"{temp_pred:.1f} °C")

# --- TAB 2: MANUAL TESTING CONTROL PANEL ---
with tab_manual:
    st.markdown("### CONFIGURE MANUAL VECTOR SPACE VARIABLES")
    
    c1, c2 = st.columns(2)
    with c1:
        manual_temp = st.number_input("Core Temperature (°C)", value=22.5)
        manual_app = st.number_input("Apparent Feels-Like Temp (°C)", value=21.0)
        manual_hum = st.slider("Humidity Ratio Vector", 0.0, 1.0, 0.65)
        manual_wind = st.number_input("Wind Speed Field Vector (km/h)", value=14.0)
    with c2:
        manual_vis = st.number_input("Optical Visibility Grid Range (km)", value=12.5)
        manual_pres = st.number_input("Barometric Pressure Base (millibars)", value=1013.2)
        manual_rain = st.radio("Did It Rain Current Cycle?", ["No Event (0)", "Precipitation Confirmed (1)"])
        
    if st.button("RUN ISOLATED SYSTEM SIMULATION"):
        manual_input = pd.DataFrame([{
            "Temperature (C)": manual_temp, "Apparent Temperature (C)": manual_app,
            "Humidity": manual_hum, "Wind Speed (km/h)": manual_wind,
            "Visibility (km)": manual_vis, "Pressure (millibars)": manual_pres,
            "RainToday": 1 if "Confirmed" in manual_rain else 0
        }])[features]
        
        m_rain_pred = clf_model.predict(scaler_clf.transform(manual_input))[0]
        m_temp_pred = reg_model.predict(scaler_reg.transform(manual_input))[0]
        
        st.markdown("---")
        st.markdown("### MANUAL MATRIX CALCULATION COMPLETE")
        
        res1, res2 = st.columns(2)
        with res1:
            if m_rain_pred == 1:
                st.subheader("🌧️ Rain Projected")
            else:
                st.subheader("☀️ Stable Conditions")
        with res2:
            st.metric("Simulated Temp Yield", f"{m_temp_pred:.2f} °C")
