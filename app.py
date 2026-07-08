"""
=========================================================================
 WEATHER PREDICTION WEB APP (Streamlit) - Premium Edition
=========================================================================
"""

import pandas as pd
import numpy as np
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix

# ------------------------------------------------------------------
# Page setup + Premium "Million Dollar" Styling
# ------------------------------------------------------------------
st.set_page_config(page_title="Weather AI", page_icon="🌤️", layout="centered")

# Injecting the 'Inter' font and modern minimalist CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    
    .block-container {
        padding-top: 2rem; 
        max-width: 900px;
    }
    
    h1, h2, h3 {
        font-weight: 600;
        letter-spacing: -0.5px;
    }

    /* Modern Metric Cards */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #f0f0f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(0,0,0,0.06);
    }
    
    div[data-testid="stMetric"] label {
        font-weight: 500;
        color: #888888;
    }

    /* Premium Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
        border: none;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Major world cities
# ------------------------------------------------------------------
CITIES = {
    "New Delhi, India": (28.6139, 77.2090), "Mumbai, India": (19.0760, 72.8777),
    "New York, USA": (40.7128, -74.0060), "Los Angeles, USA": (34.0522, -118.2437),
    "London, UK": (51.5074, -0.1278), "Paris, France": (48.8566, 2.3522),
    "Berlin, Germany": (52.5200, 13.4050), "Rome, Italy": (41.9028, 12.4964),
    "Madrid, Spain": (40.4168, -3.7038), "Moscow, Russia": (55.7558, 37.6173),
    "Dubai, UAE": (25.2048, 55.2708), "Beijing, China": (39.9042, 116.4074),
    "Shanghai, China": (31.2304, 121.4737), "Tokyo, Japan": (35.6762, 139.6503),
    "Seoul, South Korea": (37.5665, 126.9780), "Singapore": (1.3521, 103.8198),
    "Bangkok, Thailand": (13.7563, 100.5018), "Sydney, Australia": (-33.8688, 151.2093),
    "Cairo, Egypt": (30.0444, 31.2357), "Lagos, Nigeria": (6.5244, 3.3792),
    "Johannesburg, SA": (-26.2041, 28.0473), "Rio de Janeiro": (-22.9068, -43.1729),
    "Sao Paulo, Brazil": (-23.5505, -46.6333), "Buenos Aires": (-34.6037, -58.3816),
    "Toronto, Canada": (43.6532, -79.3832), "Mexico City": (19.4326, -99.1332),
    "Istanbul, Turkey": (41.0082, 28.9784),
}

# ------------------------------------------------------------------
# Data Loading & Prep
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

    return df, daily

# ------------------------------------------------------------------
# Model Training
# ------------------------------------------------------------------
@st.cache_resource
def train_models(daily):
    features = [
        "Temperature (C)", "Apparent Temperature (C)", "Humidity",
        "Wind Speed (km/h)", "Visibility (km)", "Pressure (millibars)", "RainToday"
    ]
    X = daily[features]

    # Classifier
    y_class = daily["RainTomorrow"]
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
    scaler_clf = StandardScaler()
    Xc_train_scaled = scaler_clf.fit_transform(Xc_train)
    Xc_test_scaled = scaler_clf.transform(Xc_test)
    clf_model = SVC(kernel="rbf")
    clf_model.fit(Xc_train_scaled, yc_train)
    clf_pred = clf_model.predict(Xc_test_scaled)
    clf_accuracy = accuracy_score(yc_test, clf_pred)
    clf_cm = confusion_matrix(yc_test, clf_pred)

    # Regressor
    y_reg = daily["TempTomorrow"]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    scaler_reg = StandardScaler()
    Xr_train_scaled = scaler_reg.fit_transform(Xr_train)
    Xr_test_scaled = scaler_reg.transform(Xr_test)
    reg_model = LinearRegression()
    reg_model.fit(Xr_train_scaled, yr_train)
    reg_pred = reg_model.predict(Xr_test_scaled)
    reg_rmse = np.sqrt(mean_squared_error(yr_test, reg_pred))
    reg_r2 = r2_score(yr_test, reg_pred)

    return {
        "features": features,
        "clf_model": clf_model, "scaler_clf": scaler_clf,
        "clf_accuracy": clf_accuracy, "clf_cm": clf_cm,
        "reg_model": reg_model, "scaler_reg": scaler_reg,
        "reg_rmse": reg_rmse, "reg_r2": reg_r2,
        "yr_test": yr_test, "reg_pred": reg_pred,
    }

# ------------------------------------------------------------------
# Live Weather API
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
    idx = hourly_times.index(current["time"]) if current["time"] in hourly_times else 0
    visibility_km = hourly_vis[idx] / 1000  

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
# Premium 3D Globe
# ------------------------------------------------------------------
def build_globe(selected_city):
    names = list(CITIES.keys())
    lats = [CITIES[n][0] for n in names]
    lons = [CITIES[n][1] for n in names]
    
    # Modern minimalistic colors: Teal and Coral
    colors = ["#14B8A6" if n != selected_city else "#F43F5E" for n in names]
    sizes = [8 if n != selected_city else 18 for n in names]

    fig = go.Figure(go.Scattergeo(
        lat=lats, lon=lons, text=names,
        mode="markers",
        marker=dict(
            size=sizes, color=colors, 
            line=dict(width=1, color="rgba(255,255,255,0.8)")
        ),
        hovertemplate="<b>%{text}</b><extra></extra>",
    ))
    
    fig.update_geos(
        projection_type="orthographic",
        showland=True, landcolor="#F3F4F6",  # Light minimalist gray
        showocean=True, oceancolor="rgba(0,0,0,0)", # Transparent ocean
        showcountries=True, countrycolor="#E5E7EB",
        showcoastlines=False,
        bgcolor="rgba(0,0,0,0)",
    )
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), 
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# ------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------
with st.spinner("Initializing Models..."):
    raw_df, daily = load_and_prepare_data()
    models = train_models(daily)

# App Header
st.title("Weather Intelligence")
st.markdown("<p style='color:#666666; margin-top:-10px; margin-bottom: 30px;'>Predictive forecasting powered by Machine Learning.</p>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🌧️ Rain", "🌡️ Temp", "🌍 Global Live", "📊 Analytics"])

# ==================================================================
# TAB 1: RAIN PREDICTION
# ==================================================================
with tab1:
    st.markdown("### Will it rain tomorrow?")
    
    col1, col2 = st.columns(2)
    with col1:
        temp_c = st.number_input("Temperature (°C)", value=20.0, key="rain_temp")
        app_temp_c = st.number_input("Feels Like (°C)", value=19.0, key="rain_apptemp")
        humidity_c = st.slider("Humidity", 0.0, 1.0, 0.70, key="rain_hum")
        wind_c = st.number_input("Wind Speed (km/h)", value=12.0, key="rain_wind")
    with col2:
        vis_c = st.number_input("Visibility (km)", value=10.0, key="rain_vis")
        pres_c = st.number_input("Pressure (mb)", value=1015.0, key="rain_pres")
        rain_today_c = st.radio("Did it rain today?", ["No", "Yes"], key="rain_today", horizontal=True)

    if st.button("Generate Rain Forecast", type="primary", key="btn_rain"):
        input_df = pd.DataFrame([{
            "Temperature (C)": temp_c, "Apparent Temperature (C)": app_temp_c,
            "Humidity": humidity_c, "Wind Speed (km/h)": wind_c,
            "Visibility (km)": vis_c, "Pressure (millibars)": pres_c,
            "RainToday": 1 if rain_today_c == "Yes" else 0
        }])[models["features"]]
        
        pred = models["clf_model"].predict(models["scaler_clf"].transform(input_df))[0]
        if pred == 1:
            st.error("🌧️ **Precipitation likely.** Keep an umbrella handy.")
        else:
            st.success("☀️ **Clear skies expected.** No rain forecasted.")
    
    st.caption(f"Powered by Support Vector Machine · Accuracy: {models['clf_accuracy']*100:.1f}%")

# ==================================================================
# TAB 2: TEMPERATURE PREDICTION
# ==================================================================
with tab2:
    st.markdown("### Tomorrow's Temperature Forecast")
    
    col1, col2 = st.columns(2)
    with col1:
        temp_r = st.number_input("Temperature (°C)", value=20.0, key="temp_temp")
        app_temp_r = st.number_input("Feels Like (°C)", value=19.0, key="temp_apptemp")
        humidity_r = st.slider("Humidity", 0.0, 1.0, 0.70, key="temp_hum")
        wind_r = st.number_input("Wind Speed (km/h)", value=12.0, key="temp_wind")
    with col2:
        vis_r = st.number_input("Visibility (km)", value=10.0, key="temp_vis")
        pres_r = st.number_input("Pressure (mb)", value=1015.0, key="temp_pres")
        rain_today_r = st.radio("Did it rain today?", ["No", "Yes"], key="temp_today", horizontal=True)

    if st.button("Generate Temp Forecast", type="primary", key="btn_temp"):
        input_df = pd.DataFrame([{
            "Temperature (C)": temp_r, "Apparent Temperature (C)": app_temp_r,
            "Humidity": humidity_r, "Wind Speed (km/h)": wind_r,
            "Visibility (km)": vis_r, "Pressure (millibars)": pres_r,
            "RainToday": 1 if rain_today_r == "Yes" else 0
        }])[models["features"]]
        
        pred = models["reg_model"].predict(models["scaler_reg"].transform(input_df))[0]
        st.info(f"🌡️ **Forecasted Temperature: {pred:.1f}°C**")

    st.caption(f"Powered by Linear Regression · RMSE: {models['reg_rmse']:.2f}°C")

# ==================================================================
# TAB 3: LIVE GLOBAL PREDICTION
# ==================================================================
with tab3:
    st.markdown("### Real-Time Global Telemetry")
    
    selected_city = st.selectbox("Target Location", list(CITIES.keys()), key="globe_city", label_visibility="collapsed")
    
    # Premium Globe Rendering
    st.plotly_chart(build_globe(selected_city), use_container_width=True, config={'displayModeBar': False})

    if st.button("Sync Live Data & Forecast", type="primary", key="btn_globe", use_container_width=True):
        lat, lon = CITIES[selected_city]
        try:
            with st.spinner(f"Establishing uplink to {selected_city}..."):
                live = fetch_live_weather(lat, lon)

            st.markdown("#### Current Conditions")
            c1, c2, c3 = st.columns(3)
            c1.metric("Temperature", f"{live['Temperature (C)']:.1f}°C")
            c2.metric("Humidity", f"{live['Humidity']*100:.0f}%")
            c3.metric("Wind Speed", f"{live['Wind Speed (km/h)']:.1f} km/h")

            input_df = pd.DataFrame([live])[models["features"]]
            rain_pred = models["clf_model"].predict(models["scaler_clf"].transform(input_df))[0]
            temp_pred = models["reg_model"].predict(models["scaler_reg"].transform(input_df))[0]

            st.markdown("#### 24-Hour Forecast")
            colA, colB = st.columns(2)
            with colA:
                if rain_pred == 1:
                    st.error(f"🌧️ Precipitation expected")
                else:
                    st.success(f"☀️ Clear skies expected")
            with colB:
                st.info(f"🌡️ Expected Temp: {temp_pred:.1f}°C")

        except requests.exceptions.RequestException:
            st.error("Network Error: Unable to reach meteorological uplink.")
        except Exception as e:
            st.error(f"Telemetry Error: {e}")

# ==================================================================
# TAB 4: INTERACTIVE ANALYTICS (Fully Plotly)
# ==================================================================
with tab4:
    st.markdown("### Data Architecture & Metrics")

    with st.expander("📂 Explore Raw Datasets"):
        st.dataframe(daily.head(100), use_container_width=True)

    with st.expander("🖼️ Interactive Visualizations", expanded=True):
        g1, g2 = st.columns(2)

        with g1:
            st.markdown("**Feature Correlation Matrix**")
            corr = daily.drop(columns=["Date"]).corr()
            fig_corr = px.imshow(
                corr, text_auto=".1f", aspect="auto", 
                color_continuous_scale="Tealgrn"
            )
            fig_corr.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=300)
            st.plotly_chart(fig_corr, use_container_width=True, config={'displayModeBar': False})

            st.markdown("**Temperature: Actual vs Predicted**")
            fig_scatter = px.scatter(
                x=models["yr_test"], y=models["reg_pred"], 
                opacity=0.6, color_discrete_sequence=["#14B8A6"]
            )
            # Add perfect prediction line
            fig_scatter.add_trace(go.Scatter(
                x=[models["yr_test"].min(), models["yr_test"].max()], 
                y=[models["yr_test"].min(), models["yr_test"].max()],
                mode='lines', line=dict(color='red', dash='dash'), name='Ideal'
            ))
            fig_scatter.update_layout(
                xaxis_title="Actual (°C)", yaxis_title="Predicted (°C)", 
                margin=dict(l=0, r=0, t=20, b=0), height=300, showlegend=False
            )
            st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})

        with g2:
            st.markdown("**Classifier Confusion Matrix**")
            fig_cm = px.imshow(
                models["clf_cm"], text_auto=True, aspect="auto", 
                color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual"),
                x=["No Rain", "Rain"], y=["No Rain", "Rain"]
            )
            fig_cm.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=300)
            st.plotly_chart(fig_cm, use_container_width=True, config={'displayModeBar': False})

            st.markdown("**Temperature Distribution**")
            fig_hist = px.histogram(
                daily, x="Temperature (C)", nbins=25, 
                color_discrete_sequence=["#3B82F6"]
            )
            fig_hist.update_layout(
                xaxis_title="Temperature (°C)", yaxis_title="Frequency",
                margin=dict(l=0, r=0, t=20, b=0), height=300
            )
            st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})
