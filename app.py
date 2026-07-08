"""
=========================================================================
 WEATHER PREDICTION WEB APP (Streamlit) - v2
=========================================================================
 4 tabs:
   1) Rain Prediction        - manual input -> Yes/No for tomorrow
   2) Temperature Prediction - manual input -> tomorrow's temperature
   3) Live Global Prediction - pick a real city on a 3D globe, pull its
                               ACTUAL current weather from the internet
                               (Open-Meteo, free, no API key), and predict
                               its tomorrow live
   4) Analysis                - dataset viewer + source code viewer +
                               collage of all our analysis graphs

 HOW TO RUN:
   1) Put weatherHistory.csv in the SAME folder as this file.
   2) Install what's needed (only once):
        pip install -r requirements.txt
   3) In terminal/cmd, inside that folder, run:
        streamlit run app.py
   4) It opens automatically in your browser (usually localhost:8501).

 NOTE: Tab 3 (Live Global Prediction) needs an actual internet connection
 on the computer you're running this on, since it calls a real weather API.
=========================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import streamlit as st
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix

# ------------------------------------------------------------------
# Page setup + light "minimalist" styling
# ------------------------------------------------------------------
st.set_page_config(page_title="Weather Prediction", page_icon="⛅", layout="centered")

st.markdown("""
<style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 2.5rem; max-width: 850px;}
    h1, h2, h3 {font-weight: 600;}
    div[data-testid="stMetric"] {
        background-color: #f7f9fb;
        border: 1px solid #e6e9ec;
        border-radius: 10px;
        padding: 12px;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# A short list of major world cities for the globe (name -> lat, lon)
# ------------------------------------------------------------------
CITIES = {
    "New Delhi, India": (28.6139, 77.2090),
    "Mumbai, India": (19.0760, 72.8777),
    "New York, USA": (40.7128, -74.0060),
    "Los Angeles, USA": (34.0522, -118.2437),
    "London, UK": (51.5074, -0.1278),
    "Paris, France": (48.8566, 2.3522),
    "Berlin, Germany": (52.5200, 13.4050),
    "Rome, Italy": (41.9028, 12.4964),
    "Madrid, Spain": (40.4168, -3.7038),
    "Moscow, Russia": (55.7558, 37.6173),
    "Dubai, UAE": (25.2048, 55.2708),
    "Beijing, China": (39.9042, 116.4074),
    "Shanghai, China": (31.2304, 121.4737),
    "Tokyo, Japan": (35.6762, 139.6503),
    "Seoul, South Korea": (37.5665, 126.9780),
    "Singapore": (1.3521, 103.8198),
    "Bangkok, Thailand": (13.7563, 100.5018),
    "Sydney, Australia": (-33.8688, 151.2093),
    "Cairo, Egypt": (30.0444, 31.2357),
    "Lagos, Nigeria": (6.5244, 3.3792),
    "Johannesburg, South Africa": (-26.2041, 28.0473),
    "Rio de Janeiro, Brazil": (-22.9068, -43.1729),
    "Sao Paulo, Brazil": (-23.5505, -46.6333),
    "Buenos Aires, Argentina": (-34.6037, -58.3816),
    "Toronto, Canada": (43.6532, -79.3832),
    "Mexico City, Mexico": (19.4326, -99.1332),
    "Istanbul, Turkey": (41.0082, 28.9784),
}


# ------------------------------------------------------------------
# Load + clean + prepare our training data (same as before)
# ------------------------------------------------------------------
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("weatherHistory.csv")

    df = df.drop_duplicates()
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
# Train both models (cached so this runs once, not on every click)
# ------------------------------------------------------------------
@st.cache_resource
def train_models(daily):
    features = [
        "Temperature (C)", "Apparent Temperature (C)", "Humidity",
        "Wind Speed (km/h)", "Visibility (km)", "Pressure (millibars)", "RainToday"
    ]
    X = daily[features]

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
# Fetch REAL current weather for a city from Open-Meteo (free, no key)
# cached for 5 minutes so repeated clicks on the same city during a
# demo don't spam the API unnecessarily
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

    # match today's visibility reading to the current hour
    hourly_times = data["hourly"]["time"]
    hourly_vis = data["hourly"]["visibility"]
    idx = hourly_times.index(current["time"]) if current["time"] in hourly_times else 0
    visibility_km = hourly_vis[idx] / 1000  # API gives meters, we trained on km

    precip_today = data["daily"]["precipitation_sum"][0]
    rain_today = 1 if precip_today and precip_today > 0.1 else 0

    return {
        "Temperature (C)": current["temperature_2m"],
        "Apparent Temperature (C)": current["apparent_temperature"],
        "Humidity": current["relative_humidity_2m"] / 100,   # % -> 0-1 scale, same as training data
        "Wind Speed (km/h)": current["wind_speed_10m"],
        "Visibility (km)": visibility_km,
        "Pressure (millibars)": current["pressure_msl"],
        "RainToday": rain_today,
    }


# ------------------------------------------------------------------
# Build the rotating globe (Plotly orthographic projection)
# ------------------------------------------------------------------
def build_globe(selected_city):
    names = list(CITIES.keys())
    lats = [CITIES[n][0] for n in names]
    lons = [CITIES[n][1] for n in names]
    colors = ["#3B82C4" if n != selected_city else "#E4572E" for n in names]
    sizes = [7 if n != selected_city else 15 for n in names]

    fig = go.Figure(go.Scattergeo(
        lat=lats, lon=lons, text=names,
        mode="markers",
        marker=dict(size=sizes, color=colors, line=dict(width=1, color="white")),
        hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_geos(
        projection_type="orthographic",
        showland=True, landcolor="rgb(235,235,235)",
        showocean=True, oceancolor="rgb(217,232,245)",
        showcountries=True, countrycolor="rgb(200,200,200)",
        showcoastlines=True, coastlinecolor="rgb(180,180,180)",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=420)
    return fig


# ------------------------------------------------------------------
# Load data + train models once
# ------------------------------------------------------------------
with st.spinner("Loading data and training models..."):
    raw_df, daily = load_and_prepare_data()
    models = train_models(daily)


# ------------------------------------------------------------------
# App header
# ------------------------------------------------------------------
st.title("⛅ Weather Prediction")
st.caption("Machine Learning Mini Project · predicts tomorrow's rain & temperature")

tab1, tab2, tab3, tab4 = st.tabs([
    "🌧️ Rain", "🌡️ Temperature", "🌍 Live Global", "📊 Analysis"
])


# ==================================================================
# TAB 1: RAIN PREDICTION (manual input)
# ==================================================================
with tab1:
    st.subheader("Will it rain tomorrow?")

    col1, col2 = st.columns(2)
    with col1:
        temp_c = st.number_input("Temperature (°C)", value=20.0, key="rain_temp")
        app_temp_c = st.number_input("Apparent Temperature (°C)", value=19.0, key="rain_apptemp")
        humidity_c = st.slider("Humidity", 0.0, 1.0, 0.70, key="rain_hum")
        wind_c = st.number_input("Wind Speed (km/h)", value=12.0, key="rain_wind")
    with col2:
        vis_c = st.number_input("Visibility (km)", value=10.0, key="rain_vis")
        pres_c = st.number_input("Pressure (millibars)", value=1015.0, key="rain_pres")
        rain_today_c = st.radio("Did it rain TODAY?", ["No", "Yes"], key="rain_today")

    if st.button("Predict Rain", type="primary", key="btn_rain"):
        input_df = pd.DataFrame([{
            "Temperature (C)": temp_c, "Apparent Temperature (C)": app_temp_c,
            "Humidity": humidity_c, "Wind Speed (km/h)": wind_c,
            "Visibility (km)": vis_c, "Pressure (millibars)": pres_c,
            "RainToday": 1 if rain_today_c == "Yes" else 0
        }])[models["features"]]
        pred = models["clf_model"].predict(models["scaler_clf"].transform(input_df))[0]
        if pred == 1:
            st.error("🌧️ **Yes, likely to RAIN tomorrow.** Carry an umbrella!")
        else:
            st.success("☀️ **No, not likely to rain tomorrow.**")

    st.caption(f"Model: SVM Classifier · Test Accuracy: {models['clf_accuracy']*100:.2f}%")


# ==================================================================
# TAB 2: TEMPERATURE PREDICTION (manual input)
# ==================================================================
with tab2:
    st.subheader("What will tomorrow's temperature be?")

    col1, col2 = st.columns(2)
    with col1:
        temp_r = st.number_input("Temperature (°C)", value=20.0, key="temp_temp")
        app_temp_r = st.number_input("Apparent Temperature (°C)", value=19.0, key="temp_apptemp")
        humidity_r = st.slider("Humidity", 0.0, 1.0, 0.70, key="temp_hum")
        wind_r = st.number_input("Wind Speed (km/h)", value=12.0, key="temp_wind")
    with col2:
        vis_r = st.number_input("Visibility (km)", value=10.0, key="temp_vis")
        pres_r = st.number_input("Pressure (millibars)", value=1015.0, key="temp_pres")
        rain_today_r = st.radio("Did it rain TODAY?", ["No", "Yes"], key="temp_today")

    if st.button("Predict Temperature", type="primary", key="btn_temp"):
        input_df = pd.DataFrame([{
            "Temperature (C)": temp_r, "Apparent Temperature (C)": app_temp_r,
            "Humidity": humidity_r, "Wind Speed (km/h)": wind_r,
            "Visibility (km)": vis_r, "Pressure (millibars)": pres_r,
            "RainToday": 1 if rain_today_r == "Yes" else 0
        }])[models["features"]]
        pred = models["reg_model"].predict(models["scaler_reg"].transform(input_df))[0]
        st.info(f"🌡️ **Predicted temperature for tomorrow: {pred:.1f}°C**")

    st.caption(f"Model: Linear Regression · RMSE: {models['reg_rmse']:.2f}°C · R²: {models['reg_r2']:.2f}")


# ==================================================================
# TAB 3: LIVE GLOBAL PREDICTION (real internet data)
# ==================================================================
with tab3:
    st.subheader("Pick a city, get its REAL weather, predict its tomorrow")
    st.caption("Pulls live current weather from the internet (Open-Meteo) — needs an internet connection.")

    selected_city = st.selectbox("Choose a city", list(CITIES.keys()), key="globe_city")
    st.plotly_chart(build_globe(selected_city), use_container_width=True)

    if st.button("🌐 Get Live Weather & Predict Tomorrow", type="primary", key="btn_globe"):
        lat, lon = CITIES[selected_city]
        try:
            with st.spinner(f"Fetching live weather for {selected_city}..."):
                live = fetch_live_weather(lat, lon)

            c1, c2, c3 = st.columns(3)
            c1.metric("Temperature", f"{live['Temperature (C)']:.1f}°C")
            c2.metric("Humidity", f"{live['Humidity']*100:.0f}%")
            c3.metric("Wind Speed", f"{live['Wind Speed (km/h)']:.1f} km/h")

            input_df = pd.DataFrame([live])[models["features"]]
            rain_pred = models["clf_model"].predict(models["scaler_clf"].transform(input_df))[0]
            temp_pred = models["reg_model"].predict(models["scaler_reg"].transform(input_df))[0]

            st.markdown("**Prediction for tomorrow:**")
            colA, colB = st.columns(2)
            with colA:
                if rain_pred == 1:
                    st.error(f"🌧️ Rain likely in {selected_city}")
                else:
                    st.success(f"☀️ No rain likely in {selected_city}")
            with colB:
                st.info(f"🌡️ Temperature: {temp_pred:.1f}°C")

        except requests.exceptions.RequestException:
            st.error("Couldn't reach the weather service. Check your internet connection and try again.")
        except Exception as e:
            st.error(f"Something went wrong while fetching/predicting: {e}")


# ==================================================================
# TAB 4: ANALYSIS - dataset viewer + source code + graph collage
# ==================================================================
with tab4:
    st.subheader("Dataset & Model Analysis")

    # ---------- Dataset viewer ----------
    with st.expander("📂 View Dataset"):
        sub_tab1, sub_tab2 = st.tabs(["Raw hourly data", "Processed daily data"])
        with sub_tab1:
            st.write(f"Shape: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")
            st.dataframe(raw_df.head(50), use_container_width=True)
        with sub_tab2:
            st.write(f"Shape: {daily.shape[0]} rows × {daily.shape[1]} columns")
            st.dataframe(daily.head(50), use_container_width=True)

    # ---------- Source code viewer ----------
    with st.expander("💻 View Source Code"):
        code_tab1, code_tab2, code_tab3 = st.tabs(["Data Cleaning", "Model Training", "Live Prediction"])

        with code_tab1:
            st.code('''
df = pd.read_csv("weatherHistory.csv")
df = df.drop_duplicates()                          # remove duplicate rows
df = df.drop(columns=["Loud Cover"])                # always 0, useless column
df["Precip Type"] = df["Precip Type"].fillna("none")  # missing = no rain/snow that hour

# convert hourly readings into one row per day
df["Formatted Date"] = pd.to_datetime(df["Formatted Date"], utc=True)
df["Date"] = df["Formatted Date"].dt.date
df["is_rain_hour"] = (df["Precip Type"] == "rain").astype(int)

daily = df.groupby("Date").agg({
    "Temperature (C)": "mean", "Humidity": "mean", ...
    "is_rain_hour": "max"
}).reset_index()

# create "tomorrow" targets by shifting one day back
daily["RainTomorrow"] = daily["RainToday"].shift(-1)
daily["TempTomorrow"] = daily["Temperature (C)"].shift(-1)
''', language="python")

        with code_tab2:
            st.code('''
features = ["Temperature (C)", "Apparent Temperature (C)", "Humidity",
            "Wind Speed (km/h)", "Visibility (km)", "Pressure (millibars)", "RainToday"]
X = daily[features]

# --- Rain classifier: SVM ---
y_class = daily["RainTomorrow"]
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
clf_model = SVC(kernel="rbf")
clf_model.fit(X_train_scaled, y_train)

# --- Temperature regressor: Linear Regression ---
y_reg = daily["TempTomorrow"]
X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
reg_model = LinearRegression()
reg_model.fit(scaler.fit_transform(X_train), y_train)
''', language="python")

        with code_tab3:
            st.code('''
# fetch REAL current weather for any city (Open-Meteo, free, no key needed)
url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
       "&current=temperature_2m,relative_humidity_2m,apparent_temperature,"
       "pressure_msl,wind_speed_10m&hourly=visibility&daily=precipitation_sum"
       "&timezone=auto&forecast_days=1")
data = requests.get(url, timeout=10).json()

live_features = {
    "Temperature (C)": data["current"]["temperature_2m"],
    "Humidity": data["current"]["relative_humidity_2m"] / 100,
    ...
}

# feed straight into our already-trained model
input_df = pd.DataFrame([live_features])[features]
rain_pred = clf_model.predict(scaler_clf.transform(input_df))[0]
temp_pred = reg_model.predict(scaler_reg.transform(input_df))[0]
''', language="python")

    # ---------- Graph collage ----------
    with st.expander("🖼️ Analysis Graphs", expanded=True):
        g1, g2 = st.columns(2)

        with g1:
            st.write("**Correlation Heatmap**")
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            corr = daily.drop(columns=["Date"]).corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".1f", annot_kws={"size": 6}, ax=ax1)
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)

            st.write("**Actual vs Predicted Temperature**")
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            ax3.scatter(models["yr_test"], models["reg_pred"], alpha=0.4, color="teal", s=10)
            ax3.plot([models["yr_test"].min(), models["yr_test"].max()],
                     [models["yr_test"].min(), models["yr_test"].max()], 'r--', linewidth=1.5)
            ax3.set_xlabel("Actual (°C)")
            ax3.set_ylabel("Predicted (°C)")
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)

        with g2:
            st.write("**Confusion Matrix — Rain Classifier**")
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            sns.heatmap(models["clf_cm"], annot=True, fmt="d", cmap="Blues",
                        xticklabels=["No Rain", "Rain"], yticklabels=["No Rain", "Rain"], ax=ax2)
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("Actual")
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

            st.write("**Feature Distributions**")
            fig4, ax4 = plt.subplots(figsize=(5, 4))
            daily["Temperature (C)"].hist(bins=25, color="steelblue", edgecolor="black", ax=ax4)
            ax4.set_xlabel("Temperature (°C)")
            ax4.set_ylabel("Number of days")
            st.pyplot(fig4, use_container_width=True)
            plt.close(fig4)

    # ---------- Metrics summary ----------
    st.write("**Model Performance Summary**")
    summary = pd.DataFrame({
        "Task": ["Rain Tomorrow (Classification)", "Temperature Tomorrow (Regression)"],
        "Model": ["SVM", "Linear Regression"],
        "Metric": [f"Accuracy: {models['clf_accuracy']*100:.2f}%",
                   f"RMSE: {models['reg_rmse']:.2f}°C · R²: {models['reg_r2']:.2f}"]
    })
    st.table(summary)
