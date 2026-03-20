import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- SAFETY CHECK FOR METEOROLOGICAL ENGINE ---
try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    st.warning("⚙️ System Update in Progress: Installing Meteorological Engine (scikit-learn)...")
    st.info("Please wait 30 seconds and refresh the page.")
    st.stop()

# 1. Page Configuration
st.set_page_config(page_title="Ghana Climate Intelligence", layout="wide")

# --- PREDICTIVE ENGINE ---
def calculate_trend(df_input, target_year=2040):
    X = df_input['Year'].values.reshape(-1, 1)
    y_temp = df_input['Temp_Anomaly_C'].values
    y_rain = df_input['Rain_Anomaly_mm'].values
    model_t = LinearRegression().fit(X, y_temp)
    model_r = LinearRegression().fit(X, y_rain)
    future_years = np.arange(df_input['Year'].max() + 1, target_year + 1).reshape(-1, 1)
    pred_t = model_t.predict(future_years)
    pred_r = model_r.predict(future_years)
    return future_years.flatten(), pred_t, pred_r

# 2. Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Ghana_Climate_Anomalies_Aligned.csv')
        return df
    except Exception as e:
        st.error(f"Data Connection Error: {e}")
        return pd.DataFrame()

df = load_data()

# 3. Sidebar & Controls
st.sidebar.header("🕹️ Dashboard Controls")
ghana_regions = {
    "National Average": [7.9465, -1.0232, 5.5],
    "Ashanti": [6.75, -1.5, 8.5], "Greater Accra": [5.81, 0.0, 10.0],
    "Northern": [9.4, -0.8, 7.5], "Western": [5.9, -2.1, 8.5],
    "Eastern": [6.3, -0.3, 8.5], "Central": [5.5, -1.2, 9.0],
    "Volta": [6.6, 0.4, 8.5], "Upper East": [10.8, -0.8, 9.5],
    "Upper West": [10.3, -2.1, 9.5]
}

selected_region = st.sidebar.selectbox("Select Study Area", options=list(ghana_regions.keys()))
target_var = st.sidebar.selectbox("Climate Variable", options=["Rainfall", "Temperature", "Both"], index=2)
enable_forecast = st.sidebar.toggle("Enable 2040 Forecast", value=True)

# 4. Main Interface
st.title(f"🇬🇭 {selected_region} Meteorological Intelligence")

if not df.empty:
    # --- METRICS ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Rain Anomaly", f"{df['Rain_Anomaly_mm'].mean():.2f} mm")
    col2.metric("Avg Temp Anomaly", f"+{df['Temp_Anomaly_C'].mean():.2f} °C")
    col3.metric("Map Layer", "Satellite-Streets")

    st.divider()

    # --- CHARTS ---
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df['Year'], y=df['Rain_Anomaly_mm'], name="Rain Anomaly", marker_color='royalblue'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Year'], y=df['Temp_Anomaly_C'], name="Temp Anomaly", line=dict(color='crimson')), secondary_y=True)
    
    if enable_forecast:
        f_yrs, f_t, f_r = calculate_trend(df)
        fig.add_trace(go.Scatter(x=f_yrs, y=f_r, name="Rain Proj.", line=dict(dash='dot', color='blue')), secondary_y=False)
        fig.add_trace(go.Scatter(x=f_yrs, y=f_t, name="Temp Proj.", line=dict(dash='dot', color='orange')), secondary_y=True)
    
    fig.update_layout(height=500, template="plotly_white", margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # --- SATELLITE MAP ---
    st.subheader("🌍 Satellite Observation")
    coords = ghana_regions[selected_region]
    fig_map = go.Figure(go.Scattermapbox(
        lat=[coords[0]], lon=[coords[1]],
        mode='markers',
        marker=go.scattermapbox.Marker(size=22, color='gold', symbol='circle'),
        text=[selected_region],
    ))

    fig_map.update_layout(
        mapbox=dict(
            style="satellite-streets", 
            center=dict(lat=coords[0], lon=coords[1]),
            zoom=coords[2]
        ),
        height=600, margin={"r":0,"t":0,"l":0,"b":0}
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # 5. Export
    st.sidebar.divider()
    st.sidebar.download_button("📥 Export CSV", df.to_csv(index=False).encode('utf-8'), "ghana_climate.csv")