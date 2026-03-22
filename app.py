import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- METEOROLOGICAL ENGINE CHECK ---
try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    st.warning("⚙️ System Update: Initializing Meteorological Engine...")
    st.stop()

st.set_page_config(page_title="Ghana Climate Intelligence", layout="wide")

# --- PREDICTIVE ENGINE ---
def calculate_trend(df_input, target_year=2040):
    X = df_input['Year'].values.reshape(-1, 1)
    model_t = LinearRegression().fit(X, df_input['Temp_Anomaly_C'].values)
    model_r = LinearRegression().fit(X, df_input['Rain_Anomaly_mm'].values)
    future_years = np.arange(df_input['Year'].max() + 1, target_year + 1).reshape(-1, 1)
    return future_years.flatten(), model_t.predict(future_years), model_r.predict(future_years)

# 2. Load Dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Ghana_Climate_Anomalies_Aligned.csv')
        return df
    except:
        return pd.DataFrame()

df = load_data()

# 3. Sidebar
ghana_regions = {
    "National Average": [7.9465, -1.0232, 6],
    "Ashanti": [6.75, -1.5, 9], "Greater Accra": [5.81, 0.0, 10],
    "Northern": [9.4, -0.8, 8], "Western": [5.9, -2.1, 9],
    "Eastern": [6.3, -0.3, 9], "Central": [5.5, -1.2, 9],
    "Volta": [6.6, 0.4, 9], "Upper East": [10.8, -0.8, 10],
    "Upper West": [10.3, -2.1, 10], "Bono": [7.5, -2.5, 9],
    "Bono East": [7.8, -1.0, 9], "Ahafo": [7.0, -2.3, 10],
    "Savannah": [9.1, -1.8, 9], "North East": [10.4, -0.2, 10],
    "Oti": [8.1, 0.3, 9], "Western North": [6.3, -2.8, 9]
}

selected_region = st.sidebar.selectbox("Select Study Area", options=list(ghana_regions.keys()))
enable_forecast = st.sidebar.toggle("Enable 2040 Forecast", value=True)

# 4. Main Interface
st.title(f"🇬🇭 {selected_region} Meteorological Analysis")

if not df.empty:
    # --- METRICS ---
    c1, c2 = st.columns(2)
    c1.metric("Avg Rain Anomaly", f"{df['Rain_Anomaly_mm'].mean():.2f} mm")
    c2.metric("Avg Temp Anomaly", f"+{df['Temp_Anomaly_C'].mean():.2f} °C")

    # --- CHART ---
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df['Year'], y=df['Rain_Anomaly_mm'], name="Rain", marker_color='#3498db'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Year'], y=df['Temp_Anomaly_C'], name="Temp", line=dict(color='#e74c3c')), secondary_y=True)
    
    if enable_forecast:
        f_yrs, f_t, f_r = calculate_trend(df)
        fig.add_trace(go.Scatter(x=f_yrs, y=f_r, name="Rain Proj.", line=dict(dash='dot', color='#2980b9')), secondary_y=False)
        fig.add_trace(go.Scatter(x=f_yrs, y=f_t, name="Temp Proj.", line=dict(dash='dot', color='#f39c12')), secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    # --- THE SHARP MAP FIX ---
    st.subheader("🛰️ High-Resolution Geospatial Context")
    coords = ghana_regions[selected_region]

    fig_map = go.Figure(go.Scattermapbox(
        lat=[coords[0]], lon=[coords[1]],
        mode='markers',
        marker=go.scattermapbox.Marker(size=20, color='red'),
        text=[selected_region]
    ))

    fig_map.update_layout(
        mapbox=dict(
            style="carto-positron", # Clean, sharp, and very clear names
            center=dict(lat=coords[0], lon=coords[1]),
            zoom=coords[2]
        ),
        height=600, margin={"r":0,"t":0,"l":0,"b":0}
    )
    st.plotly_chart(fig_map, use_container_width=True)