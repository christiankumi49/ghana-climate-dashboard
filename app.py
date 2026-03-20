import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

# 1. Page Configuration
st.set_page_config(page_title="Ghana Climate Dashboard", layout="wide")

# --- PREDICTIVE ENGINE ---
def calculate_trend(df_input, target_year=2040):
    X = df_input['Year'].values.reshape(-1, 1)
    y_temp = df_input['Temp_Anomaly_C'].values
    y_rain = df_input['Rain_Anomaly_mm'].values
    model_t = LinearRegression().fit(X, y_temp)
    model_r = LinearRegression().fit(X, y_rain)
    last_year = int(df_input['Year'].max())
    future_years = np.arange(last_year + 1, target_year + 1).reshape(-1, 1)
    pred_t = model_t.predict(future_years)
    pred_r = model_r.predict(future_years)
    return future_years.flatten(), pred_t, pred_r

# 2. Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Ghana_Climate_Anomalies_Aligned.csv')
        return df
    except:
        st.error("Dataset not found.")
        return pd.DataFrame()

df = load_data()

# 3. Sidebar Controls (All your original controls are back)
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

min_year, max_year = int(df['Year'].min()), int(df['Year'].max())
selected_years = st.sidebar.slider("Historical Range", min_year, max_year, (1980, 2020))

enable_forecast = st.sidebar.toggle("Enable 2040 Forecast", value=True)

# --- DOWNLOAD BUTTON (At the bottom of sidebar) ---
st.sidebar.divider()
csv_data = df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("📥 Download Data (CSV)", data=csv_data, file_name='ghana_climate_data.csv')

# 4. Main Title
st.title(f"🇬🇭 {selected_region} Climate Intelligence")

# 5. Graph FIRST (As requested)
filtered_df = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]

if not filtered_df.empty:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    if target_var in ["Rainfall", "Both"]:
        fig.add_trace(go.Bar(x=filtered_df['Year'], y=filtered_df['Rain_Anomaly_mm'], name="Rain Anomaly"), secondary_y=False)
    
    if target_var in ["Temperature", "Both"]:
        fig.add_trace(go.Scatter(x=filtered_df['Year'], y=filtered_df['Temp_Anomaly_C'], name="Temp Anomaly", line=dict(color='red')), secondary_y=True)
    
    if enable_forecast:
        f_years, f_temp, f_rain = calculate_trend(df)
        if target_var in ["Rainfall", "Both"]:
            fig.add_trace(go.Scatter(x=f_years, y=f_rain, name="Rain Forecast", line=dict(dash='dash')), secondary_y=False)
        if target_var in ["Temperature", "Both"]:
            fig.add_trace(go.Scatter(x=f_years, y=f_temp, name="Temp Forecast", line=dict(color='orange', dash='dash')), secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

# 6. Map SECOND
st.subheader("Location Map")
coords = ghana_regions[selected_region]
map_df = pd.DataFrame({'lat': [coords[0]], 'lon': [coords[1]]})
st.map(map_df, zoom=coords[2])