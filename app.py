import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

# 1. Page Configuration
st.set_page_config(page_title="Ghana Climate Intelligence Dashboard", layout="wide", page_icon="🇬🇭")

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

# 2. Data Loading
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Ghana_Climate_Anomalies_Aligned.csv')
        return df
    except:
        st.error("⚠️ Data file not found. Please ensure 'Ghana_Climate_Anomalies_Aligned.csv' is in your repository.")
        return pd.DataFrame()

df = load_data()

# 3. Sidebar Navigation & Export
st.sidebar.title("🇬🇭 Navigation")
st.sidebar.markdown("---")

ghana_regions = {
    "National Average": [7.9465, -1.0232, 6],
    "Ashanti": [6.75, -1.5, 8], "Greater Accra": [5.81, 0.0, 10],
    "Northern": [9.4, -0.8, 7], "Western": [5.9, -2.1, 8],
    "Eastern": [6.3, -0.3, 8], "Central": [5.5, -1.2, 9],
    "Volta": [6.6, 0.4, 8], "Upper East": [10.8, -0.8, 9],
    "Upper West": [10.3, -2.1, 9]
}

selected_region = st.sidebar.selectbox("📍 Select Region", options=list(ghana_regions.keys()))
enable_forecast = st.sidebar.toggle("🔮 Enable 2040 Forecast", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Data Export")
if not df.empty:
    csv = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("Download Full CSV", data=csv, file_name='ghana_climate_report.csv', mime='text/csv')

# 4. Main Dashboard Header
st.title(f"Climate Intelligence: {selected_region}")
st.markdown(f"**Historical Analysis and Predictive Modeling for Ghana's Climate Trends**")

# 5. Professional Metrics
if not df.empty:
    latest_temp = df['Temp_Anomaly_C'].iloc[-1]
    latest_rain = df['Rain_Anomaly_mm'].iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Temp Anomaly", f"{latest_temp:.2f} °C", delta=f"{latest_temp - df['Temp_Anomaly_C'].iloc[-2]:.2f}")
    col2.metric("Latest Rain Anomaly", f"{latest_rain:.1f} mm", delta=f"{latest_rain - df['Rain_Anomaly_mm'].iloc[-2]:.1f}")
    col3.metric("Data Range", f"{int(df['Year'].min())} - {int(df['Year'].max())}")

# 6. Interactive Map
coords = ghana_regions[selected_region]
map_data = pd.DataFrame({'lat': [coords[0]], 'lon': [coords[1]]})
st.subheader("Regional Context")
st.map(map_data, zoom=coords[2])

# 7. Advanced Analytics Chart
st.subheader("Climate Trend Analysis")
if not df.empty:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Historical Bars & Lines
    fig.add_trace(go.Bar(x=df['Year'], y=df['Rain_Anomaly_mm'], name="Rain (Observed)", marker_color='#3498db', opacity=0.7), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Year'], y=df['Temp_Anomaly_C'], name="Temp (Observed)", line=dict(color='#e74c3c', width=3)), secondary_y=True)

    if enable_forecast:
        f_years, f_temp, f_rain = calculate_trend(df)
        fig.add_trace(go.Scatter(x=f_years, y=f_rain, name="Rain (2040 Projection)", line=dict(color='#2980b9', dash='dot')), secondary_y=False)
        fig.add_trace(go.Scatter(x=f_years, y=f_temp, name="Temp (2040 Projection)", line=dict(color='#c0392b', dash='dot')), secondary_y=True)

    fig.update_layout(height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)