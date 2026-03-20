import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

# 1. Page Configuration
st.set_page_config(page_title="Ghana Climate Intelligence Dashboard", layout="wide")

# --- PREDICTIVE ENGINE ---
def calculate_trend(df_input, target_year=2040):
    """Uses Linear Regression to project trends to 2040."""
    X = df_input['Year'].values.reshape(-1, 1)
    y_temp = df_input['Temp_Anomaly_C'].values
    y_rain = df_input['Rain_Anomaly_mm'].values
    
    # Train Models
    model_t = LinearRegression().fit(X, y_temp)
    model_r = LinearRegression().fit(X, y_rain)
    
    # Predict future
    future_years = np.arange(df_input['Year'].max() + 1, target_year + 1).reshape(-1, 1)
    pred_t = model_t.predict(future_years)
    pred_r = model_r.predict(future_years)
    
    return future_years.flatten(), pred_t, pred_r

# 2. Load the Processed Data
@st.cache_data
def load_data():
    # Ensure this filename matches your CSV
    df = pd.read_csv('Ghana_Climate_Anomalies_Aligned.csv')
    return df

df = load_data()

# 3. Interactive Control Panel (Sidebar)
st.sidebar.header("🕹️ Dashboard Controls")

ghana_regions = {
    "National Average": [7.9465, -1.0232, 5.5],
    "Ashanti": [6.75, -1.5, 8.5], "Greater Accra": [5.81, 0.0, 10.0],
    "Northern": [9.4, -0.8, 7.5], "Western": [5.9, -2.1, 8.5],
    "Eastern": [6.3, -0.3, 8.5], "Central": [5.5, -1.2, 9.0],
    "Volta": [6.6, 0.4, 8.5], "Upper East": [10.8, -0.8, 9.5],
    "Upper West": [10.3, -2.1, 9.5], "Bono": [7.5, -2.5, 9.0],
    "Bono East": [7.8, -1.0, 9.0], "Ahafo": [7.0, -2.3, 9.5],
    "Savannah": [9.1, -1.8, 8.5], "North East": [10.4, -0.2, 9.5],
    "Oti": [8.1, 0.3, 9.0], "Western North": [6.3, -2.8, 9.0]
}

selected_region = st.sidebar.selectbox("Select Study Area", options=list(ghana_regions.keys()))
target_var = st.sidebar.selectbox("Climate Variable", options=["Rainfall", "Temperature", "Both"], index=2)

min_year, max_year = int(df['Year'].min()), int(df['Year'].max())
selected_years = st.sidebar.slider("Historical Range", min_year, max_year, (1980, 2020))

st.sidebar.divider()
enable_forecast = st.sidebar.toggle("Enable 2040 Forecast", value=False)

# 4. Data Filtering and Dynamic Math
filtered_df = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]

# DYNAMIC METRIC CALCULATIONS
avg_rain_anomaly = filtered_df['Rain_Anomaly_mm'].mean()
avg_temp_anomaly = filtered_df['Temp_Anomaly_C'].mean()
peak_temp = filtered_df['Temp_Anomaly_C'].max()

# 5. Header Section
st.title(f"🇬🇭 {selected_region} Climate Intelligence")
st.markdown(f"Real-time analysis and 2040 statistical projections for **{selected_region}**.")

# --- DYNAMIC EXECUTIVE SUMMARY ---
st.subheader("Executive Climate Summary")
col1, col2, col3 = st.columns(3)

if enable_forecast:
    f_yrs, f_t, f_r = calculate_trend(df)
    col1.metric("Projected 2040 Rain Anomaly", f"{f_r[-1]:.2f} mm", f"{'Wetter' if f_r[-1] > 0 else 'Drier'}")
    col2.metric("Projected 2040 Temp Anomaly", f"+{f_t[-1]:.2f} °C", "Rising Trend")
    col3.metric("Model Logic", "Linear Regression", "Scikit-Learn")
else:
    col1.metric("Avg Rain Anomaly (Selected)", f"{avg_rain_anomaly:.2f} mm")
    col2.metric("Avg Temp Anomaly (Selected)", f"+{avg_temp_anomaly:.2f} °C", delta="Historical")
    col3.metric("Max Recorded Anomaly", f"+{peak_temp:.2f} °C", delta_color="inverse")

st.divider()

# 6. Interactive Dual-Axis Chart
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Historical Data
if target_var in ["Rainfall", "Both"]:
    fig.add_trace(go.Bar(x=filtered_df['Year'], y=filtered_df['Rain_Anomaly_mm'], 
                         name="Historical Rain", marker_color='royalblue', opacity=0.7), secondary_y=False)

if target_var in ["Temperature", "Both"]:
    fig.add_trace(go.Scatter(x=filtered_df['Year'], y=filtered_df['Temp_Anomaly_C'], 
                             name="Historical Temp", line=dict(color='crimson', width=2.5)), secondary_y=True)

# Forecast Data
if enable_forecast:
    f_yrs, f_t, f_r = calculate_trend(df)
    if target_var in ["Rainfall", "Both"]:
        fig.add_trace(go.Bar(x=f_yrs, y=f_r, name="Proj. Rain", marker_color='lightblue', opacity=0.4), secondary_y=False)
    if target_var in ["Temperature", "Both"]:
        fig.add_trace(go.Scatter(x=f_yrs, y=f_t, name="Proj. Trend", line=dict(color='crimson', width=2, dash='dot')), secondary_y=True)
    fig.update_xaxes(range=[selected_years[0], 2040])

fig.update_layout(
    title=f"<b>Integrated Anomalies: {selected_region} ({selected_years[0]}-{'2040' if enable_forecast else selected_years[1]})</b>",
    xaxis_title="Year", hovermode="x unified", template="plotly_white",
    legend=dict(x=0, y=1.1, orientation="h")
)

fig.update_yaxes(title_text="Rainfall (mm)", secondary_y=False, color="blue")
fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True, color="red")

st.plotly_chart(fig, use_container_width=True)

# 7. Spatial Analysis & Insights
st.divider()
st.header(f"🌍 {selected_region} Geographic Analysis")
map_col, stats_col = st.columns([2, 1])
region_data = ghana_regions[selected_region]

with map_col:
    fig_map = go.Figure(go.Scattermapbox(
        lat=[region_data[0]], lon=[region_data[1]],
        mode='markers', marker=go.scattermapbox.Marker(size=20, color='gold', symbol='star'),
        text=[selected_region]
    ))
    fig_map.update_layout(
        mapbox_style="open-street-map", mapbox_center_lat=region_data[0], mapbox_center_lon=region_data[1],
        mapbox_zoom=region_data[2], height=450, margin={"r":0,"t":0,"l":0,"b":0}
    )
    st.plotly_chart(fig_map, use_container_width=True)

with stats_col:
    st.subheader("Quick Insights")
    st.write(f"**Focus Area:** {selected_region}")
    st.write(f"**Selected Data Points:** {len(filtered_df)} years")
    
    if avg_temp_anomaly > 0.5:
        st.warning(f"This region shows significant warming of {avg_temp_anomaly:.2f}°C during the selected period.")
    else:
        st.info("Temperature deviations are within moderate historical bounds for this selection.")

    st.download_button("📥 Download Region Data", filtered_df.to_csv(index=False).encode('utf-8'), 
                       f"{selected_region}_climate_report.csv", "text/csv")