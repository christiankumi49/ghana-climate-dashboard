import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Safety check for the ML engine
try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    st.info("⚙️ Initializing Meteorological Engine... Please refresh in 30 seconds.")
    st.stop()

st.set_page_config(page_title="Ghana Climate Intel", layout="wide")

# Predictive Engine
def calculate_trend(df_input):
    X = df_input['Year'].values.reshape(-1, 1)
    model_t = LinearRegression().fit(X, df_input['Temp_Anomaly_C'].values)
    model_r = LinearRegression().fit(X, df_input['Rain_Anomaly_mm'].values)
    f_yrs = np.arange(df_input['Year'].max() + 1, 2041).reshape(-1, 1)
    return f_yrs.flatten(), model_t.predict(f_yrs), model_r.predict(f_yrs)

# Load Data
df = pd.read_csv('Ghana_Climate_Anomalies_Aligned.csv')

# Sidebar & Map Coords
ghana_regions = {"National Average": [7.9, -1.0, 6], "Ashanti": [6.7, -1.5, 9]} # Add others as needed
sel_region = st.sidebar.selectbox("Region", list(ghana_regions.keys()))
coords = ghana_regions[sel_region]

st.title(f"🌍 {sel_region} Satellite Observation")

# The Professional Map
fig_map = go.Figure(go.Scattermapbox(lat=[coords[0]], lon=[coords[1]], mode='markers', marker=dict(size=15, color='red')))
fig_map.update_layout(
    mapbox=dict(style="open-street-map", center=dict(lat=coords[0], lon=coords[1]), zoom=coords[2]),
    height=500, margin={"r":0,"t":0,"l":0,"b":0}
)
st.plotly_chart(fig_map, use_container_width=True)

# Charts & Forecast
f_yrs, f_t, f_r = calculate_trend(df)
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=df['Year'], y=df['Rain_Anomaly_mm'], name="Rain"), secondary_y=False)
fig.add_trace(go.Scatter(x=df['Year'], y=df['Temp_Anomaly_C'], name="Temp"), secondary_y=True)
fig.add_trace(go.Scatter(x=f_yrs, y=f_r, name="Rain Proj", line=dict(dash='dot')), secondary_y=False)
fig.add_trace(go.Scatter(x=f_yrs, y=f_t, name="Temp Proj", line=dict(dash='dot')), secondary_y=True)
st.plotly_chart(fig, use_container_width=True)