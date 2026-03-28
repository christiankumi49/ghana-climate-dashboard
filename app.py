import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

# --- 1. PRO-SUITE UI ARCHITECTURE ---
st.set_page_config(page_title="GCI Pro-Suite | CAT Risk Intel", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0b0e14; }
    .glass-card {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(15px);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(0, 210, 255, 0.1);
        margin-bottom: 20px;
    }
    .metric-label { color: #8892b0; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; }
    .metric-value { color: #00d2ff; font-size: 28px; font-weight: 800; }
    .sector-header { color: #ffffff; font-size: 18px; font-weight: 800; border-bottom: 1px solid #34495e; padding-bottom: 8px; margin-bottom: 15px; }
    section[data-testid="stSidebar"] { background-color: #1a1c23; border-right: 1px solid #34495e; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CORE DATA ENGINE ---
@st.cache_data
def load_and_weight_data():
    try:
        df = pd.read_csv('Ghana_Climate_Anomalies_Aligned.csv')
    except:
        years = np.arange(1901, 2027)
        df = pd.DataFrame({
            'Year': years, 
            'Temp_Anomaly_C': np.random.normal(0.6, 0.15, len(years)) + (years-1901)*0.004, 
            'Rain_Anomaly_mm': np.random.normal(0, 15, len(years))
        })
    
    # YOUR VARIABLES (16 Regions Mapping)
    regions = {
        "Ashanti": [6.74, -1.52, 0.1, 5], "Greater Accra": [5.60, -0.19, 0.2, -8],
        "Northern": [9.40, -0.85, 0.8, -25], "Western": [5.55, -2.15, -0.2, 20],
        "Upper East": [10.80, -0.90, 0.9, -30], "Volta": [6.50, 0.45, 0.3, 10],
        "Central": [5.50, -1.20, 0.0, 12], "Eastern": [6.10, -0.30, 0.1, 8],
        "Bono": [7.58, -2.33, 0.2, -5], "Savannah": [9.08, -1.82, 0.7, -20],
        "Upper West": [10.20, -2.10, 0.85, -28], "Ahafo": [7.0, -2.4, 0.15, 2],
        "Bono East": [7.7, -1.0, 0.3, -10], "North East": [10.5, -0.5, 0.88, -28],
        "Oti": [8.2, 0.3, 0.4, 5], "Western North": [6.3, -2.8, -0.1, 15]
    }
    
    all_dfs = []
    for reg, (lat, lon, t_off, r_off) in regions.items():
        temp = df.copy().assign(Region=reg, Lat=lat, Lon=lon)
        temp['Temp_Anomaly_C'] += t_off
        temp['Rain_Anomaly_mm'] += r_off
        all_dfs.append(temp)
    return pd.concat(all_dfs)

df_raw = load_and_weight_data()

# --- 3. COMMAND CENTER ---
st.sidebar.title("💎 COMMAND CENTER")
selected_region = st.sidebar.selectbox("Geographic Focus", options=sorted(df_raw['Region'].unique()))
analysis_mode = st.sidebar.radio("Primary Stream", ["Both", "Temperature Focus", "Precipitation Focus"])

st.sidebar.divider()
st.sidebar.markdown("**FORECAST SETTINGS**")
predictive_mode = st.sidebar.toggle("Enable Predictive Analytics", value=True)
show_shade = st.sidebar.toggle("Uncertainty Shading", value=True)
forecast_horizon = st.sidebar.slider("Projection Horizon", 2026, 2060, 2050)

# SIGNAL PROCESSING
df = df_raw[df_raw['Region'] == selected_region].copy()
df['Temp_Signal'] = df['Temp_Anomaly_C'].rolling(window=25, center=True).mean().ffill().bfill()
df['Rain_Signal'] = df['Rain_Anomaly_mm'].rolling(window=25, center=True).mean().ffill().bfill()

# --- 4. EXECUTIVE METRICS ---
avg_t, avg_r = df['Temp_Anomaly_C'].mean(), df['Rain_Anomaly_mm'].mean()
st.title(f"{selected_region} | Climate Intelligence")
st.markdown("---")

m1, m2, m3, m4 = st.columns(4)
render_metric = lambda col, lab, val: col.markdown(f'<div class="glass-card"><p class="metric-label">{lab}</p><p class="metric-value">{val}</p></div>', unsafe_allow_html=True)

render_metric(m1, "Mean Thermal Δ", f"+{avg_t:.2f} °C")
render_metric(m2, "Mean Precip Δ", f"{avg_r:.1f} mm")
render_metric(m3, "Reliability Index", "74.0%")  # RESTORED AS REQUESTED
render_metric(m4, "Data Freshness", "Q1 2026")

# --- 5. MAIN VISUALIZATION (CLEANED) ---
fig_main = make_subplots(specs=[[{"secondary_y": True}]])
fut_x = np.arange(2026, forecast_horizon + 1).reshape(-1, 1)
hist_x = df['Year'].values.reshape(-1, 1)

# Rainfall (Primary Y)
if analysis_mode in ["Both", "Precipitation Focus"]:
    fig_main.add_trace(go.Bar(x=df['Year'], y=df['Rain_Anomaly_mm'], name="Annual Rain", marker_color='rgba(0, 210, 255, 0.4)'), secondary_y=False)
    if predictive_mode:
        model_r = LinearRegression().fit(hist_x, df['Rain_Signal'])
        preds_r = model_r.predict(fut_x)
        if show_shade:
            fig_main.add_trace(go.Scatter(x=np.concatenate([fut_x.flatten(), fut_x.flatten()[::-1]]), y=np.concatenate([preds_r + 12, (preds_r - 12)[::-1]]), fill='toself', fillcolor='rgba(0, 210, 255, 0.05)', line=dict(color='rgba(0,0,0,0)'), name="Rain Uncertainty"), secondary_y=False)
        fig_main.add_trace(go.Scatter(x=fut_x.flatten(), y=preds_r, name="Rain Trend", line=dict(dash='dot', color='#00d2ff')), secondary_y=False)

# Temperature (Secondary Y)
if analysis_mode in ["Both", "Temperature Focus"]:
    fig_main.add_trace(go.Scatter(x=df['Year'], y=df['Temp_Anomaly_C'], name="Annual Temp", line=dict(color='rgba(255, 75, 75, 0.5)', width=2.5)), secondary_y=True)
    fig_main.add_trace(go.Scatter(x=df['Year'], y=df['Temp_Signal'], name="Climate Signal", line=dict(color='#ff4b4b', width=1.8)), secondary_y=True)
    if predictive_mode:
        model_t = LinearRegression().fit(hist_x, df['Temp_Signal'])
        preds_t = model_t.predict(fut_x)
        if show_shade:
            # THINNER SHADING (±0.15 instead of ±0.25)
            fig_main.add_trace(go.Scatter(x=np.concatenate([fut_x.flatten(), fut_x.flatten()[::-1]]), y=np.concatenate([preds_t + 0.15, (preds_t - 0.15)[::-1]]), fill='toself', fillcolor='rgba(255, 75, 75, 0.05)', line=dict(color='rgba(0,0,0,0)'), name="Temp Uncertainty"), secondary_y=True)
        fig_main.add_trace(go.Scatter(x=fut_x.flatten(), y=preds_t, name="AI Projection", line=dict(dash='dot', color='#ff4b4b', width=2)), secondary_y=True)

fig_main.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500, hovermode="x")
st.plotly_chart(fig_main, use_container_width=True)

# --- 6. SPATIAL, CLIMATOLOGY & CAT GAUGE (BOTTOM ROW) ---
st.divider()
c_map, c_cycle, c_risk = st.columns([1, 1.2, 1])

with c_map:
    st.markdown('<p class="sector-header">Geographic Analysis</p>', unsafe_allow_html=True)
    st.map(df[['Lat', 'Lon']].head(1).rename(columns={'Lat': 'lat', 'Lon': 'lon'}), zoom=6)

with c_cycle:
    st.markdown('<p class="sector-header">Monthly Climatology</p>', unsafe_allow_html=True)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    is_north = selected_region in ["Upper East", "Upper West", "Northern", "Savannah", "North East"]
    clim_r = [5, 12, 28, 60, 100, 160, 215, 270, 225, 90, 20, 5] if is_north else [20, 35, 75, 115, 170, 225, 145, 85, 170, 130, 50, 25]
    fig_clim = go.Figure(go.Scatter(x=months, y=clim_r, fill='tozeroy', line=dict(color='#00d2ff', width=3)))
    fig_clim.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=250, margin=dict(t=0, b=0))
    st.plotly_chart(fig_clim, use_container_width=True)

with c_risk:
    st.markdown('<p class="sector-header">CAT Risk Gauge</p>', unsafe_allow_html=True)
    risk_score = min(int((avg_t / 1.1) * 100), 100) if avg_t > 0 else 10
    fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=risk_score, number={'suffix': "%"},
        title={'text': "Exposure Score", 'font': {'size': 14}},
        gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#00d2ff"}, 
               'steps': [{'range': [0, 70], 'color': '#2c3e50'}, {'range': [70, 100], 'color': '#e74c3c'}]}))
    fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=250, margin=dict(t=0, b=0))
    st.plotly_chart(fig_gauge, use_container_width=True)

st.sidebar.download_button("📂 Export Pro Report", df.to_csv(index=False), f"GCI_Report_{selected_region}.csv")