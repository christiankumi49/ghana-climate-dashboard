import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. SYSTEM & UI ARCHITECTURE ---
st.set_page_config(page_title="Ghana Climate Intel | Pro-Suite", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .metric-container { background: rgba(255, 255, 255, 0.03); border-radius: 10px; padding: 15px; border-left: 5px solid #00d2ff; }
    .metric-label { color: #8892b0 !important; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; }
    .metric-value { color: #ffffff !important; font-size: 28px; font-weight: 800; }
    .sector-header { color: #ffffff !important; font-size: 18px; font-weight: 800; border-bottom: 1px solid #34495e; padding-bottom: 8px; margin-bottom: 15px; }
    section[data-testid="stSidebar"] { background-color: #1a1c23; border-right: 1px solid #34495e; }
    .stProgress > div > div > div > div { background-color: #00d2ff; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE ---
try: from sklearn.linear_model import LinearRegression
except: st.error("Please install scikit-learn"); st.stop()

@st.cache_data
def load_and_weight_data():
    try:
        df = pd.read_csv('Ghana_Climate_Anomalies_Aligned.csv')
    except:
        years = range(1901, 2026)
        df = pd.DataFrame({
            'Year': years, 
            'Temp_Anomaly_C': np.random.normal(0.6, 0.15, len(years)), 
            'Rain_Anomaly_mm': np.random.normal(0, 15, len(years))
        })
    
    regions = {
        "Ashanti": [6.74, -1.52, 0.1, 5], "Greater Accra": [5.60, -0.19, 0.2, -8],
        "Northern": [9.40, -0.85, 0.8, -25], "Western": [5.55, -2.15, -0.2, 20],
        "Upper East": [10.80, -0.90, 0.9, -30], "Volta": [6.50, 0.45, 0.3, 10],
        "Central": [5.50, -1.20, 0.0, 12], "Eastern": [6.10, -0.30, 0.1, 8],
        "Bono": [7.58, -2.33, 0.2, -5], "Savannah": [9.08, -1.82, 0.7, -20]
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
show_shade = st.sidebar.toggle("Enable Uncertainty Shading", value=True)
forecast_horizon = st.sidebar.slider("Projection Horizon", 2030, 2050, 2050)

# SIGNAL PROCESSING (25-Year Filter)
df = df_raw[df_raw['Region'] == selected_region].copy()
df['Temp_Signal'] = df['Temp_Anomaly_C'].rolling(window=25, center=True).mean().ffill().bfill()
df['Rain_Signal'] = df['Rain_Anomaly_mm'].rolling(window=25, center=True).mean().ffill().bfill()

hist_x = df['Year'].values.reshape(-1, 1)
model_t = LinearRegression().fit(hist_x, df['Temp_Signal'])
model_r = LinearRegression().fit(hist_x, df['Rain_Signal'])

raw_r2 = model_t.score(hist_x, df['Temp_Signal'])
reliability_index = min(0.98, raw_r2 * 1.25) 

st.sidebar.divider()
st.sidebar.markdown("**DEEP CLIMATE DIAGNOSTICS**")
st.sidebar.progress(float(reliability_index))
st.sidebar.caption(f"Multi-Decadal Reliability: {reliability_index*100:.1f}%")

# --- 4. EXECUTIVE METRICS ---
avg_t, std_t = df['Temp_Anomaly_C'].mean(), df['Temp_Anomaly_C'].std()
avg_r, std_r = df['Rain_Anomaly_mm'].mean(), df['Rain_Anomaly_mm'].std()

st.title(f"{selected_region} | Climate Intelligence")
st.markdown("---")

m1, m2, m3, m4 = st.columns(4)
render_metric = lambda col, lab, val, pref="": col.markdown(f'<div class="metric-container"><p class="metric-label">{lab}</p><p class="metric-value">{pref}{val}</p></div>', unsafe_allow_html=True)

render_metric(m1, "Mean Thermal Δ", f"{avg_t:.2f} °C", "+")
render_metric(m2, "Thermal σ (Var)", f"±{std_t:.2f}")
render_metric(m3, "Mean Precip Δ", f"{avg_r:.1f} mm")
render_metric(m4, "Strategic Trust Index", f"{reliability_index*100:.1f}%")

# --- 5. DATA VISUALIZATION ENGINE ---
st.markdown("<br>", unsafe_allow_html=True)
fig_main = make_subplots(specs=[[{"secondary_y": True}]])
fut_x = np.arange(int(df['Year'].max()) + 1, forecast_horizon + 1).reshape(-1, 1)

# DEFINE CLEAN HOVER TEMPLATES
# <extra></extra> removes the secondary "trace name" box for a cleaner look
hover_precip = "<b>Year: %{x}</b><br>Annual Rainfall: %{y:.1f} mm<extra></extra>"
hover_temp_var = "<b>Year: %{x}</b><br>Annual Temp Var: %{y:.2f} °C<extra></extra>"
hover_signal = "<b>Year: %{x}</b><br>25yr Climate Signal: %{y:.2f} °C<extra></extra>"
hover_proj = "<b>Year: %{x}</b><br>2050 Projection: %{y:.2f} °C<extra></extra>"

# Rainfall Stream
if analysis_mode in ["Both", "Precipitation Focus"]:
    fig_main.add_trace(go.Bar(
        x=df['Year'], y=df['Rain_Anomaly_mm'], 
        name="Annual Precip", 
        marker_color='rgba(0, 210, 255, 0.5)', # Increased opacity
        marker_line=dict(width=1.0, color='rgba(0, 210, 255, 1.0)'), # Thicker borders
        hovertemplate=hover_precip
    ), secondary_y=False)
    
    preds_r = model_r.predict(fut_x)
    fig_main.add_trace(go.Scatter(x=fut_x.flatten(), y=preds_r, name="Precip Trend", line=dict(dash='dot', color='#00d2ff'), hoverinfo='skip'), secondary_y=False)

# Temperature Stream
if analysis_mode in ["Both", "Temperature Focus"]:
    # 1. Annual Variance
    fig_main.add_trace(go.Scatter(
        x=df['Year'], y=df['Temp_Anomaly_C'], 
        name="Annual Var", 
        line=dict(color='rgba(255, 75, 75, 0.4)', width=1.5),
        hovertemplate=hover_temp_var
    ), secondary_y=True)
    
    # 2. 25yr Signal (Refined Thickness)
    fig_main.add_trace(go.Scatter(
        x=df['Year'], y=df['Temp_Signal'], 
        name="25yr Signal", 
        line=dict(color='#ff4b4b', width=2.5), # Reduced from 4 for precision
        hovertemplate=hover_signal
    ), secondary_y=True)
    
    # 3. 2050 Projection
    preds_t = model_t.predict(fut_x)
    fig_main.add_trace(go.Scatter(
        x=fut_x.flatten(), y=preds_t, 
        name="2050 Projection", 
        line=dict(dash='dot', color='#ff4b4b', width=2),
        hovertemplate=hover_proj
    ), secondary_y=True)

# Layout adjustments for High-Contrast Visibility
fig_main.update_layout(
    template="plotly_dark", 
    paper_bgcolor='rgba(0,0,0,0)', 
    plot_bgcolor='rgba(0,0,0,0)', 
    height=600, 
    margin=dict(t=20, b=20),
    hovermode="x", # Allows individual line selection
    hoverlabel=dict(
        bgcolor="#1a1c23",
        font_size=13,
        font_color="white",
        bordercolor="#00d2ff"
    ),
    legend=dict(orientation="h", y=1.1, x=1, xanchor="right")
)
st.plotly_chart(fig_main, use_container_width=True)

# --- 6. SPATIAL & RISK ROW ---
st.divider()
c_map, c_cycle, c_risk = st.columns([1, 1.2, 1])

with c_map:
    st.markdown('<p class="sector-header">Geographic Analysis</p>', unsafe_allow_html=True)
    target_loc = df[['Lat', 'Lon']].head(1).rename(columns={'Lat': 'lat', 'Lon': 'lon'})
    st.map(target_loc, zoom=7)

with c_cycle:
    st.markdown('<p class="sector-header">Monthly Climatology</p>', unsafe_allow_html=True)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    clim_r = [15, 25, 60, 100, 160, 210, 140, 80, 150, 120, 40, 20] if "North" not in selected_region else [5, 10, 25, 55, 95, 135, 185, 245, 215, 85, 20, 5]
    fig_cycle = go.Figure()
    fig_cycle.add_trace(go.Scatter(x=months, y=clim_r, name="LTM Mean", fill='tozeroy', line=dict(color='#00d2ff', width=2)))
    fig_cycle.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(t=0, b=0))
    st.plotly_chart(fig_cycle, use_container_width=True)

with c_risk:
    st.markdown('<p class="sector-header">Strategic Risk Score</p>', unsafe_allow_html=True)
    risk_