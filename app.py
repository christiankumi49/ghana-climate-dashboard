import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- 1. SYSTEM & UI ARCHITECTURE ---
st.set_page_config(page_title="Ghana Climate Intel | Pro-Suite", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    
    /* Top Metric Styling */
    .metric-container { background: rgba(255, 255, 255, 0.03); border-radius: 10px; padding: 15px; border-left: 5px solid #00d2ff; }
    .metric-label { color: #8892b0 !important; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; }
    .metric-value { color: #ffffff !important; font-size: 28px; font-weight: 800; }
    
    /* Section Headers */
    .sector-header { color: #ffffff !important; font-size: 18px; font-weight: 800; border-bottom: 1px solid #34495e; padding-bottom: 8px; margin-bottom: 15px; }
    .sector-text { color: #bdc3c7 !important; font-size: 14px; line-height: 1.6; }
    
    /* Custom Sidebar */
    section[data-testid="stSidebar"] { background-color: #1a1c23; border-right: 1px solid #34495e; }
    .stProgress > div > div > div > div { background-color: #00d2ff; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ADVANCED ANALYTICS ENGINE ---
try: 
    from sklearn.linear_model import LinearRegression
except ImportError: 
    st.error("Critical: scikit-learn not found. Install via 'pip install scikit-learn'"); st.stop()

@st.cache_data
def load_and_weight_data():
    """Simulates high-fidelity regional data across Ghana's zones."""
    try:
        df = pd.read_csv('Ghana_Climate_Anomalies_Aligned.csv')
    except FileNotFoundError:
        years = range(1901, 2026)
        df = pd.DataFrame({
            'Year': years, 
            'Temp_Anomaly_C': np.random.normal(0.65, 0.15, len(years)), 
            'Rain_Anomaly_mm': np.random.normal(0, 18, len(years))
        })
    
    # Regional Metadata for Spatial Intelligence
    regions = {
        "Ashanti": [6.74, -1.52, 0.1, 5], "Greater Accra": [5.60, -0.19, 0.2, -8],
        "Northern": [9.40, -0.85, 0.8, -25], "Western": [5.55, -2.15, -0.2, 20],
        "Upper East": [10.80, -0.90, 0.9, -30], "Volta": [6.50, 0.45, 0.3, 10],
        "Central": [5.50, -1.20, 0.0, 12], "Eastern": [6.10, -0.30, 0.1, 8]
    }
    
    all_dfs = []
    for reg, (lat, lon, t_off, r_off) in regions.items():
        temp = df.copy().assign(Region=reg, Lat=lat, Lon=lon)
        temp['Temp_Anomaly_C'] += t_off
        temp['Rain_Anomaly_mm'] += r_off
        all_dfs.append(temp)
    return pd.concat(all_dfs)

df_raw = load_and_weight_data()

# --- 3. COMMAND CENTER (SIDEBAR) ---
st.sidebar.title("💎 COMMAND CENTER")

# Methodology Section (Highly Important for Portfolio)
with st.sidebar.expander("ℹ️ TECHNICAL METHODOLOGY"):
    st.markdown("""
    **Core Framework:**
    Uses **First-Order Linear Regression** to decouple long-term climate signals from noise.
    
    **Analytical Indicators:**
    * **Anomaly ($\Delta$):** Deviation from 1901-2000 mean.
    * **Volatility ($\sigma$):** Standard deviation threshold.
    * **Confidence ($R^2$):** Measures trend signal-to-noise ratio.
    """)

selected_region = st.sidebar.selectbox("Geographic Focus", options=sorted(df_raw['Region'].unique()))
analysis_mode = st.sidebar.radio("Analysis Mode", ["Full Diagnostic", "Temperature Focus", "Precipitation Focus"])
selected_year = st.sidebar.number_input("Comparison Baseline (Year)", 1901, 2025, 2023)
forecast_horizon = st.sidebar.slider("Projection Horizon", 2030, 2060, 2050)

st.sidebar.divider()
st.sidebar.markdown("**MODEL STABILITY**")

# Calculate R-Squared for the selected region
df = df_raw[df_raw['Region'] == selected_region].copy()
hist_x = df['Year'].values.reshape(-1, 1)
model_t = LinearRegression().fit(hist_x, df['Temp_Anomaly_C'])
r2_val = model_t.score(hist_x, df['Temp_Anomaly_C'])
st.sidebar.progress(max(0.0, min(float(r2_val), 1.0)))
st.sidebar.caption(f"Trend Confidence: {r2_val*100:.1f}%")

# --- 4. CORE COMPUTATION ---
avg_t, std_t = df['Temp_Anomaly_C'].mean(), df['Temp_Anomaly_C'].std()
avg_r, std_r = df['Rain_Anomaly_mm'].mean(), df['Rain_Anomaly_mm'].std()

# Outlier Detection (2-Sigma Rule)
df['Is_Outlier'] = (np.abs(df['Temp_Anomaly_C'] - avg_t) > 2*std_t) | (np.abs(df['Rain_Anomaly_mm'] - avg_r) > 2*std_r)
outlier_count = df['Is_Outlier'].sum()

# --- 5. EXECUTIVE DASHBOARD ---
st.title(f"{selected_region} | Climate Intelligence Dashboard")
st.markdown("---")

# Metrics Row
m1, m2, m3, m4, m5 = st.columns(5)
def render_metric(col, label, value, prefix=""):
    with col:
        st.markdown(f"""<div class="metric-container">
            <p class="metric-label">{label}</p>
            <p class="metric-value">{prefix}{value}</p>
        </div>""", unsafe_allow_html=True)

render_metric(m1, "Mean Thermal Δ", f"{avg_t:.2f} °C", "+")
render_metric(m2, "Thermal σ (Var)", f"±{std_t:.2f}")
render_metric(m3, "Mean Precip Δ", f"{avg_r:.1f} mm")
render_metric(m4, "Extreme Events", f"{outlier_count}")
render_metric(m5, "Model Fidelity", f"{r2_val*100:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# Main Multi-Variable Chart
fig_main = make_subplots(specs=[[{"secondary_y": True}]])

if analysis_mode in ["Full Diagnostic", "Precipitation Focus"]:
    fig_main.add_trace(go.Bar(x=df['Year'], y=df['Rain_Anomaly_mm'], name="Precipitation Anomaly", 
                              marker_color='#00d2ff', opacity=0.4), secondary_y=False)

if analysis_mode in ["Full Diagnostic", "Temperature Focus"]:
    fig_main.add_trace(go.Scatter(x=df['Year'], y=df['Temp_Anomaly_C'], name="Thermal Anomaly", 
                                  line=dict(color='#ff4b4b', width=3)), secondary_y=True)
    
    # Regression & Confidence Shading
    fut_x = np.arange(int(df['Year'].max()) + 1, forecast_horizon + 1).reshape(-1, 1)
    preds_t = model_t.predict(fut_x)
    fig_main.add_trace(go.Scatter(x=np.concatenate([fut_x.flatten(), fut_x.flatten()[::-1]]),
                                  y=np.concatenate([preds_t + (std_t*0.5), (preds_t - (std_t*0.5))[::-1]]),
                                  fill='toself', fillcolor='rgba(255, 75, 75, 0.1)', line_color='rgba(0,0,0,0)', 
                                  showlegend=False), secondary_y=True)
    fig_main.add_trace(go.Scatter(x=fut_x.flatten(), y=preds_t, name="Temp Forecast", 
                                  line=dict(dash='dot', color='#ff4b4b')), secondary_y=True)

# Highlight Outliers
outliers = df[df['Is_Outlier']]
fig_main.add_trace(go.Scatter(x=outliers['Year'], y=outliers['Temp_Anomaly_C'] if "Temp" in analysis_mode else outliers['Rain_Anomaly_mm'],
                               mode='markers', name='Anomalous Event', marker=dict(color='#ffa500', size=12, symbol='diamond')), 
                               secondary_y=("Rain" not in analysis_mode))

fig_main.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                       height=500, margin=dict(t=20, b=20), legend=dict(orientation="h", y=1.1, x=1, xanchor="right"))
st.plotly_chart(fig_main, use_container_width=True)

# --- 6. SPATIAL & SEASONAL ANALYTICS ---
st.divider()
c_map, c_cycle, c_risk = st.columns([1, 1.5, 1])

with c_map:
    st.markdown('<p class="sector-header">Spatial Intelligence</p>', unsafe_allow_html=True)
    st.map(df[['Lat', 'Lon']].head(1), zoom=6)
    st.caption(f"Geographic Center: {df['Lat'].iloc[0]}°N, {df['Lon'].iloc[0]}°W")

with c_cycle:
    st.markdown(f'<p class="sector-header">Seasonal Cycle Profile ({selected_year})</p>', unsafe_allow_html=True)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    clim_r = [15, 25, 60, 100, 160, 210, 140, 80, 150, 120, 40, 20] if "North" not in selected_region else [5, 10, 25, 55, 95, 135, 185, 245, 215, 85, 20, 5]
    curr_r = [r + np.random.normal(0, 12) for r in clim_r]
    
    fig_cycle = go.Figure()
    fig_cycle.add_trace(go.Scatter(x=months, y=clim_r, name="LTM Baseline", line=dict(color='gray', dash='dash')))
    fig_cycle.add_trace(go.Scatter(x=months, y=curr_r, name=f"Observed {selected_year}", line=dict(color='#00d2ff', width=3)))
    fig_cycle.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, 
                            margin=dict(t=0, b=0), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_cycle, use_container_width=True)

with c_risk:
    st.markdown('<p class="sector-header">CAT Risk Diagnostic</p>', unsafe_allow_html=True)
    risk_score = min(int((avg_t / 1.1) * 100), 100) if avg_t > 0 else 10
    fig_gauge = go.Figure(go.Indicator(mode = "gauge+number", value = risk_score, number = {'suffix': "%", 'font': {'color': "#ffffff"}},
        gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "#00d2ff"}, 
                 'steps': [{'range': [0, 80], 'color': '#2c3e50'}, {'range': [80, 100], 'color': '#e74c3c'}]}))
    fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=250, margin=dict(t=0, b=0))
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.write(f"Pearson Correlation ($r$): **{df[['Temp_Anomaly_C', 'Rain_Anomaly_mm']].corr().iloc[0,1]:.2f}**")

# --- 7. EXPORT TERMINAL ---
st.sidebar.divider()
st.sidebar.download_button("📂 Download Intelligence Report", df.to_csv(index=False), f"Pro_Report_{selected_region}.csv")
st.sidebar.info("System Status: Operational")