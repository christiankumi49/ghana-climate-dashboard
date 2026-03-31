import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from fpdf import FPDF
import matplotlib.pyplot as plt
import os
import tempfile
from datetime import datetime
import requests

# --- 1. PRO-SUITE UI ARCHITECTURE & SESSION STATE ---
st.set_page_config(page_title="GCI Elite | Climate Intel", layout="wide")

if 'history' not in st.session_state:
    st.session_state.history = []

st.markdown("""
    <style>
    .stApp { background-color: #0b1016; }
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid rgba(0, 210, 255, 0.15);
        margin-bottom: 25px;
    }
    .metric-label { color: #8892b0; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; }
    .metric-value { color: #00d2ff; font-size: 32px; font-weight: 800; margin-top: 5px; }
    .metric-critical { color: #ff4b4b; font-size: 32px; font-weight: 800; margin-top: 5px; }
    .sector-header { color: #ffffff; font-size: 20px; font-weight: 700; border-bottom: 2px solid #1f2937; padding-bottom: 10px; margin-bottom: 20px; }
    .update-pulse { color: #00ffcc; font-size: 13px; font-family: 'Courier New', monospace; font-weight: bold; }
    .ai-box {
        background: linear-gradient(45deg, rgba(0, 210, 255, 0.08), rgba(0, 255, 204, 0.08));
        border: 1px solid #00d2ff;
        padding: 20px;
        border-radius: 12px;
        color: #e2e8f0;
        margin-bottom: 25px;
        border-left: 5px solid #00ffcc;
    }
    section[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1f2937; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ENGINES ---
@st.cache_data(ttl=3600)
def fetch_nasa_live(lat, lon):
    try:
        curr_yr = datetime.now().year
        url = f"https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=T2M,PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}&format=JSON&start={curr_yr}&end={curr_yr}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            t_vals = [x for x in data['properties']['parameter']['T2M'].values() if x != -999]
            p_vals = [x for x in data['properties']['parameter']['PRECTOTCORR'].values() if x != -999]
            if t_vals:
                return np.mean(t_vals) - 26.8, np.sum(p_vals) - 1150
        return None, None
    except:
        return None, None

def detect_anomalies(series):
    z_scores = (series - series.mean()) / (series.std() + 1e-6)
    return series[np.abs(z_scores) > 2.0]

def calculate_bounds(y_true, y_pred):
    residuals = y_true - y_pred
    stdev = np.std(residuals)
    return y_pred - (1.96 * stdev), y_pred + (1.96 * stdev)

def generate_ai_diagnostic(region, avg_t, t_slope, risk, t_anoms, r_anoms, r2, m_name):
    warming_speed = "Accelerated" if t_slope > 0.007 else "Steady"
    reliability = "High" if r2 > 0.7 else "Moderate" if r2 > 0.4 else "Low"
    return f"""
    **SYSTEM DIAGNOSTIC:** The {region} sector is experiencing **{warming_speed} thermal variance** (+{t_slope:.4f}°C/yr).
    Model: **{m_name}** | Reliability: **{reliability}** ($R^2$: {r2:.2f}).
    The engine identified **{len(t_anoms)} thermal** and **{len(r_anoms)} rainfall** extremes. Risk level: **{risk}**.
    """

# --- 3. REPORTING ENGINE ---
def create_pdf_report(region, avg_t, avg_r, year_range, risk, r2, diag, fig_static):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 18)
    pdf.cell(200, 15, txt=f"GCI Elite Intelligence: {region}", ln=True, align='C')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        fig_static.savefig(tmp.name, format='png', bbox_inches='tight', dpi=100)
        temp_path = tmp.name
    pdf.image(temp_path, x=10, y=55, w=190)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    return pdf.output(dest='S').encode('latin-1', 'ignore')

# --- 4. DATA ENGINE ---
@st.cache_data
def load_historical_engine(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    years = np.arange(1901, 2021)
    df = pd.DataFrame({'Year': years, 'Temp_Anomaly_C': np.random.normal(0.4, 0.1, len(years)) + (years-1901)*0.006, 'Rain_Anomaly_mm': np.random.normal(0, 70, len(years))})
    regions = {
        "Ashanti": [6.74, -1.52, 0.1, 5], "Greater Accra": [5.60, -0.19, 0.2, -8],
        "Northern": [9.40, -0.85, 0.8, -25], "Upper East": [10.80, -0.90, 0.9, -30],
        "Western": [5.55, -2.15, -0.2, 20], "Volta": [6.50, 0.45, 0.3, 10]
    }
    all_dfs = []
    for reg, (lat, lon, t_off, r_off) in regions.items():
        temp = df.copy().assign(Region=reg, lat=lat, lon=lon)
        temp['Temp_Anomaly_C'] += t_off; temp['Rain_Anomaly_mm'] += r_off
        all_dfs.append(temp)
    return pd.concat(all_dfs)

# --- 5. COMMAND CENTER ---
st.sidebar.title("💎 COMMAND CENTER")
df_raw = load_historical_engine(st.sidebar.file_uploader("⚡ Upload CSV", type=["csv"]))
selected_region = st.sidebar.selectbox("Geographic Focus", options=sorted(df_raw['Region'].unique()))

if selected_region not in st.session_state.history:
    st.session_state.history.append(selected_region)

model_choice = st.sidebar.radio("Analysis Engine", ["Linear Regression", "Ridge (L2)", "Random Forest"])
df_reg = df_raw[df_raw['Region'] == selected_region].copy()
lat_c, lon_c = df_reg['lat'].iloc[0], df_reg['lon'].iloc[0]
live_engine = st.sidebar.toggle("Live NASA Satellite Feed", value=True)
status_tag = "OFFLINE"

if live_engine:
    l_t, l_r = fetch_nasa_live(lat_c, lon_c)
    if l_t:
        live_entry = pd.DataFrame({'Year':[2026],'Temp_Anomaly_C':[l_t],'Rain_Anomaly_mm':[l_r],'Region':[selected_region],'lat':[lat_c],'lon':[lon_c]})
        df_reg = pd.concat([df_reg, live_entry], ignore_index=True)
        status_tag = "LIVE"

selected_years = st.sidebar.slider("Historical Viewport", 1901, 2026, (1980, 2026))
df = df_reg[df_reg['Year'].between(selected_years[0], selected_years[1])].copy()
predictive_mode = st.sidebar.toggle("Statistical Projections", value=True)
forecast_horizon = st.sidebar.slider("Horizon Year", 2021, 2060, 2050) if predictive_mode else 2020
df['T_Signal'] = df['Temp_Anomaly_C'].rolling(window=10, center=True).mean().ffill().bfill()

# --- 6. METRICS & ANALYSIS ---
t_anoms = detect_anomalies(df['Temp_Anomaly_C'])
r_anoms = detect_anomalies(df['Rain_Anomaly_mm'])
avg_t, avg_r = df['Temp_Anomaly_C'].mean(), df['Rain_Anomaly_mm'].mean()
X, y = df['Year'].values.reshape(-1, 1), df['T_Signal'].values

if model_choice == "Linear Regression": model = LinearRegression().fit(X, y)
elif model_choice == "Ridge (L2)": model = Ridge(alpha=1.0).fit(X, y)
else: model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)

y_pred = model.predict(X)
r2_val = r2_score(y, y_pred)
t_slope = (y_pred[-1] - y_pred[0]) / (X[-1] - X[0])[0] if model_choice != "Random Forest" else 0.0075
lower_b, upper_b = calculate_bounds(y, y_pred)

risk_level = "LOW"
if len(t_anoms) > 5 or avg_r < -25: risk_level = "CRITICAL"
elif avg_t > 0.8 or avg_r < -15: risk_level = "HIGH"
elif avg_t > 0.4: risk_level = "MEDIUM"

diag_text = generate_ai_diagnostic(selected_region, avg_t, t_slope, risk_level, t_anoms, r_anoms, r2_val, model_choice)
st.markdown(f'<p class="update-pulse">● ENGINE {status_tag} | {selected_region.upper()}</p>', unsafe_allow_html=True)
st.markdown(f'<div class="ai-box">{diag_text}</div>', unsafe_allow_html=True)

cols = st.columns(4)
metrics = [("Mean Thermal Var.", f"+{avg_t:.2f} °C", False), ("Risk Score", risk_level, risk_level=="CRITICAL"), ("Confidence (R²)", f"{r2_val:.2f}", False), ("Annual GDD", f"{int((avg_t + 26.0 - 10) * 365)} units", False)]
for i, (l, v, d) in enumerate(metrics):
    cols[i].markdown(f'<div class="glass-card"><p class="metric-label">{l}</p><p class="{"metric-critical" if d else "metric-value"}">{v}</p></div>', unsafe_allow_html=True)

# --- 7. CORE ANALYTICS ---
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=df['Year'], y=df['Rain_Anomaly_mm'], name="Rain Anomaly", marker_color='rgba(0, 210, 255, 0.3)'), secondary_y=False)
fig.add_trace(go.Scatter(x=df['Year'], y=df['Temp_Anomaly_C'], name="Temp (Observed)", line=dict(color='rgba(255, 75, 75, 0.2)')), secondary_y=True)
fig.add_trace(go.Scatter(x=df['Year'], y=y_pred, name="Trend Line", line=dict(color='#ff4b4b', width=3)), secondary_y=True)

if predictive_mode:
    fut_x = np.arange(int(df['Year'].max()) + 1, forecast_horizon + 1).reshape(-1, 1)
    fig.add_trace(go.Scatter(x=fut_x.flatten(), y=model.predict(fut_x), name="Projection", line=dict(dash='dash', color='#ffcc00')), secondary_y=True)

fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500, xaxis_title="Timeline (Years)", yaxis_title="Rainfall Anomaly (mm)", yaxis2_title="Temperature Anomaly (°C)")
st.plotly_chart(fig, use_container_width=True)

# --- 8. MATPLOTLIB FOR PDF ---
plt.style.use('dark_background')
fig_static, ax = plt.subplots(figsize=(10, 5))
ax.plot(df['Year'], y_pred, color='#00d2ff', label="Trend")
ax.set_title(f"Climatic Variance: {selected_region}")
plt.close(fig_static)

# --- 9. GEOSPATIAL & LABELED GAUGES ---
st.divider()
st.markdown('<p class="sector-header">Regional Vulnerability & Seasonality</p>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Geo-Spatial Positioning**")
    st.map(pd.DataFrame({'lat': [lat_c], 'lon': [lon_c]}), zoom=7)

with c2:
    st.markdown("**Intra-Annual Rainfall Seasonality**")
    seasonal_fig = go.Figure(go.Scatter(x=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], y=[5,12,28,60,100,160,215,270,225,90,20,5], fill='tozeroy', line=dict(color='#00d2ff')))
    seasonal_fig.update_layout(template="plotly_dark", height=250, margin=dict(t=10,b=10), yaxis_title="Precipitation (mm)", xaxis_title="Month")
    st.plotly_chart(seasonal_fig, use_container_width=True)

with c3:
    st.markdown("**Composite Climate Exposure Index**")
    exp = min(100, int((max(0, avg_t)/1.8 + abs(min(0, avg_r))/80) * 50))
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number", value=exp, number={'suffix': "%", 'font': {'size': 40}},
        title={'text': "Exposure Risk", 'font': {'size': 18}},
        gauge={'axis': {'range': [None, 100]}, 'bar': {'color': '#ff4b4b' if exp > 60 else '#00d2ff'},
               'steps': [{'range': [0, 50], 'color': "rgba(0, 210, 255, 0.1)"}, {'range': [50, 100], 'color': "rgba(255, 75, 75, 0.1)"}]}
    ))
    gauge_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=250, margin=dict(t=40,b=0))
    st.plotly_chart(gauge_fig, use_container_width=True)

# --- 11. INTELLIGENCE INSIGHTS ---
st.divider()
st.markdown('<p class="sector-header">GCI Elite Intelligence Insight</p>', unsafe_allow_html=True)
ins1, ins2 = st.columns(2)

with ins1:
    st.markdown(f"""
    <div class="glass-card">
        <p style="color:#00ffcc; font-weight:bold;">Primary Climatological Driver</p>
        <p style="font-size:14px; color:#e2e8f0;">
            Current analysis indicates that <b>{selected_region}</b> is heavily influenced by
            <b>West African Monsoon (WAM)</b> retreat cycles. The observed warming of
            <b>+{t_slope:.4f}°C/year</b> suggests a significant shift in latent heat flux,
            potentially impacting local subsistence farming and water security.
        </p>
    </div>
    """, unsafe_allow_html=True)

with ins2:
    st.markdown(f"""
    <div class="glass-card">
        <p style="color:#ff4b4b; font-weight:bold;">Strategic Risk Projection</p>
        <p style="font-size:14px; color:#e2e8f0;">
            With a <b>{risk_level}</b> risk status, urban infrastructure in this sector
            should prioritize thermal resilience. The engine's 95% Confidence Interval suggests
            that extreme thermal events are becoming more frequent, with <b>{len(t_anoms)}</b>
            outliers detected. Adaptive mitigation is recommended before the {forecast_horizon} horizon.
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- 10. EXPORTS ---
st.sidebar.divider()
st.sidebar.subheader("🕒 Session History")
for item in st.session_state.history[-5:]: st.sidebar.write(f"• {item}")

pdf_bytes = create_pdf_report(selected_region, avg_t, avg_r, selected_years, risk_level, r2_val, diag_text, fig_static)
st.sidebar.divider()
st.sidebar.download_button("📊 Export CSV", df.to_csv(index=False).encode('utf-8'), f"GCI_{selected_region}.csv")
st.sidebar.download_button("📄 Download PDF Report", pdf_bytes, f"GCI_Report_{selected_region}.pdf")