import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from fpdf import FPDF
from datetime import datetime
import requests

# --- 1. PRO-SUITE UI ARCHITECTURE ---
st.set_page_config(page_title="GCI Pro-Suite | Climate Intel", layout="wide")

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
    except: return None, None

def detect_anomalies(series):
    z_scores = (series - series.mean()) / series.std()
    return series[np.abs(z_scores) > 2.0]

def generate_ai_diagnostic(region, avg_t, t_slope, risk, anoms):
    warming_speed = "Accelerated" if t_slope > 0.007 else "Steady"
    extreme_count = len(anoms)
    return f"""
    **SYSTEM DIAGNOSTIC:** The {region} sector is experiencing **{warming_speed} thermal variance** (+{t_slope:.4f}°C/yr). 
    The Risk Engine has flagged a **{risk}** rating. Statistical analysis identified **{extreme_count}** extreme thermal 
    anomalies. Current moisture flux suggests a need for CAT modeling adjustments in agricultural planning.
    """

# --- 3. REPORTING ENGINE ---
def create_pdf_report(region, avg_t, avg_r, year_range, is_risky):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 18)
    pdf.cell(200, 15, txt=f"GCI Climate Intel Report: {region}", ln=True, align='C')
    pdf.set_font("Helvetica", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(200, 10, txt=f"Mean Thermal Variance: +{avg_t:.2f} C", ln=True)
    pdf.ln(15)
    status_text = "WARNING: HIGH CLIMATIC RISK" if is_risky else "Stable levels."
    pdf.multi_cell(0, 10, txt=f"Diagnostic Outcome: {status_text}.")
    return pdf.output(dest='S').encode('latin-1', 'ignore')

# --- 4. DATA ENGINE ---
@st.cache_data
def load_historical_engine(uploaded_file=None):
    if uploaded_file is not None: return pd.read_csv(uploaded_file)
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
uploaded_file = st.sidebar.file_uploader("⚡ Upload Custom CSV", type=["csv"])
df_raw = load_historical_engine(uploaded_file)
selected_region = st.sidebar.selectbox("Geographic Focus", options=sorted(df_raw['Region'].unique()))
df_reg = df_raw[df_raw['Region'] == selected_region].copy()
lat_c, lon_c = df_reg['lat'].iloc[0], df_reg['lon'].iloc[0]

live_engine = st.sidebar.toggle("Live NASA Satellite Feed", value=True)
status_tag = "OFFLINE"
if live_engine:
    with st.sidebar.status("Syncing Satellite..."):
        l_t, l_r = fetch_nasa_live(lat_c, lon_c)
        if l_t:
            live_entry = pd.DataFrame({'Year':[2026],'Temp_Anomaly_C':[l_t],'Rain_Anomaly_mm':[l_r],'Region':[selected_region],'lat':[lat_c],'lon':[lon_c]})
            df_reg = pd.concat([df_reg, live_entry], ignore_index=True)
            status_tag = "LIVE"

selected_years = st.sidebar.slider("Historical Viewport", 1901, 2026, (1980, 2026))
df = df_reg[df_reg['Year'].between(selected_years[0], selected_years[1])].copy()
analysis_mode = st.sidebar.radio("Primary Stream", ["Both", "Temperature", "Precipitation"])
predictive_mode = st.sidebar.toggle("Statistical Projections", value=True)
forecast_horizon = st.sidebar.slider("Horizon Year", 2021, 2060, 2050) if predictive_mode else 2020

# Signal Processing
df['T_Signal'] = df['Temp_Anomaly_C'].rolling(window=10, center=True).mean().ffill().bfill()
df['R_Signal'] = df['Rain_Anomaly_mm'].rolling(window=10, center=True).mean().ffill().bfill()

# --- 6. METRICS & RISK LOGIC ---
avg_t, avg_r = df['Temp_Anomaly_C'].mean(), df['Rain_Anomaly_mm'].mean()
X_train = df['Year'].values.reshape(-1, 1)
model_t = LinearRegression().fit(X_train, df['T_Signal'])
t_slope = model_t.coef_[0]
gdd_est = int((avg_t + 26.0 - 10) * 365)

risk_level = "LOW"
if avg_t > 1.2 or t_slope > 0.009: risk_level = "CRITICAL"
elif avg_t > 0.8 or avg_r < -15: risk_level = "HIGH"
elif avg_t > 0.4: risk_level = "MEDIUM"

st.markdown(f'<p class="update-pulse">● ENGINE {status_tag} | {selected_region.upper()}</p>', unsafe_allow_html=True)
anomalies_thermal = detect_anomalies(df['Temp_Anomaly_C'])
ai_summary = generate_ai_diagnostic(selected_region, avg_t, t_slope, risk_level, anomalies_thermal)
st.markdown(f'<div class="ai-box">{ai_summary}</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
def render_metric(col, lab, val, is_danger=False):
    style_class = "metric-critical" if is_danger else "metric-value"
    col.markdown(f'<div class="glass-card"><p class="metric-label">{lab}</p><p class="{style_class}">{val}</p></div>', unsafe_allow_html=True)

render_metric(m1, "Mean Thermal Var.", f"+{avg_t:.2f} °C")
render_metric(m2, "Rain Anomaly Δ", f"{avg_r:.1f} mm")
render_metric(m3, "Risk Score", risk_level, is_danger=(risk_level == "CRITICAL"))
render_metric(m4, "Annual GDD", f"{gdd_est} units")

# --- 7. CORE ANALYTICS ---
st.markdown('<p class="sector-header">Standardized Hydro-Climatic Analysis</p>', unsafe_allow_html=True)
fig_main = make_subplots(specs=[[{"secondary_y": True}]])
if analysis_mode in ["Both", "Precipitation"]:
    fig_main.add_trace(go.Bar(x=df['Year'], y=df['Rain_Anomaly_mm'], name="Rain Anomaly", marker_color='rgba(0, 210, 255, 0.3)'), secondary_y=False)
if analysis_mode in ["Both", "Temperature"]:
    fig_main.add_trace(go.Scatter(x=df['Year'], y=df['Temp_Anomaly_C'], name="Temp Anomaly", line=dict(color='rgba(255, 75, 75, 0.2)')), secondary_y=True)
    fig_main.add_trace(go.Scatter(x=df['Year'], y=df['T_Signal'], name="Decadal Trend", line=dict(color='#ff4b4b', width=3)), secondary_y=True)
    if not anomalies_thermal.empty:
        fig_main.add_trace(go.Scatter(x=anomalies_thermal.index + df['Year'].min(), y=anomalies_thermal.values, mode='markers', name='Extreme Event', marker=dict(color='#00ffcc', size=10, symbol='diamond')), secondary_y=True)

fig_main.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=480)
st.plotly_chart(fig_main, use_container_width=True)

# --- 8. INTELLIGENCE INSIGHTS (RESTORED DROUGHT/FLOOD LOGIC) ---
st.markdown('<p class="sector-header">🧠 Intelligence & Real-World Insights</p>', unsafe_allow_html=True)
i1, i2 = st.columns(2)
with i1:
    if t_slope > 0.007: st.warning(f"🔥 **Rapid Warming:** {selected_region} shows high thermal acceleration.")
    elif t_slope > 0.003: st.info("⚠️ **Moderate Warming:** Upward decadal trend detected.")
    else: st.success("✅ **Thermal Stability:** Temperature trends are within normal bounds.")
with i2:
    if avg_r < -25: st.error("🚨 **Drought Signal:** Critical negative rainfall anomaly detected.")
    elif avg_r < -10: st.warning("⚠️ **Moisture Deficit:** Region is trending towards dry conditions.")
    elif avg_r > 25: st.info("🌧️ **Flood Risk:** Significant positive rainfall anomaly; monitor runoff.")
    else: st.success("🌊 **Hydrological Balance:** Precipitation levels are stable.")

# --- 9. GEOSPATIAL & EXPOSURE ---
st.divider()
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<p class="sector-header">Regional Anchor</p>', unsafe_allow_html=True)
    st.map(pd.DataFrame({'lat': [lat_c], 'lon': [lon_c]}), zoom=7)
with c2:
    st.markdown('<p class="sector-header">Monthly Climatology</p>', unsafe_allow_html=True)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    clim_r = [5, 12, 28, 60, 100, 160, 215, 270, 225, 90, 20, 5]
    st.plotly_chart(go.Figure(go.Scatter(x=months, y=clim_r, fill='tozeroy', line=dict(color='#00d2ff'))).update_layout(template="plotly_dark", height=220, margin=dict(t=0,b=0,l=0,r=0)), use_container_width=True)
with c3:
    st.markdown('<p class="sector-header">Exposure Index</p>', unsafe_allow_html=True)
    exposure_val = min(100, int((max(0, avg_t)/1.8 + abs(min(0, avg_r))/80) * 50))
    st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=exposure_val, number={'suffix': "%"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#ff4b4b" if exposure_val > 70 else "#00d2ff"}})).update_layout(paper_bgcolor='rgba(0,0,0,0)', height=220, margin=dict(t=0,b=0)), use_container_width=True)

# --- 10. EXPORTS ---
st.sidebar.divider()
st.sidebar.download_button("📊 Export Trends (.CSV)", df.to_csv(index=False).encode('utf-8'), f"GCI_{selected_region}_Trends.csv", "text/csv")
pdf_data = create_pdf_report(selected_region, avg_t, avg_r, selected_years, risk_level in ["HIGH", "CRITICAL"])
st.sidebar.download_button("📄 Download Intelligence Report", pdf_data, f"GCI_{selected_region}.pdf")