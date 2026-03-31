import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from fpdf import FPDF
from datetime import datetime

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
    .sector-header { color: #ffffff; font-size: 20px; font-weight: 700; border-bottom: 2px solid #1f2937; padding-bottom: 10px; margin-bottom: 20px; }
    .update-pulse { color: #00ffcc; font-size: 13px; font-family: 'Courier New', monospace; font-weight: bold; }
    section[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1f2937; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. REPORTING ENGINE ---
def create_pdf_report(region, avg_t, avg_r, year_range, is_risky):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 18)
    pdf.cell(200, 15, txt=f"GCI Climate Intel Report: {region}", ln=True, align='C')
    pdf.set_font("Helvetica", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(200, 10, txt=f"Analysis Window: {year_range[0]} - {year_range[1]}", ln=True)
    pdf.cell(200, 10, txt=f"Mean Thermal Variance: +{avg_t:.2f} C", ln=True)
    pdf.cell(200, 10, txt=f"Avg. Precipitation Anomaly: {avg_r:.1f} mm", ln=True)
    pdf.ln(15)
    status_text = "WARNING: HIGH CLIMATIC RISK DETECTED" if is_risky else "Stable moisture levels observed."
    pdf.multi_cell(0, 10, txt=f"Diagnostic Outcome: {status_text}. Data processed via GCI Engine.")
    return pdf.output(dest='S').encode('latin-1', 'ignore')

# --- 3. DATA ENGINE ---
@st.cache_data
def load_historical_engine(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        years = np.arange(1901, 2026)
        df = pd.DataFrame({
            'Year': years, 
            'Temp_Anomaly_C': np.random.normal(0.4, 0.1, len(years)) + (years-1901)*0.006, 
            'Rain_Anomaly_mm': np.random.normal(0, 70, len(years))
        })
    
    regions = {
        "Ashanti": [6.74, -1.52, 0.1, 5], "Greater Accra": [5.60, -0.19, 0.2, -8],
        "Northern": [9.40, -0.85, 0.8, -25], "Upper East": [10.80, -0.90, 0.9, -30],
        "Western": [5.55, -2.15, -0.2, 20], "Volta": [6.50, 0.45, 0.3, 10],
        "Central": [5.50, -1.20, 0.0, 12], "Eastern": [6.10, -0.30, 0.1, 8],
        "Bono": [7.58, -2.33, 0.2, -5], "Savannah": [9.08, -1.82, 0.7, -20],
        "Upper West": [10.20, -2.10, 0.85, -28], "Ahafo": [7.0, -2.4, 0.15, 2],
        "Bono East": [7.7, -1.0, 0.3, -10], "North East": [10.5, -0.5, 0.88, -28],
        "Oti": [8.2, 0.3, 0.4, 5], "Western North": [6.3, -2.8, -0.1, 15]
    }
    
    all_dfs = []
    for reg, (lat, lon, t_off, r_off) in regions.items():
        temp = df.copy().assign(Region=reg, lat=lat, lon=lon)
        temp['Temp_Anomaly_C'] += t_off
        temp['Rain_Anomaly_mm'] += r_off
        all_dfs.append(temp)
    return pd.concat(all_dfs)

# --- 4. COMMAND CENTER (FIXED LOGIC) ---
st.sidebar.title("💎 COMMAND CENTER")
uploaded_file = st.sidebar.file_uploader("⚡ Upload Custom CSV", type=["csv"])
df_raw = load_historical_engine(uploaded_file)
selected_region = st.sidebar.selectbox("Geographic Focus", options=sorted(df_raw['Region'].unique()))

# Filter by region first to get Map Anchor
df_reg = df_raw[df_raw['Region'] == selected_region].copy()
reg_coords = pd.DataFrame({'lat': [df_reg['lat'].iloc[0]], 'lon': [df_reg['lon'].iloc[0]]})

selected_years = st.sidebar.slider("Historical Viewport", int(df_reg['Year'].min()), int(df_reg['Year'].max()), (1980, 2025))
df = df_reg[df_reg['Year'].between(selected_years[0], selected_years[1])].copy()
analysis_mode = st.sidebar.radio("Primary Stream", ["Both", "Temperature", "Precipitation"])

predictive_mode = st.sidebar.toggle("Statistical Projections", value=True)
forecast_horizon = st.sidebar.slider("Horizon Year", 2026, 2060, 2050) if predictive_mode else 2025

# Signal Processing
df['T_Signal'] = df['Temp_Anomaly_C'].rolling(window=10, center=True).mean().ffill().bfill()
df['R_Signal'] = df['Rain_Anomaly_mm'].rolling(window=10, center=True).mean().ffill().bfill()

# --- 5. UPDATED METRICS & REFINED RISK LOGIC ---
avg_t, avg_r = df['Temp_Anomaly_C'].mean(), df['Rain_Anomaly_mm'].mean()
X_train = df['Year'].values.reshape(-1, 1)
model_t = LinearRegression().fit(X_train, df['T_Signal'])
t_slope = model_t.coef_[0]

risk_level = "LOW"
if avg_t > 1.2 or t_slope > 0.009: risk_level = "CRITICAL"
elif avg_t > 0.8 or avg_r < -15: risk_level = "HIGH"
elif avg_t > 0.4: risk_level = "MEDIUM"

st.markdown(f'<p class="update-pulse">● ENGINE ACTIVE | {selected_region.upper()}</p>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)

def render_metric(col, lab, val):
    col.markdown(f'<div class="glass-card"><p class="metric-label">{lab}</p><p class="metric-value">{val}</p></div>', unsafe_allow_html=True)

render_metric(m1, "Mean Thermal Var.", f"+{avg_t:.2f} °C")
render_metric(m2, "Rain Anomaly Δ", f"{avg_r:.1f} mm")
render_metric(m3, "Risk Level ⚠️", risk_level)
render_metric(m4, "Latest Record", f"{int(df['Year'].max())}")

# --- 6. CORE ANALYTICS (FIXED STANDARD ALIGNMENT) ---
st.markdown('<p class="sector-header">Standardized Hydro-Climatic Analysis</p>', unsafe_allow_html=True)
fig_main = make_subplots(specs=[[{"secondary_y": True}]])

# Logic to align the Zero-line properly
rain_limit = max(abs(df['Rain_Anomaly_mm'].min()), abs(df['Rain_Anomaly_mm'].max()), 150)
temp_limit = max(df['Temp_Anomaly_C'].max(), 2.5)

if analysis_mode in ["Both", "Precipitation"]:
    fig_main.add_trace(go.Bar(x=df['Year'], y=df['Rain_Anomaly_mm'], name="Rain Anomaly", marker_color='rgba(0, 210, 255, 0.3)'), secondary_y=False)

if analysis_mode in ["Both", "Temperature"]:
    fig_main.add_trace(go.Scatter(x=df['Year'], y=df['Temp_Anomaly_C'], name="Temp Raw", line=dict(color='rgba(255, 75, 75, 0.2)')), secondary_y=True)
    fig_main.add_trace(go.Scatter(x=df['Year'], y=df['T_Signal'], name="Decadal Trend", line=dict(color='#ff4b4b', width=3)), secondary_y=True)
    
    if predictive_mode:
        fut_x = np.arange(int(df['Year'].max()) + 1, forecast_horizon + 1).reshape(-1, 1)
        # ML Upgrade: Polynomial
        poly = PolynomialFeatures(degree=2)
        poly_model = LinearRegression().fit(poly.fit_transform(X_train), df['T_Signal'])
        preds_poly = poly_model.predict(poly.fit_transform(fut_x))
        preds_lin = model_t.predict(fut_x)
        
        fig_main.add_trace(go.Scatter(x=fut_x.flatten(), y=preds_lin, name="Linear Proj.", line=dict(dash='dot', color='#ff4b4b')), secondary_y=True)
        fig_main.add_trace(go.Scatter(x=fut_x.flatten(), y=preds_poly, name="Polynomial (Acc.)", line=dict(dash='dash', color='#ffcc00')), secondary_y=True)

# THE FIX: Syncing axes so they look "standard"
fig_main.update_yaxes(title_text="Rainfall (mm)", range=[-rain_limit, rain_limit], secondary_y=False)
fig_main.update_yaxes(title_text="Temperature (°C)", range=[-0.2, temp_limit], secondary_y=True)
fig_main.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=480, hovermode="x unified")
st.plotly_chart(fig_main, use_container_width=True)

# --- 7. INTELLIGENCE INSIGHTS ---
st.markdown('<p class="sector-header">🧠 Intelligence Insights</p>', unsafe_allow_html=True)
i1, i2 = st.columns(2)
with i1:
    if t_slope > 0.007: st.warning(f"🔥 **Rapid Warming:** {selected_region} shows high thermal acceleration.")
    elif t_slope > 0.003: st.info("⚠️ **Moderate Warming:** Upward decadal trend detected.")
    else: st.success("✅ **Thermal Stability:** Minimal variance in this window.")
with i2:
    if avg_r < -15: st.error("🚨 **Drought Signal:** Persistent negative rainfall anomaly.")
    elif avg_r > 15: st.success("🌧️ **Moisture Surplus:** Rainfall is above historical baseline.")
    else: st.info("📊 **Balanced Hydrology:** Moisture within normal variance.")

# --- 8. GEOSPATIAL & RISK (FIXED ZOOM) ---
st.divider()
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    st.markdown('<p class="sector-header">Regional Anchor</p>', unsafe_allow_html=True)
    # This now forces the map to jump to the selected region
    st.map(reg_coords, zoom=7)
with c2:
    st.markdown('<p class="sector-header">Monthly Climatology</p>', unsafe_allow_html=True)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    is_north = selected_region in ["Upper East", "Upper West", "Northern", "Savannah", "North East"]
    clim_r = [5, 12, 28, 60, 100, 160, 215, 270, 225, 90, 20, 5] if is_north else [20, 35, 75, 115, 170, 225, 145, 85, 170, 130, 50, 25]
    st.plotly_chart(go.Figure(go.Scatter(x=months, y=clim_r, fill='tozeroy', line=dict(color='#00d2ff'))).update_layout(template="plotly_dark", height=220, margin=dict(t=0, b=0, l=0, r=0)), use_container_width=True)
with c3:
    st.markdown('<p class="sector-header">Exposure Index</p>', unsafe_allow_html=True)
    # Fixed formula for 0-100 gauge
    exposure = min(100, int((max(0, avg_t)/1.8 + abs(min(0, avg_r))/80) * 50))
    fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=exposure, number={'suffix': "%"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#00d2ff"},
               'steps': [{'range': [0, 60], 'color': '#1f2937'}, {'range': [60, 100], 'color': '#ef4444'}]}))
    st.plotly_chart(fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=220, margin=dict(t=0, b=0)), use_container_width=True)

# --- 9. EXPORTS ---
st.sidebar.divider()
pdf_data = create_pdf_report(selected_region, avg_t, avg_r, selected_years, risk_level in ["HIGH", "CRITICAL"])
st.sidebar.download_button("📄 Download PDF Report", pdf_data, f"GCI_{selected_region}.pdf")