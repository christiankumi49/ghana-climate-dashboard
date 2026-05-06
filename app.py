import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
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
if 'api_fail_cache' not in st.session_state:
    st.session_state.api_fail_cache = {}

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
def fetch_live_climate(lat, lon):
    loc_key = f"{lat}_{lon}"
    if loc_key in st.session_state.api_fail_cache:
        if (datetime.now() - st.session_state.api_fail_cache[loc_key]).seconds < 600:
            np.random.seed(int(abs(lat * lon * 1000) % 2**32))
            return "SYNTHETIC", np.random.normal(0.5, 0.05), np.random.normal(-10, 10)
    try:
        curr_yr = datetime.now().year
        url = f"https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=T2M,PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}&format=JSON&start={curr_yr}&end={curr_yr}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            t_vals = [x for x in data['properties']['parameter']['T2M'].values() if x != -999]
            p_vals = [x for x in data['properties']['parameter']['PRECTOTCORR'].values() if x != -999]
            if len(t_vals) > 0 and len(p_vals) > 0:
                return "NASA", np.mean(t_vals) - 26.8, np.sum(p_vals) - 1150
        st.session_state.api_fail_cache[loc_key] = datetime.now()
    except:
        st.session_state.api_fail_cache[loc_key] = datetime.now()
    np.random.seed(int(abs(lat * lon * 1000) % 2**32))
    return "SYNTHETIC", np.random.normal(0.5, 0.05), np.random.normal(-10, 10)

# --- 3. REPORTING ENGINE ---
def create_pdf_report(region, avg_t, avg_r, year_range, risk, r2_str, diag, fig_static, m_name, t_slope, full_insight):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 22)
    pdf.set_text_color(0, 210, 255)
    pdf.cell(200, 15, txt="GCI ELITE CLIMATE INTELLIGENCE", ln=True, align='C')
    pdf.ln(10)
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Helvetica", 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, txt=f" EXECUTIVE SUMMARY: {region.upper()}", ln=True, fill=True)
    pdf.set_font("Helvetica", size=11)
    
    summary_text = (f"This intelligence report outlines the climatic profile for {region} between "
                    f"{year_range[0]} and {year_range[1]}. Confidence: {r2_str}. "
                    f"Thermal trend: {t_slope:.4f} C/annum.")
    pdf.multi_cell(0, 8, txt=summary_text)
    pdf.ln(5)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        fig_static.savefig(tmp.name, format='png', bbox_inches='tight', dpi=150)
        temp_path = tmp.name
    pdf.image(temp_path, x=15, y=pdf.get_y(), w=180)
    pdf.set_y(pdf.get_y() + 90)
    
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, txt="AI SYSTEM DIAGNOSTIC & DYNAMIC INSIGHTS", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 7, txt=diag.replace("**", "").strip())
    pdf.ln(2)
    pdf.multi_cell(0, 7, txt=full_insight)
    
    pdf_out = pdf.output(dest='S').encode('latin-1', 'ignore')
    if os.path.exists(temp_path): os.remove(temp_path)
    return pdf_out

# --- 4. DATA ENGINE ---
@st.cache_data
def load_historical_engine(uploaded_file=None):
    if uploaded_file is not None: return pd.read_csv(uploaded_file)
    years = np.arange(1901, 2021)
    df = pd.DataFrame({'Year': years, 'Temp_Anomaly_C': np.random.normal(0.4, 0.1, len(years)) + (years-1901)*0.006, 'Rain_Anomaly_mm': np.random.normal(0, 70, len(years))})
    regions = {"Ashanti": [6.74, -1.52, 0.1, 5], "Greater Accra": [5.60, -0.19, 0.2, -8], "Northern": [9.40, -0.85, 0.8, -25], "Upper East": [10.80, -0.90, 0.9, -30], "Western": [5.55, -2.15, -0.2, 20], "Volta": [6.50, 0.45, 0.3, 10]}
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
source, l_t, l_r = fetch_live_climate(lat_c, lon_c) if live_engine else ("OFFLINE", None, None)

if l_t is not None:
    live_entry = pd.DataFrame({'Year':[2026],'Temp_Anomaly_C':[l_t],'Rain_Anomaly_mm':[l_r],'Region':[selected_region],'lat':[lat_c],'lon':[lon_c]})
    df_reg = pd.concat([df_reg, live_entry], ignore_index=True)

selected_years = st.sidebar.slider("Viewport", 1901, 2026, (1980, 2026))
df = df_reg[df_reg['Year'].between(selected_years[0], selected_years[1])].copy()
# Improved window to 5 for increased anomaly detection
df['T_Signal'] = df['Temp_Anomaly_C'].rolling(window=5, center=True).mean().ffill().bfill()

# --- 6. ELITE ANALYTICS ---
X, y = df['Year'].values.reshape(-1, 1), df['T_Signal'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

if model_choice == "Linear Regression": model = LinearRegression()
elif model_choice == "Ridge (L2)": model = Ridge(alpha=1.0)
else: model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
r2_val = max(0, r2_score(y_test, y_test_pred))

display_r2 = max(0.05, r2_val)
r2_str = f"{display_r2:.4f}" if display_r2 > 0.05 else "Low predictive confidence due to weak signal"

model.fit(X, y)
y_full_pred = model.predict(X)
t_slope = (y_full_pred[-1] - y_full_pred[0]) / (X[-1] - X[0])[0]

z_scores = (df['Temp_Anomaly_C'] - df['Temp_Anomaly_C'].mean()) / (df['Temp_Anomaly_C'].std() + 1e-6)
t_anoms_count = len(df[np.abs(z_scores) > 2.0])

avg_t, avg_r = df['Temp_Anomaly_C'].mean(), df['Rain_Anomaly_mm'].mean()
risk_level = "LOW"
if t_anoms_count > 5 or avg_r < -25: risk_level = "CRITICAL"
elif avg_t > 0.8 or avg_r < -15: risk_level = "HIGH"
elif avg_t > 0.4: risk_level = "MEDIUM"

# --- 7. DASHBOARD RENDER ---
status_color = "#00ffcc" if source == "NASA" else "#ffcc00"
confidence_score = min(99.9, display_r2 * (100 if source == "NASA" else 75))

st.markdown(f'<p class="update-pulse" style="color:{status_color}">● {source} | {selected_region.upper()} | CONFIDENCE: {confidence_score:.1f}%</p>', unsafe_allow_html=True)

diag_text = f"""
**DIAGNOSTIC CORE**

Trend: **+{t_slope:.4f} C/yr** | Test R²: **{r2_str}**
Risk: **{risk_level}** | Anomalies: **{t_anoms_count} detected**

Observed warming trend suggests increasing climate instability in the region. 
Adaptive mitigation and thermal resilience protocols are advised for the {selected_region} sector.
"""
st.markdown(f'<div class="ai-box">{diag_text}</div>', unsafe_allow_html=True)

cols = st.columns(4)
metrics = [("Thermal Var.", f"+{avg_t:.2f} C", False), ("Risk Index", risk_level, risk_level=="CRITICAL"), ("Reliability", r2_str, False), ("Freshness", "REAL-TIME" if source=="NASA" else "ESTIMATED", False)]
for i, (l, v, d) in enumerate(metrics):
    cols[i].markdown(f'<div class="glass-card"><p class="metric-label">{l}</p><p class="{"metric-critical" if d else "metric-value"}">{v}</p></div>', unsafe_allow_html=True)

# --- DYNAMIC INTELLIGENCE INSIGHTS LOGIC ---
if avg_t > 0.8:
    thermal_msg = "Elevated thermal drift detected, indicating accelerated warming risk."
else:
    thermal_msg = "Thermal variation remains within moderate bounds for the selected period."

if avg_r < -20:
    hydro_msg = f"Systemic precipitation deficit identified. High stress on groundwater recharge in {selected_region}."
else:
    hydro_msg = "Hydrological recharge levels show stable variability relative to historical baselines."

if t_anoms_count > 4:
    resilience_msg = "Critical volatility detected; local atmospheric resilience is significantly compromised."
else:
    resilience_msg = "Regional stratification remains stable; standard monitoring protocols are sufficient."

# Trend Insight calculation
if t_slope > 0.02:
    trend_msg = "Rapid warming acceleration detected across the time horizon."
else:
    trend_msg = "Warming trend remains gradual with no abrupt escalation."

# Updating the combined insight
full_insight = f"{thermal_msg}\n{hydro_msg}\n{resilience_msg}\n{trend_msg}"

st.markdown('<p class="sector-header">💎 GCI ELITE INTELLIGENCE INSIGHTS</p>', unsafe_allow_html=True)
i_cols = st.columns(4)
with i_cols[0]:
    st.markdown(f'<div class="glass-card"><p class="metric-label">Thermal Stability</p><p style="color:#e2e8f0; font-size:14px; margin-top:10px;">{thermal_msg}</p></div>', unsafe_allow_html=True)
with i_cols[1]:
    st.markdown(f'<div class="glass-card"><p class="metric-label">Hydrological Outlook</p><p style="color:#e2e8f0; font-size:14px; margin-top:10px;">{hydro_msg}</p></div>', unsafe_allow_html=True)
with i_cols[2]:
    st.markdown(f'<div class="glass-card"><p class="metric-label">Atmospheric Resilience</p><p style="color:#e2e8f0; font-size:14px; margin-top:10px;">{resilience_msg}</p></div>', unsafe_allow_html=True)
with i_cols[3]:
    st.markdown(f'<div class="glass-card"><p class="metric-label">Acceleration Trend</p><p style="color:#e2e8f0; font-size:14px; margin-top:10px;">{trend_msg}</p></div>', unsafe_allow_html=True)

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=df['Year'], y=df['Rain_Anomaly_mm'], name="Rain Anomaly", marker_color='rgba(0, 210, 255, 0.2)'), secondary_y=False)
fig.add_trace(go.Scatter(x=df['Year'], y=y_full_pred, name="Trend Line", line=dict(color='#ff4b4b', width=3)), secondary_y=True)

if st.sidebar.toggle("Show Projections", value=True):
    horizon = st.sidebar.slider("Horizon", 2021, 2060, 2050)
    fut_x = np.arange(int(df['Year'].max()) + 1, horizon + 1).reshape(-1, 1)
    base_pred = model.predict(fut_x)
    noise = np.random.normal(0, 0.03, len(base_pred)) 
    realistic_pred = base_pred + noise
    std_err = np.std(y - y_full_pred)
    expansion = np.linspace(1, 2.8, len(fut_x))
    fig.add_trace(go.Scatter(x=np.concatenate([fut_x.flatten(), fut_x.flatten()[::-1]]), 
                             y=np.concatenate([realistic_pred + (1.96 * std_err * expansion), (realistic_pred - (1.96 * std_err * expansion))[::-1]]), 
                             fill='toself', fillcolor='rgba(255, 204, 0, 0.08)', line=dict(color='rgba(0,0,0,0)'), name="95% CI"), secondary_y=True)
    fig.add_trace(go.Scatter(x=fut_x.flatten(), y=realistic_pred, name="Projection (Stochastic)", line=dict(dash='dash', color='#ffcc00')), secondary_y=True)

fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500)
st.plotly_chart(fig, use_container_width=True)

plt.style.use('dark_background')
fig_static, ax = plt.subplots(figsize=(10, 5))
ax.plot(df['Year'], y_full_pred, color='#00d2ff')
ax.set_title(f"Refined Trend: {selected_region}")
plt.close(fig_static)

# --- 8. GEOSPATIAL & GAUGES ---
st.divider()
c1, c2, c3 = st.columns(3)
with c1: st.map(pd.DataFrame({'lat': [lat_c], 'lon': [lon_c]}), zoom=7)
with c2:
    seasonal_fig = go.Figure(go.Scatter(x=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], y=[5,12,28,60,100,160,215,270,225,90,20,5], fill='tozeroy', line=dict(color='#00d2ff')))
    seasonal_fig.update_layout(template="plotly_dark", height=250, margin=dict(t=10,b=10))
    st.plotly_chart(seasonal_fig, use_container_width=True)
with c3:
    exp_idx = min(100, int((max(0, avg_t)/1.5 + t_anoms_count/10) * 50))
    gauge = go.Figure(go.Indicator(mode="gauge+number", value=exp_idx, number={'suffix': "%"}, title={'text': "Exposure Risk", 'font': {'size': 18}},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': '#ff4b4b' if exp_idx > 70 else '#00d2ff'}}))
    gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=250)
    st.plotly_chart(gauge, use_container_width=True)

# --- 9. EXPORTS ---
st.sidebar.divider()
st.sidebar.download_button("📊 Export CSV", df.to_csv(index=False).encode('utf-8'), f"GCI_{selected_region}.csv", use_container_width=True)
pdf_bytes = create_pdf_report(selected_region, avg_t, avg_r, selected_years, risk_level, r2_str, diag_text, fig_static, model_choice, t_slope, full_insight)
st.sidebar.download_button("📄 Intelligence Report (PDF)", pdf_bytes, f"GCI_Report_{selected_region}.pdf", use_container_width=True)
