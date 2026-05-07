import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from scipy import stats
from fpdf import FPDF
import matplotlib.pyplot as plt
import os
import tempfile
import requests

# --- 1. SETTINGS & STYLING ---
st.set_page_config(page_title="GCI Elite | Professional Grade", layout="wide")

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
    .ai-box {
        background: linear-gradient(45deg, rgba(0, 210, 255, 0.08), rgba(0, 255, 204, 0.08));
        border: 1px solid #00d2ff;
        padding: 20px;
        border-radius: 12px;
        color: #e2e8f0;
        margin-bottom: 25px;
        border-left: 5px solid #00ffcc;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINES ---
@st.cache_data(ttl=86400)
def get_scientific_data(lat, lon):
    """Fetches real NASA historical data and computes 30yr climatology."""
    url = f"https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=T2M,PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}&format=JSON&start=1981&end=2025"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            raw = resp.json()['properties']['parameter']
            df = pd.DataFrame({
                'Temp': list(raw['T2M'].values()),
                'Precip': list(raw['PRECTOTCORR'].values())
            })
            df = df[df['Temp'] > -50]
            df['Year'] = [int(str(k)[:4]) for k in raw['T2M'].keys()]
            annual = df.groupby('Year').agg({'Temp': 'mean', 'Precip': 'sum'}).reset_index()
            
            baseline_df = annual[(annual['Year'] >= 1991) & (annual['Year'] <= 2020)]
            t_base = baseline_df['Temp'].mean()
            r_base = baseline_df['Precip'].mean()
            
            annual['Temp_Anomaly_C'] = annual['Temp'] - t_base
            annual['Rain_Anomaly_mm'] = annual['Precip'] - r_base
            # Smoothing for trend visibility
            annual['T_Signal'] = annual['Temp_Anomaly_C'].rolling(window=5, center=True).mean().ffill().bfill()
            return annual, t_base, r_base
    except:
        return None, None, None

def run_mann_kendall(data):
    """Statistical significance testing."""
    n = len(data)
    s = 0
    for i in range(n-1):
        for j in range(i+1, n):
            s += np.sign(data[j] - data[i])
    z = s / (np.sqrt(n*(n-1)*(2*n+5)/18) + 1e-6)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return p

def generate_forecast(df, horizon=25):
    """Holt-Winters Exponential Smoothing with uncertainty zones."""
    series = df['Temp_Anomaly_C'].values
    model = ExponentialSmoothing(series, trend='add', seasonal=None, initialization_method="estimated")
    model_fit = model.fit()
    
    forecast_steps = np.arange(1, horizon + 1)
    forecast_values = model_fit.forecast(horizon)
    rmse = np.sqrt(mean_squared_error(series, model_fit.fittedvalues))
    uncertainty = 1.96 * rmse * np.sqrt(forecast_steps)
    
    last_year = df['Year'].max()
    forecast_years = np.arange(last_year + 1, last_year + horizon + 1)
    return forecast_years, forecast_values, uncertainty, rmse

# --- 3. REPORTING ENGINE ---
def create_pdf_report(region, stats_dict, img_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(11, 16, 22)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_font("Arial", 'B', 24)
    pdf.set_text_color(0, 210, 255)
    pdf.text(10, 25, "GCI ELITE INTELLIGENCE REPORT")
    
    pdf.set_y(50)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, f"Analysis Target: {region.upper()}", ln=True)
    
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 7, "This document contains high-fidelity climate risk analysis based on 30-year climatology baselines and non-linear trend modeling.")
    
    pdf.ln(5)
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(95, 10, " METRIC", border=1, fill=True)
    pdf.cell(95, 10, " VALUE", border=1, fill=True, ln=True)
    
    pdf.set_font("Arial", '', 10)
    for label, val in stats_dict.items():
        pdf.cell(95, 10, f" {label}", border=1)
        pdf.cell(95, 10, f" {val}", border=1, ln=True)
    
    pdf.ln(10)
    pdf.image(img_path, x=10, w=190)
    return pdf.output(dest='S').encode('latin-1')

# --- 4. MAIN INTERFACE ---
st.sidebar.title("💎 GCI ELITE SUITE")
nav = st.sidebar.radio("Navigation", ["Executive Hub", "Regional Comparator"])

regions = {
    "Accra": [5.60, -0.19], "Kumasi": [6.68, -1.62], 
    "Tamale": [9.40, -0.85], "Takoradi": [4.90, -1.77]
}
selected_name = st.sidebar.selectbox("Geographic Focus", options=list(regions.keys()))
lat, lon = regions[selected_name]

# Global Data Load
df, t_base, r_base = get_scientific_data(lat, lon)

if df is not None:
    # Scientific Calcs
    X = df['Year'].values.reshape(-1, 1)
    y = df['T_Signal'].values
    lin_model = LinearRegression().fit(X, y)
    t_slope = lin_model.coef_[0]
    p_val = run_mann_kendall(df['Temp_Anomaly_C'].values)
    is_sig = p_val < 0.05
    
    # Forecast Calcs
    f_yrs, f_vals, f_err, f_rmse = generate_forecast(df, 25)

    if nav == "Executive Hub":
        st.markdown(f'<p style="color:#00ffcc">● LIVE SCIENTIFIC ARCHIVE | {selected_name.upper()}</p>', unsafe_allow_html=True)
        
        # Diagnostic Box
        diag = f"**{selected_name} Resilience Profile:** Trend is **{'+' if t_slope>0 else ''}{t_slope*10:.2f}°C/decade**. Statistical significance is **{'VALIDATED' if is_sig else 'PENDING'}** ($p={p_val:.4f}$)."
        st.markdown(f'<div class="ai-box">{diag}</div>', unsafe_allow_html=True)
        
        # Dashboard Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="glass-card"><p class="metric-label">Decadal Trend</p><p class="metric-value">+{t_slope*10:.2f}°C</p></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="glass-card"><p class="metric-label">Significance</p><p class="metric-value">{"PASS" if is_sig else "FAIL"}</p></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="glass-card"><p class="metric-label">2050 Risk</p><p class="metric-value">+{f_vals[-1]:.2f}°C</p></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="glass-card"><p class="metric-label">Model RMSE</p><p class="metric-value">{f_rmse:.3f}</p></div>', unsafe_allow_html=True)

        # Main Viz
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df['Year'], y=df['Temp_Anomaly_C'], name="Historical Anomaly", mode='lines', line=dict(color='#8892b0', width=1)))
        fig.add_trace(go.Scatter(x=f_yrs, y=f_vals, name="AI Projection", line=dict(color='#ffcc00', dash='dash', width=3)))
        fig.add_trace(go.Scatter(
            x=np.concatenate([f_yrs, f_yrs[::-1]]),
            y=np.concatenate([f_vals + f_err, (f_vals - f_err)[::-1]]),
            fill='toself', fillcolor='rgba(255, 204, 0, 0.1)', line=dict(color='rgba(0,0,0,0)'), name="95% CI"
        ))
        fig.add_trace(go.Bar(x=df['Year'], y=df['Rain_Anomaly_mm'], name="Rain Anomaly", marker_color='rgba(0, 210, 255, 0.2)'), secondary_y=True)
        
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=550)
        st.plotly_chart(fig, use_container_width=True)

        # Report Export
        if st.sidebar.button("📥 Generate Intelligence Report"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                plt.style.use('dark_background')
                plt.figure(figsize=(10, 5))
                plt.plot(df['Year'], df['Temp_Anomaly_C'], color='#00d2ff')
                plt.title(f"Thermal Profile: {selected_name}")
                plt.savefig(tmp.name)
                plt.close()
                
                payload = {
                    "Decadal Slope": f"{t_slope*10:.3f} C",
                    "Significance (p)": f"{p_val:.4f}",
                    "2050 Projection": f"+{f_vals[-1]:.2f} C",
                    "Historical RMSE": f"{f_rmse:.4f}"
                }
                report = create_pdf_report(selected_name, payload, tmp.name)
                st.sidebar.download_button("📩 Download PDF", report, f"GCI_{selected_name}.pdf")
                os.remove(tmp.name)

    elif nav == "Regional Comparator":
        st.markdown('<p class="sector-header">MULTI-REGION CLIMATE DRIFT</p>', unsafe_allow_html=True)
        targets = st.multiselect("Select regions", options=list(regions.keys()), default=list(regions.keys())[:2])
        
        comp_fig = go.Figure()
        for t in targets:
            t_lat, t_lon = regions[t]
            t_df, _, _ = get_scientific_data(t_lat, t_lon)
            if t_df is not None:
                comp_fig.add_trace(go.Scatter(x=t_df['Year'], y=t_df['T_Signal'], name=t, line=dict(width=3)))
        
        comp_fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=600)
        st.plotly_chart(comp_fig, use_container_width=True)

else:
    st.error("Real-time data synchronization failed. Please check your internet connection.")
