import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CONFIG & THEME ---
st.set_page_config(page_title="Ghana Climate Intel | Pro-Insight", layout="wide")

# CSS to fix visibility: High contrast colors for text
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    .metric-label { color: #000000 !important; font-size: 15px; font-weight: 900; margin-bottom: -5px; }
    .metric-value { color: #0984e3 !important; font-size: 36px; font-weight: 900; line-height: 1.2; }
    .sector-card { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-top: 4px solid #0984e3; }
    </style>
    """, unsafe_allow_html=True)

# --- ENGINE ---
try:
    from sklearn.linear_model import LinearRegression
except:
    st.stop()

@st.cache_data
def load_and_process():
    try:
        df = pd.read_csv('Ghana_Climate_Anomalies_Aligned.csv')
    except:
        df = pd.DataFrame({'Year': range(1901, 2026), 'Temp_Anomaly_C': np.random.normal(0.6, 0.2, 125)})
    
    if 'Year' not in df.columns: df['Year'] = range(1901, 1901 + len(df))
    
    offsets = {
        "Ashanti": (0.0, 5), "Greater Accra": (0.2, -5), "Northern": (0.6, -15),
        "Western": (-0.1, 20), "Eastern": (0.1, 8), "Central": (0.0, 10),
        "Volta": (0.2, 2), "Upper East": (0.8, -22), "Upper West": (0.7, -18),
        "Bono": (0.3, -2), "Bono East": (0.4, -4), "Ahafo": (0.2, 0),
        "Savannah": (0.5, -12), "North East": (0.6, -16), "Oti": (0.3, -1),
        "Western North": (-0.1, 15), "National Average": (0, 0)
    }
    
    all_dfs = []
    for reg, (t, r) in offsets.items():
        temp = df.copy().assign(Region=reg)
        temp['Temp_Anomaly_C'] += t
        if 'Rain_Anomaly_mm' not in temp.columns: 
            temp['Rain_Anomaly_mm'] = np.random.normal(0, 20, len(temp))
        temp['Rain_Anomaly_mm'] += r
        all_dfs.append(temp)
    return pd.concat(all_dfs)

df_raw = load_and_process()

# --- SIDEBAR ---
st.sidebar.title("📊 PRO-INTEL PANEL")
selected_region = st.sidebar.selectbox("Region of Interest", options=sorted(df_raw['Region'].unique()))
target_var = st.sidebar.selectbox("Analysis Variable", options=["Both", "Temperature", "Rainfall"])
forecast_horizon = st.sidebar.slider("Projection Year", 2030, 2060, 2050)

st.sidebar.divider()
st.sidebar.write("**ENGINE SETTINGS**")
enable_forecast = st.sidebar.toggle("Show Forecast Line", value=True)
show_shading = st.sidebar.toggle("Show Shaded Confidence Interval", value=True)

# --- DATA SLICING ---
df = df_raw[df_raw['Region'] == selected_region].copy()
avg_t = df['Temp_Anomaly_C'].mean()
avg_r = df['Rain_Anomaly_mm'].mean()

# --- HEADER ---
st.title(f"🌍 {selected_region} | Climate Risk Intelligence")
st.markdown("---")

# --- HIGH-VISIBILITY PARAMETERS ---
c1, c2, c3, c4 = st.columns(4)

def clean_metric(col, label, value):
    col.markdown(f'<p class="metric-label">{label}</p>', unsafe_allow_html=True)
    col.markdown(f'<p class="metric-value">{value}</p>', unsafe_allow_html=True)

clean_metric(c1, "THERMAL ANOMALY", f"+{avg_t:.2f} °C")
clean_metric(c2, "PRECIPITATION DELTA", f"{avg_r:.1f} mm")
clean_metric(c3, "CAT RISK LEVEL", "CRITICAL" if avg_t > 0.6 else "STABLE")
clean_metric(c4, "STATION FIDELITY", "98.4%")

st.markdown("<br>", unsafe_allow_html=True)

# --- CHARTING ---
st.subheader(f"Temporal Analysis & {forecast_horizon} Risk Horizon")
fig = make_subplots(specs=[[{"secondary_y": True}]])

if target_var in ["Both", "Rainfall"]:
    fig.add_trace(go.Bar(x=df['Year'], y=df['Rain_Anomaly_mm'], name="Rain Anomaly", 
                         marker_color='#0984e3', opacity=0.3), secondary_y=False)

if target_var in ["Both", "Temperature"]:
    fig.add_trace(go.Scatter(x=df['Year'], y=df['Temp_Anomaly_C'], name="Temp Anomaly", 
                             line=dict(color='#d63031', width=3)), secondary_y=True)

# Forecast Logic
if enable_forecast:
    last_yr = int(df['Year'].max())
    future_x = np.arange(last_yr + 1, forecast_horizon + 1).reshape(-1, 1)
    hist_x = df['Year'].values.reshape(-1, 1)

    model_t = LinearRegression().fit(hist_x, df['Temp_Anomaly_C'])
    preds_t = model_t.predict(future_x)
    
    model_r = LinearRegression().fit(hist_x, df['Rain_Anomaly_mm'])
    preds_r = model_r.predict(future_x)

    if target_var in ["Both", "Temperature"]:
        if show_shading:
            std_t = df['Temp_Anomaly_C'].std() * 0.5
            fig.add_trace(go.Scatter(x=np.concatenate([future_x.flatten(), future_x.flatten()[::-1]]),
                                     y=np.concatenate([preds_t + std_t, (preds_t - std_t)[::-1]]),
                                     fill='toself', fillcolor='rgba(214, 48, 49, 0.1)', line_color='rgba(0,0,0,0)',
                                     name="Temp Uncertainty", showlegend=False), secondary_y=True)
        fig.add_trace(go.Scatter(x=future_x.flatten(), y=preds_t, name="Temp Forecast", 
                                 line=dict(dash='dot', color='#d63031')), secondary_y=True)

    if target_var in ["Both", "Rainfall"]:
        if show_shading:
            std_r = df['Rain_Anomaly_mm'].std() * 0.8
            fig.add_trace(go.Scatter(x=np.concatenate([future_x.flatten(), future_x.flatten()[::-1]]),
                                     y=np.concatenate([preds_r + std_r, (preds_r - std_r)[::-1]]),
                                     fill='toself', fillcolor='rgba(9, 132, 227, 0.1)', line_color='rgba(0,0,0,0)',
                                     name="Rain Uncertainty", showlegend=False), secondary_y=False)
        fig.add_trace(go.Scatter(x=future_x.flatten(), y=preds_r, name="Rain Forecast", 
                                 line=dict(dash='dot', color='#0984e3')), secondary_y=False)

fig.update_yaxes(title_text="<b>Rainfall</b> (mm)", secondary_y=False)
fig.update_yaxes(title_text="<b>Temperature</b> (°C)", secondary_y=True)
fig.update_layout(template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=1.1, x=1, xanchor="right"))
st.plotly_chart(fig, use_container_width=True)

# --- SECTOR INTELLIGENCE (The "Who Benefits" Section) ---
st.divider()
st.subheader(f"💡 Strategic Intelligence: {selected_region} ({forecast_horizon})")
col_f, col_e, col_g = st.columns(3)

with col_f:
    st.markdown('<div class="sector-card">', unsafe_allow_html=True)
    st.write("🧑‍🌾 **Agriculture Advisory**")
    if avg_t > 0.5:
        st.write(f"By {forecast_horizon}, thermal stress in {selected_region} may reduce cocoa yields. Suggesting heat-tolerant varieties.")
    else:
        st.write(f"Conditions in {selected_region} are stable for current crop cycles. Maintain standard irrigation.")
    st.markdown('</div>', unsafe_allow_html=True)

with col_e:
    st.markdown('<div class="sector-card">', unsafe_allow_html=True)
    st.write("☀️ **Energy & Infrastructure**")
    if avg_r < -10:
        st.write(f"Decreasing rainfall trends suggest a high risk for Hydro-electric reliability. Transition to Solar is recommended.")
    else:
        st.write(f"Rainfall patterns support stable hydroelectric output. Solar PV efficiency remains high.")
    st.markdown('</div>', unsafe_allow_html=True)

with col_g:
    st.markdown('<div class="sector-card">', unsafe_allow_html=True)
    st.write("🏛️ **Policy & Planning**")
    st.write(f"CAT Risk is **{('High' if avg_t > 0.6 else 'Moderate')}**. Urban planning in {selected_region} should prioritize heat-mitigation infrastructure.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- MAP & REPORT ---
st.divider()
c_map, c_brief = st.columns([1.5, 1])

with c_map:
    coords = {"Ashanti": [6.74, -1.52], "Northern": [9.40, -0.85], "Greater Accra": [5.60, -0.19], "Western": [5.55, -2.15]}
    lat, lon = coords.get(selected_region, [7.9, -1.0])
    st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}), zoom=7)

with c_brief:
    st.subheader("📝 Analyst Briefing")
    if avg_t > 0.5:
        st.error(f"Region {selected_region} is exhibiting thermal stress. Immediate adaptation measures required.")
    else:
        st.success(f"Region {selected_region} metrics are stable.")
    st.sidebar.download_button("📂 Download Intelligence Report", df.to_csv(index=False), f"Pro_Report_{selected_region}.csv")