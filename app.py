import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from fpdf import FPDF
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="GCI Pro-Suite", layout="wide")

# --- STYLE ---
st.markdown("""
<style>
.stApp { background-color: #0b1016; }
.glass-card {
    background: rgba(255,255,255,0.03);
    padding:20px; border-radius:12px;
    border:1px solid rgba(0,210,255,0.15);
}
.metric-label { color:#8892b0; font-size:11px; }
.metric-value { color:#00d2ff; font-size:28px; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# --- PDF ---
def create_pdf(region, avg_t, avg_r):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200,10,f"{region} Climate Report",ln=True)
    pdf.cell(200,10,f"Temp: {avg_t:.2f}C | Rain: {avg_r:.1f}mm",ln=True)
    return pdf.output(dest='S').encode('latin-1','ignore')

# --- DATA ---
@st.cache_data
def load_data():
    years = np.arange(1901,2026)
    df = pd.DataFrame({
        "Year":years,
        "Temp":np.random.normal(0.4,0.1,len(years))+(years-1901)*0.005,
        "Rain":np.random.normal(0,70,len(years))
    })

    regions = {
        "Ashanti":[6.7,-1.5,0.1,5],
        "Greater Accra":[5.6,-0.19,0.2,-8],
        "Northern":[9.4,-0.8,0.8,-25]
    }

    out=[]
    for r,(lat,lon,t,rn) in regions.items():
        temp=df.copy()
        temp["Region"]=r
        temp["lat"]=lat
        temp["lon"]=lon
        temp["Temp"]+=t
        temp["Rain"]+=rn
        out.append(temp)
    return pd.concat(out)

df_raw = load_data()

# --- SIDEBAR ---
st.sidebar.title("COMMAND CENTER")

regions = sorted(df_raw["Region"].unique())
r1 = st.sidebar.selectbox("Region 1", regions)
r2 = st.sidebar.selectbox("Compare With", regions)

years = st.sidebar.slider("Years",1901,2025,(1980,2025))

# --- FILTER ---
df1 = df_raw[(df_raw.Region==r1)&(df_raw.Year.between(*years))]
df2 = df_raw[(df_raw.Region==r2)&(df_raw.Year.between(*years))]

# --- METRICS ---
avg_t = df1.Temp.mean()
avg_r = df1.Rain.mean()

m1,m2,m3 = st.columns(3)

def metric(col,lab,val):
    col.markdown(f"<div class='glass-card'><div class='metric-label'>{lab}</div><div class='metric-value'>{val}</div></div>",unsafe_allow_html=True)

metric(m1,"Temp",f"{avg_t:.2f}C")
metric(m2,"Rain",f"{avg_r:.1f}mm")

risk="LOW"
if avg_t>0.8: risk="HIGH"
if avg_r<-20: risk="DROUGHT"

metric(m3,"Risk",risk)

# --- PLOT ---
fig = go.Figure()

fig.add_trace(go.Scatter(x=df1.Year,y=df1.Temp,name=f"{r1} Temp"))
fig.add_trace(go.Scatter(x=df2.Year,y=df2.Temp,name=f"{r2} Temp"))

st.plotly_chart(fig,use_container_width=True)

# --- ML ---
X = df1.Year.values.reshape(-1,1)
model = LinearRegression().fit(X,df1.Temp)

future = np.arange(2026,2050).reshape(-1,1)

poly = PolynomialFeatures(2)
model2 = LinearRegression().fit(poly.fit_transform(X),df1.Temp)

pred_lin = model.predict(future)
pred_poly = model2.predict(poly.fit_transform(future))

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=future.flatten(),y=pred_lin,name="Linear"))
fig2.add_trace(go.Scatter(x=future.flatten(),y=pred_poly,name="Polynomial"))

st.plotly_chart(fig2,use_container_width=True)

# --- INSIGHTS ---
st.subheader("Insights")

if avg_t>0.5:
    st.warning("Rising temperature → possible heat stress & crop impact")

if avg_r<-10:
    st.error("Low rainfall → drought risk, agriculture affected")

if avg_r>20:
    st.success("High rainfall → flood potential")

# --- MAP ---
st.map(df1[['lat','lon']].head(1))

# --- EXPORT ---
pdf = create_pdf(r1,avg_t,avg_r)
st.download_button("Download Report",pdf,f"{r1}.pdf")