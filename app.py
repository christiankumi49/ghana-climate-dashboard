import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. Page Configuration
st.set_page_config(page_title="Ghana Climate Intelligence Dashboard", layout="wide")

# 2. Load Data
@st.cache_data
def load_data():
    try:
        # This matches the filename in your GitHub repository
        df = pd.read_csv('Ghana_Climate_Anomalies_Aligned.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

# 3. Sidebar & Download Button
st.sidebar.header("🕹️ Dashboard Controls")

if not df.empty:
    # This creates the CSV file for the user to download
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="📥 Download Full Dataset (CSV)",
        data=csv_data,
        file_name='ghana_climate_data.csv',
        mime='text/csv',
    )

selected_region = st.sidebar.selectbox(
    "Select Study Area", 
    options=["National Average", "Ashanti", "Greater Accra", "Northern", "Western"]
)

# 4. Main Title
st.title(f"🇬🇭 {selected_region} Climate Intelligence")

# 5. Display Charts if data exists
if not df.empty:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Rainfall Bar Chart
    fig.add_trace(
        go.Bar(x=df['Year'], y=df['Rain_Anomaly_mm'], name="Rainfall Anomaly"),
        secondary_y=False
    )
    
    # Temperature Line Chart
    fig.add_trace(
        go.Scatter(x=df['Year'], y=df['Temp_Anomaly_C'], name="Temp Anomaly", line=dict(color='red')),
        secondary_y=True
    )

    fig.update_layout(title_text="Historical Climate Trends")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Please ensure 'Ghana_Climate_Anomalies_Aligned.csv' is uploaded to your GitHub repository.")