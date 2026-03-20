# 🇬🇭 Ghana Climate Intelligence Dashboard (1901-2040)

An interactive meteorological tool built to analyze 120 years of historical climate data across Ghana and project future trends using Machine Learning.

##  Key Features
- **Historical Analysis:** Processes CRU TS4.07 datasets to calculate annual rainfall and temperature anomalies.
- **Regional Intelligence:** Provides spatial filtering for all 16 regions of Ghana with interactive mapping.
- **Predictive Engine:** Integrated Linear Regression model projecting climate trajectories to the year 2040.
- **Executive Summary:** Real-time calculation of period averages and record-breaking climate extremes.

## Understanding the Data
- **Anomaly Logic:** Values represent the deviation from the 1901–2020 historical mean.
- **Positive (+) Anomaly:** Indicates conditions warmer or wetter than the century average.
- **Negative (-) Anomaly:** Indicates conditions cooler or drier than the century average.

## Tech Stack
- **Language:** Python
- **Web Framework:** Streamlit
- **Data Science:** Pandas, Numpy, Xarray
- **Visualization:** Plotly
- **Machine Learning:** Scikit-Learn