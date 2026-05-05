🌍 GCI Pro-Suite | Regional Climate Intelligence Platform
Ghana Climate Intelligence (GCI) is an enterprise-grade climate analytics system that transforms over a century of meteorological data into actionable, decision-ready intelligence.

The platform enables institutions, researchers, and analysts to understand climate variability, assess risk exposure, and project future environmental trends across Ghana’s regions.

🔗 Access the System
🌍 Live Application: https://gci-climate-engine.streamlit.app/

💻 GitHub Repository: https://github.com/christiankumi49/ghana-climate-dashboard

🚀 Core Capabilities
📊 Regional Climate Intelligence
High-resolution analysis of temperature and rainfall anomalies (1901–2026) across Ghana’s regions.

📈 Predictive Analytics Engine
Utilizes Linear Regression (OLS), Ridge, and Random Forest models to forecast climate trends up to 2060.

🌡️ Signal Processing
Applies rolling averages (decadal smoothing) to filter noise and reveal long-term climate patterns.

⚠️ Climate Risk Modeling
Implements a Climate Exposure Index (CAT Risk) to quantify vulnerability to extreme environmental conditions.

🤖 AI Diagnostics
Automated system insights highlighting:

Warming trends

Rainfall variability

Risk classification (LOW → CRITICAL)

🧠 How It Works
1. Data Engine
Historical dataset (1901–2020+)

Optional live NASA satellite data integration

2. Processing Layer
Anomaly computation (baseline deviation)

Rolling trend smoothing

Outlier detection (Z-score methodology)

3. Modeling Layer
Regression-based climate modeling

Forecast generation up to selected horizon year

4. Intelligence Output
Risk classification

Climate diagnostics

Interactive dashboards

Exportable reports

📊 Understanding the Analytics
Positive Anomaly (+): Warmer or wetter than historical baseline

Negative Anomaly (-): Cooler or drier than historical baseline

Confidence Bands:
Statistical uncertainty is represented using ≈ ±2σ (95% confidence range).

🛠️ Tech Stack
Language: Python 3.10+

Frontend: Streamlit (custom UI architecture)

Data Processing: Pandas, NumPy

Visualization: Plotly, Matplotlib

Machine Learning: Scikit-learn

Data Sources: NASA POWER API + historical datasets

💼 Use Cases
🌾 Agriculture: Crop planning and drought monitoring

🏦 Finance: Climate-informed investment and lending decisions

🛡️ Insurance: Climate risk and index-based modeling

🏙️ Urban Planning: Infrastructure resilience analysis

🔬 Research: Long-term climate trend evaluation

⚙️ Installation & Setup
git clone https://github.com/christiankumi49/ghana-climate-dashboard.git
cd ghana-climate-dashboard
pip install -r requirements.txt
streamlit run app.py
📄 Features
Interactive climate dashboards

Live + historical data integration

Predictive climate projections

Geospatial visualization

Exportable PDF intelligence reports

👨‍💻 Author
Christian Kumi
BSc Meteorology & Climate Science — KNUST

Building at the intersection of:
Climate Science × Data Science × Risk Intelligence

⚠️ Disclaimer
This platform is a decision-support system based on statistical models and available climate data.
It is intended for analytical purposes and does not guarantee exact environmental or financial outcomes
