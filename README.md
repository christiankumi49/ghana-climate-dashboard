#  GCI Pro-Suite | Regional Climate Intelligence Profile

**Ghana Climate Intelligence (GCI)** is an enterprise-grade analytical dashboard engineered to visualize 125+ years of meteorological anomalies and project future climate trajectories across Ghana’s 16 administrative regions.

---

# Core Capabilities
* **Regional Intelligence Architecture:** High-fidelity analysis of temperature and precipitation anomalies using a 1901–2026 historical baseline.
* **Predictive Analytics Engine:** Integrated **Ordinary Least Squares (OLS) Regression** providing statistical projections for climate horizons up to 2060.
* **Signal Processing:** Implementation of 25-year rolling decadal means to filter noise from long-term climatic trends.
* **CAT Risk Modeling:** Dynamic Catastrophe (CAT) Exposure Indexing to quantify regional vulnerability to extreme thermal variance.
* **Strategic Insights:** Automated alert system for severe drought and flood risks based on historical standard deviations.

# Understanding the Analytics
* **Anomaly Logic:** Values represent deviation from the century-long historical mean. 
    * **Positive (+) Anomaly:** Indicates conditions warmer or wetter than the 1901–2020 baseline.
    * **Negative (-) Anomaly:** Indicates conditions cooler or drier than the 1901–2020 baseline.
* **Confidence Intervals:** Visualized via $2\sigma$ (Sigma) shading to represent statistical variance in predictive trends.

# Tech Stack & Frameworks
* **Language:** Python 3.10+
* **UI/UX:** Streamlit (Custom CSS-injected Glassmorphism architecture)
* **Data Science:** Pandas, NumPy (Vectorized signal processing)
* **Visualization:** Plotly Graph Objects (Multi-axis scientific plotting)
* **Machine Learning:** Scikit-Learn (Linear Regression for decadal planning)

#System Execution & Deployment

### Local Environment Setup
To initialize the Intelligence Suite on your local machine:

1. *Clone the Repository:**
   ```bash
   git clone [https://github.com/christiankumi49/ghana-climate-dashboard.git](https://github.com/christiankumi49/ghana-climate-dashboard.git)
   cd ghana-climate-dashboard
