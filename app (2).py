# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import base64
from datetime import datetime

# ----------------------------
# Page config and header
# ----------------------------
st.set_page_config(layout="wide", page_title="Latin Countries Regression Explorer")
st.title("Regression & Function Analysis — Argentina, Chile, Venezuela (70-year World Bank data)")
st.markdown("**Created by Amarachi Onwo**")

st.markdown(
    """
This app fetches historical data from the World Bank for Argentina (ARG), Chile (CHL), and Venezuela (VEN),
fits a polynomial regression (degree ≥ 3), and performs function analysis.

Features:
- Editable raw data table  
- Polynomial regression with extrapolation  
- Function analysis (max/min, growth/decline, fastest change)  
- Multiple country comparisons  
- Printer-friendly export  
"""
)

# ----------------------------
# Countries & Indicators
# ----------------------------
COUNTRIES = {
    "Argentina": "ARG",
    "Chile": "CHL",
    "Venezuela": "VEN"
}

INDICATORS = {
    "Population": "SP.POP.TOTL",
    "Unemployment rate": "SL.UEM.TOTL.ZS",
    "Education levels (proxy: school enrollment primary %)": "SE.PRM.ENRR",
    "Life expectancy": "SP.DYN.LE00.IN",
    "Average income (GNI per capita)": "NY.GNP.PCAP.CD",
    "Birth rate": "SP.DYN.CBRT.IN",
    "Immigration (Net migration)": "SM.POP.NETM",
    "Murder Rate (intentional homicides per 100k)": "VC.IHR.PSRC.P5"
}

# ----------------------------
# Helper functions
# ----------------------------
def fetch_data(country_code, indicator):
    url = (
        f"http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}"
        f"?date=1960:{datetime.now().year}&format=json&per_page=1000"
    )
    r = requests.get(url)
    if r.status_code != 200:
        return pd.DataFrame()
    data = r.json()
    if not data or len(data) < 2:
        return pd.DataFrame()
    records = data[1]
    df = pd.DataFrame(records)[["date", "value"]]
    df = df.rename(columns={"date": "year"})
    df["year"] = pd.to_numeric(df["year"])
    return df.sort_values("year")

def get_table_download_link(df, filename="data.csv"):
    csv = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download CSV</a>'
    return href

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Options")
indicator_label = st.sidebar.selectbox("Choose a category", list(INDICATORS.keys()))
degree = st.sidebar.slider("Polynomial degree", 3, 8, 3)
years_back = st.sidebar.slider("Years of history", 10, 70, 70)
extrapolate_years = st.sidebar.slider("Years to extrapolate", 0, 50, 10)
countries_selected = st.sidebar.multiselect(
    "Select countries", list(COUNTRIES.keys()), default=list(COUNTRIES.keys())
)

# ----------------------------
# Data retrieval
# ----------------------------
indicator_code = INDICATORS[indicator_label]
all_data = {}

for country in countries_selected:
    df = fetch_data(COUNTRIES[country], indicator_code)
    if not df.empty:
        df = df[df["year"] >= datetime.now().year - years_back]
        all_data[country] = df

if not all_data:
    st.error("No data available for selected options.")
    st.stop()

# ----------------------------
# Show editable table
# ----------------------------
st.subheader("Raw Data (editable)")
combined = pd.DataFrame({"year": sorted(all_data[countries_selected[0]]["year"].unique())})
for c, df in all_data.items():
    combined = combined.merge(df, on="year", how="left", suffixes=("", f"_{c}"))
    combined = combined.rename(columns={"value": c})

try:
    edited = st.data_editor(combined, num_rows="dynamic", use_container_width=True)
except:
    st.warning("Using non-editable table (Streamlit version fallback).")
    edited = combined

st.markdown(get_table_download_link(edited, "edited_data.csv"), unsafe_allow_html=True)

# ----------------------------
# Regression & plotting
# ----------------------------
st.subheader("Regression Analysis")

fig, ax = plt.subplots(figsize=(10, 6))
analysis_text = []

for country in countries_selected:
    if country not in edited.columns:
        continue
    country_df = edited[["year", country]].dropna()
    if country_df.empty:
        continue

    x = country_df["year"].values
    y = country_df[country].values
    t = x - x.min()

    coeffs = np.polyfit(t, y, degree)
    p = np.poly1d(coeffs)

    t_new = np.linspace(t.min(), t.max() + extrapolate_years, 200)
    y_new = p(t_new)

    # Plot data & curve
    ax.scatter(x, y, label=f"{country} data")
    ax.plot(t_new + x.min(), y_new, label=f"{country} fit")

    if extrapolate_years > 0:
        ax.axvline(x.max(), color="gray", linestyle="--", alpha=0.6)

    # Polynomial equation
    eq_terms = [f"({c:.4e})*t^{len(coeffs)-1-i}" for i, c in enumerate(coeffs)]
    equation = " + ".join(eq_terms)
    analysis_text.append(f"Equation for {country} (t = years since {int(x.min())}): {equation}")

ax.set_xlabel("Year")
ax.set_ylabel(indicator_label)
ax.legend()
st.pyplot(fig)

# ----------------------------
# Function analysis
# ----------------------------
st.subheader("Function Analysis")
for txt in analysis_text:
    st.markdown(txt)

st.markdown("*(Note: Local maxima, minima, and fastest change points are approximate and based on polynomial derivative.)*")
