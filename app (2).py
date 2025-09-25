# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import base64

# ----------------------------
# Page config and header
# ----------------------------
st.set_page_config(layout="wide", page_title="Latin Countries Regression Explorer")
st.title("Regression & Function Analysis — Argentina, Chile, Venezuela (70-year World Bank data)")
st.markdown("**Created by Amarachi Onwo**")

st.markdown(
    """
This app fetches historical data from the World Bank for Argentina (ARG), Chile (CHL), and Venezuela (VEN),
fits a polynomial regression (degree >= 3), and performs function analysis.

Use the sidebar to pick:
- category (Population, Unemployment rate, Education proxy, Life expectancy, etc.)
- countries to include,
- polynomial degree (>= 3),
- how many years of history (up to 70),
- and optional extrapolation.

The raw data is shown in an editable table. The regression model is plotted as a scatter plot with fitted curve.
Extrapolated future data is shown with a dashed line.
"""
)

# ----------------------------
# Countries & indicators map
# ----------------------------
COUNTRIES = {
    "Argentina": "ARG",
    "Chile": "CHL",
    "Venezuela": "VEN"
}
