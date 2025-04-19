import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="Global Food Price Forecast",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("📈 Food Price Forecasting Dashboard")

country = st.selectbox("Select Country", ["Canada", "Australia", "Japan", "Sweden", "South Africa"])
item = st.selectbox("Select Food Item", ["Milk", "Bread", "Eggs", "Potatoes"])
months = st.slider("Forecast Months", min_value=3, max_value=24, step=3)

if st.button("Generate Forecast"):
    payload = {"country": country, "item": item, "months": months}
    response = requests.post("http://127.0.0.1:8000/forecast/", json=payload)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        st.line_chart(df.set_index("date")["forecast"])
        st.dataframe(df)
    else:
        st.error(f"❌ Error: {response.text}")
