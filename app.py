import streamlit as st
import requests
import pandas as pd

def get_data():
    url = "https://data-api.binance.vision/api/v3/klines"
    params = {"symbol":"BTCUSDT","interval":"1h","limit":100}
    data = requests.get(url, params=params).json()
    
    df = pd.DataFrame(data)
    df[4] = df[4].astype(float)
    return df[4]

data = get_data()

current_price = data.iloc[-1]

returns = data.pct_change().dropna()
std = returns.std()

low = current_price * (1 - 2*std)
high = current_price * (1 + 2*std)

st.title("BTC 1-Hour Predictor")

st.metric("Current Price", f"${current_price:.2f}")
st.metric("Predicted Range", f"${low:.2f} - ${high:.2f}")

st.line_chart(data)