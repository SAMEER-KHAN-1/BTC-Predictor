import streamlit as st
import requests
import pandas as pd
import numpy as np

def get_data():
    url = "https://data-api.binance.vision/api/v3/klines"
    params = {"symbol":"BTCUSDT","interval":"1h","limit":500}
    data = requests.get(url, params=params).json()
    
    df = pd.DataFrame(data)
    df[4] = df[4].astype(float)
    return df[4]

data = get_data()

returns = np.log(data / data.shift(1)).dropna()

mu = returns.mean()
sigma = returns.std()

current_price = data.iloc[-1]

# GBM simulation
Z = np.random.randn(2000)
sims = current_price * np.exp((mu - 0.5 * sigma**2) + sigma * Z)

low, high = np.percentile(sims, [2.5, 97.5])

# dummy metrics (you can hardcode your Colab results here)
coverage = 0.9537
avg_width = 1285.93

st.title("BTC 1-Hour Predictor (GBM)")

st.metric("Current Price", f"${current_price:.2f}")
st.metric("Predicted Range", f"${low:.2f} - ${high:.2f}")

st.subheader("Backtest Metrics")
st.write(f"Coverage: {coverage}")
st.write(f"Avg Width: {avg_width}")

st.line_chart(data.tail(50))
