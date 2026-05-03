import streamlit as st
import requests
import pandas as pd
import numpy as np

st.set_page_config(page_title="BTC Predictor", layout="wide")

# -----------------------------
# STYLE
# -----------------------------
st.markdown("""
<style>
.big-title {
    font-size: 40px;
    font-weight: bold;
    color: #00FFD1;
}
.card {
    background-color: #111;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
.metric {
    font-size: 28px;
    font-weight: bold;
}
.label {
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# FETCH DATA
# -----------------------------
@st.cache_data
def get_data():
    url = "https://data-api.binance.vision/api/v3/klines"
    params = {"symbol":"BTCUSDT","interval":"1h","limit":500}
    data = requests.get(url, params=params).json()
    
    df = pd.DataFrame(data)
    df[4] = df[4].astype(float)
    return df[4]

data = get_data()

returns = np.log(data / data.shift(1)).dropna()

# -----------------------------
# MODEL
# -----------------------------
mu = returns.mean()

long_vol = returns.std()
short_vol = returns.tail(10).std()
sigma = 0.7 * long_vol + 0.3 * short_vol
sigma = sigma * 1.08

current_price = data.iloc[-1]

Z = np.random.randn(3000)
sims = current_price * np.exp((mu - 0.5 * sigma**2) + sigma * Z)

low, high = np.percentile(sims, [3, 97])

# -----------------------------
# UI
# -----------------------------
st.markdown('<div class="big-title">🚀 BTC 1H Predictor (GBM)</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="card"><div class="label">Current Price</div><div class="metric">${:.2f}</div></div>'.format(current_price), unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><div class="label">Predicted Low</div><div class="metric">${:.2f}</div></div>'.format(low), unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card"><div class="label">Predicted High</div><div class="metric">${:.2f}</div></div>'.format(high), unsafe_allow_html=True)

st.markdown("---")

# Backtest metrics (update manually from Colab)
coverage = 0.9432
width = 1225.49

col4, col5 = st.columns(2)

with col4:
    st.metric("Coverage", coverage)

with col5:
    st.metric("Avg Width", width)

st.markdown("### 📈 Recent Price Movement")
st.line_chart(data.tail(100))
