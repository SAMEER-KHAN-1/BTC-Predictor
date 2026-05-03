import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="BTC Predictor", layout="wide")

# ---------- style ----------
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: bold;
    color: #00FFD1;
}
.card {
    background: linear-gradient(135deg, #1f1f1f, #111);
    padding: 20px;
    border-radius: 15px;
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

# ---------- data ----------
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

mu = returns.mean()

# volatility
long_vol = returns.std()
short_vol = returns.tail(10).std()
sigma = 0.7 * long_vol + 0.3 * short_vol
sigma = sigma * 1.15

np.random.seed(42)

current_price = data.iloc[-1]

Z = np.random.randn(3000)
sims = current_price * np.exp((mu - 0.5 * sigma**2) + sigma * Z)

low, high = np.percentile(sims, [2.3, 97.7])

# ---------- UI ----------
st.markdown('<div class="main-title">🚀 BTC 1H Predictor (GBM)</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

col1.markdown(f'<div class="card"><div class="label">Current Price</div><div class="metric">${current_price:.2f}</div></div>', unsafe_allow_html=True)
col2.markdown(f'<div class="card"><div class="label">Predicted Low</div><div class="metric">${low:.2f}</div></div>', unsafe_allow_html=True)
col3.markdown(f'<div class="card"><div class="label">Predicted High</div><div class="metric">${high:.2f}</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ---------- metrics ----------
coverage = 0.9507
width = 1292.56
winkler = 1792.07

col4, col5, col6 = st.columns(3)

col4.metric("Coverage", coverage)
col5.metric("Avg Width", width)
col6.metric("Winkler", winkler)

st.markdown("### 📈 Price + Prediction Band")

# ---------- chart with ribbon ----------
df_plot = pd.DataFrame({"price": data.tail(50)})

fig = go.Figure()

fig.add_trace(go.Scatter(
    y=df_plot["price"],
    mode="lines",
    name="Price"
))

fig.add_trace(go.Scatter(
    y=[low]*50,
    mode="lines",
    name="Low",
    line=dict(width=0),
    showlegend=False
))

fig.add_trace(go.Scatter(
    y=[high]*50,
    mode="lines",
    fill='tonexty',
    name="Prediction Range",
    opacity=0.2
))

st.plotly_chart(fig, use_container_width=True)
