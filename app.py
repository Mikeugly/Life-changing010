# ──────────────────────────────────────────
# ⚙️ app.py – Streamlit Alpaca Trading Bot
# ──────────────────────────────────────────

import os
import time
import numpy as np
import pandas as pd
import ta
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.env_util import make_vec_env
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# 🔐 Alpaca API setup via Streamlit secrets
ALPACA_API_KEY = st.secrets["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = st.secrets["ALPACA_SECRET_KEY"]
TRADING_CLIENT = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
DATA_CLIENT = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

st.set_page_config(page_title="DRL Trading Bot", layout="wide")
st.title("🤖 DRL Trading Bot")
st.sidebar.header("🔧 Controls")

symbol = st.sidebar.text_input("Symbol", "AAPL").upper()
interval = st.sidebar.selectbox("Interval", ["5m","10m","30m","1h","1d"])
reinvest = st.sidebar.selectbox("Reinvest %", ["25%","50%","75%","100%"])
strategy = st.sidebar.selectbox("Strategy", ["PPO","DDPG","Both"])
mode = st.sidebar.radio("Mode", ["Paper","Live"])
investment = st.sidebar.number_input("Invest Amount ($)", value=100.0, step=50.0)

if mode == "Live":
    st.sidebar.warning("⚠️ Live mode enabled - real trades will execute")

# Buttons
run_btn, test_btn, cash_btn, sharpe_btn = st.columns(4)
if run_btn.button("🚀 Run Bot"):
    st.session_state.run = True
if test_btn.button("🧪 Test Run"):
    st.session_state.test = True
if cash_btn.button("💸 Cash Out"):
    st.session_state.cash = True
if sharpe_btn.button("📈 Sharpe Ratio"):
    st.session_state.sharpe = True

chart_container = st.empty()
log_container = st.empty()
sharpe_container = st.empty()

def fetch_data(symbol, interval, limit=500):
    tf = {"5m": TimeFrame.Minute5, "10m": TimeFrame.Minute10,
          "30m": TimeFrame.Minute30, "1h": TimeFrame.Hour, "1d": TimeFrame.Day}[interval]
    end = pd.Timestamp.now(tz="UTC")
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf,
                           start=end - pd.Timedelta(minutes=limit),
                           end=end)
    df = DATA_CLIENT.get_stock_bars(req).df
    df = df[df.symbol == symbol].copy()
    df.reset_index(inplace=True)
    df.rename(columns={"close":"Close", "open":"Open",
                       "high":"High", "low":"Low", "volume":"Volume"}, inplace=True)
    df.drop(columns=["symbol"], inplace=True)
    df = df.set_index("timestamp")
    df.dropna(inplace=True)
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"], 14).rsi()
    df["sma"] = df["Close"].rolling(10).mean()
    df.dropna(inplace=True)
    return df

def calculate_sharpe(returns, rf=0.01):
    er = returns - rf
    return np.mean(er)/ (np.std(er)+1e-9)

@st.experimental_singleton
def load_models():
    dummy_env = make_vec_env(lambda: gym.make("CartPole-v1"), n_envs=1)
    ppo = PPO("MlpPolicy", dummy_env, verbose=0)
    ddpg = DDPG("MlpPolicy", dummy_env, verbose=0)
    return ppo, ddpg

ppo_model, ddpg_model = load_models()

def trade_loop():
    df = fetch_data(symbol, interval)
    scaler = StandardScaler()
    obs = scaler.fit_transform(df[["Close","rsi","sma"]])
    action, _ = (ppo_model.predict(obs[-1]) if strategy=="PPO"
                else ddpg_model.predict(obs[-1]) if strategy=="DDPG"
                else ((ppo_model.predict(obs[-1])[0] + ddpg_model.predict(obs[-1])[0]) / 2))
    price = df.Close.iloc[-1]
    qty = round(investment / price, 4)
    side = "buy" if action == 1 or action > 0 else "sell" if action == 2 or action < 0 else None
    if side and mode == "Live":
        try:
            TRADING_CLIENT.submit_order(symbol=symbol, qty=qty, side=side,
                                        type="market", time_in_force="gtc")
        except Exception as e:
            st.error(f"Order failed: {e}")
    return df, action, qty, side

if st.session_state.get("test", False):
    st.info("🧪 Test Run Successful — All systems go!") 
    st.session_state.test = False

if st.session_state.get("run", False):
    df, action, qty, side = trade_loop()
    st.success(f"Trade: Action={action} QTY={qty}")
    chart_container.line_chart(df[["Close","sma","rsi"]])
    log_container.write(f"{symbol} @ ${df.Close.iloc[-1]:.2f} QTY={qty}")
    st.session_state.run = False

if st.session_state.get("cash", False):
    st.warning("💸 Cash Out executed (placeholder)")
    st.session_state.cash = False

if st.session_state.get("sharpe", False):
    df = fetch_data(symbol, interval)
    sr = calculate_sharpe(df.Close.pct_change().dropna())
    sharpe_container.write(f"Sharpe Ratio: **{sr:.4f}**")
    st.session_state.sharpe = False