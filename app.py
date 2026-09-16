import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from arch import arch_model
import time

# --- 1. Page Setup & UI/UX ---
st.set_page_config(page_title="AI Financial Dashboard", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    .big-font { font-size:20px !important; font-weight: bold; color: #1f77b4; }
    </style>
""", unsafe_allow_html=True)

st.title("📈 AI Financial Forecast Dashboard")
st.markdown("Predict stock trends and market risk using Deep Learning.")

# --- 2. Load the AI Models (Bulletproof Method) ---
@st.cache_resource
def load_models():
    # Use Input layer to avoid the Keras input_shape warning
    model = Sequential()
    model.add(Input(shape=(60, 1)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.load_weights('lstm_model.weights.h5')
    scaler = joblib.load('scaler.pkl')
    
    return model, scaler

lstm_model, scaler = load_models()

# --- 3. Sidebar for User Input ---
st.sidebar.header("⚙️ User Settings")
ticker = st.sidebar.selectbox("Select a Stock", ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"])
st.sidebar.markdown("---")
st.sidebar.info("This dashboard uses an LSTM Neural Network for price forecasting and a GARCH model for real-time risk analysis.")

# --- 4. Fetch Data & Run Predictions ---
@st.cache_data(ttl=3600)
def get_data(ticker):
    # Retry mechanism using the more robust yf.Ticker method
    for attempt in range(3):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="5y") # "5y" gets 5 years of data
            if not df.empty:
                break
        except Exception:
            if attempt < 2:
                time.sleep(2) # Wait 2 seconds before retrying
            else:
                return None

    if df.empty:
        return None
        
    # yf.Ticker.history already returns a clean DataFrame, just ensure it's float
    close_prices = df['Close'].astype(float)
    
    df['Daily_Return'] = close_prices.pct_change()
    df['Rolling_Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
    
    return df.dropna()

df = get_data(ticker)

# Safety check if Yahoo Finance is blocking requests
if df is None or df.empty:
    st.error(f"⚠️ Could not fetch data for {ticker}. Yahoo Finance is temporarily blocking requests from this server. Please wait a minute and refresh the page, or try a different stock.")
    st.stop()

current_price = float(df['Close'].iloc[-1])

# Run LSTM Prediction
last_60_days = df['Close'].values[-60:]
last_60_scaled = scaler.transform(last_60_days.reshape(-1, 1))
X_new = last_60_scaled.reshape(1, 60, 1)
predicted_scaled = lstm_model.predict(X_new, verbose=0)
predicted_price = float(scaler.inverse_transform(predicted_scaled)[0][0])

# Run GARCH Volatility
returns = df['Daily_Return'].dropna() * 100
garch_spec = arch_model(returns, vol='Garch', p=1, q=1)
garch_fit = garch_spec.fit(disp='off')
forecast = garch_fit.forecast(horizon=30)
forecasted_vol = float((forecast.variance.values[-1, :] ** 0.5) * np.sqrt(252))

# --- 5. Main Dashboard Display ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label=f"Current {ticker} Price", value=f"${current_price:.2f}")
with col2:
    change = predicted_price - current_price
    st.metric(label="AI Predicted Price (Tomorrow)", value=f"${predicted_price:.2f}", delta=f"{change:.2f}")
with col3:
    st.metric(label="Forecasted Risk (30 Days)", value=f"{forecasted_vol*100:.1f}%")

st.markdown("---")

tab1, tab2 = st.tabs(["📊 Price Prediction", "⚠️ Risk Analysis"])

with tab1:
    st.subheader("Historical Price vs AI Prediction")
    chart_data = pd.DataFrame({
        "Actual Price": df['Close'].tail(100),
        "AI Prediction": [None]*99 + [predicted_price]
    })
    st.line_chart(chart_data, use_container_width=True)
    
    if predicted_price > current_price * 1.02:
        st.success("🟢 **BUY Signal:** AI predicts an upward trend.")
    elif predicted_price < current_price * 0.98:
        st.error("🔴 **SELL Signal:** AI predicts a downward trend.")
    else:
        st.warning("🟡 **HOLD Signal:** Price expected to remain stable.")

with tab2:
    st.subheader("Market Volatility Forecast (Next 30 Days)")
    vol_chart = pd.DataFrame({
        "Historical Volatility": df['Rolling_Volatility'].tail(100) * 100,
        "AI Forecasted Volatility": [None]*99 + [forecasted_vol * 100]
    })
    st.line_chart(vol_chart, use_container_width=True)
    
    if forecasted_vol > 0.25:
        st.error("⚠️ **High Risk:** Market is expected to be highly volatile.")
    elif forecasted_vol > 0.15:
        st.warning("⚠️ **Medium Risk:** Moderate market fluctuations expected.")
    else:
        st.success("✅ **Low Risk:** Market is expected to be stable.")
