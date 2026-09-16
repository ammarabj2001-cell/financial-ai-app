import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from arch import arch_model

# --- 1. Page Setup & UI/UX ---
st.set_page_config(page_title="AI Financial Dashboard", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    .big-font { font-size:20px !important; font-weight: bold; color: #1f77b4; }
    </style>
""", unsafe_allow_html=True)

st.title("📈 AI Financial Forecast Dashboard")
st.markdown("Predict stock trends and market risk using Deep Learning.")

# --- 2. Load the AI Models ---
@st.cache_resource
def load_models():
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
    # Load the real historical data directly from GitHub
    df = pd.read_csv(f'{ticker}_data.csv', index_col=0, parse_dates=True)
    
    # Calculate returns and volatility
    df['Daily_Return'] = df['Close'].pct_change()
    df['Rolling_Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
    
    return df.dropna()

df = get_data(ticker)
current_price = float(df['Close'].iloc[-1])

# Run LSTM Prediction
last_60_days = df['Close'].values[-60:]
last_60_scaled = scaler.transform(last_60_days.reshape(-1, 1))
X_new = last_60_scaled.reshape(1, 60, 1)
predicted_scaled = lstm_model.predict(X_new, verbose=0)
predicted_price = float(scaler.inverse_transform(predicted_scaled)[0][0])

# Run GARCH Volatility
# We multiply by 100 so the GARCH model outputs volatility in percentage terms (e.g., 22.5 for 22.5%)
returns = df['Daily_Return'].dropna() * 100
garch_spec = arch_model(returns, vol='Garch', p=1, q=1)
garch_fit = garch_spec.fit(disp='off')
forecast = garch_fit.forecast(horizon=30)

# Extract the first day's volatility forecast (already in percentage terms)
forecasted_vol_array = (forecast.variance.values[-1, :] ** 0.5) * np.sqrt(252)
forecasted_vol = float(forecasted_vol_array[0])

# --- 5. Main Dashboard Display ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label=f"Current {ticker} Price", value=f"${current_price:.2f}")
with col2:
    change = predicted_price - current_price
    st.metric(label="AI Predicted Price (Tomorrow)", value=f"${predicted_price:.2f}", delta=f"{change:.2f}")
with col3:
    # FIXED: Removed * 100 because forecasted_vol is already a percentage
    st.metric(label="Forecasted Risk (30 Days)", value=f"{forecasted_vol:.1f}%")

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
        # Historical is already multiplied by 100 in the dataframe, so it matches the percentage scale
        "Historical Volatility": df['Rolling_Volatility'].tail(100) * 100,
        # FIXED: Removed * 100 so it matches the historical scale
        "AI Forecasted Volatility": [None]*99 + [forecasted_vol]
    })
    st.line_chart(vol_chart, use_container_width=True)
    
    # FIXED: Updated thresholds to match percentage values (e.g., 25.0 instead of 0.25)
    if forecasted_vol > 25.0:
        st.error("⚠️ **High Risk:** Market is expected to be highly volatile.")
    elif forecasted_vol > 15.0:
        st.warning("⚠️ **Medium Risk:** Moderate market fluctuations expected.")
    else:
        st.success("✅ **Low Risk:** Market is expected to be stable.")
