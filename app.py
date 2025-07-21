# app.py

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Disable TensorFlow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Fetch historical stock data
def fetch_stock_data(stock_symbol, period="2y"):
    stock = yf.Ticker(stock_symbol)
    df = stock.history(period=period)
    df = df.reset_index()
    return df

# Prepare data for LSTM
def prepare_lstm_data(df, time_step=30):
    df = df[['Date', 'Close']].dropna()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df[['Close']])
    
    X, y = [], []
    for i in range(time_step, len(scaled)):
        X.append(scaled[i - time_step:i, 0])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y, scaler, df

# Train LSTM model
def train_lstm_model(X, y):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    history = model.fit(X, y, epochs=50, batch_size=32, verbose=0, validation_split=0.1)
    return model, history

# Predict next 7 trading days
def predict_next_7_days(model, df, scaler, time_step=30):
    last_sequence = df['Close'].values[-time_step:]
    scaled_sequence = scaler.transform(last_sequence.reshape(-1, 1))
    input_seq = scaled_sequence.reshape(1, time_step, 1)

    predictions = []
    future_dates = []
    last_date = df['Date'].iloc[-1]
    day_offset = 1

    while len(predictions) < 7:
        pred_scaled = model.predict(input_seq, verbose=0)
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
        predictions.append(pred_price)

        input_seq = np.append(input_seq[:, 1:, :], [[[pred_scaled[0][0]]]], axis=1)

        # Skip weekends
        next_date = last_date + timedelta(days=day_offset)
        if next_date.weekday() < 5:
            future_dates.append(next_date)
        day_offset += 1

    return dict(zip([d.strftime('%Y-%m-%d') for d in future_dates], predictions))


# Streamlit App
def main():
    st.set_page_config(page_title="📈 LSTM Stock Price Predictor", layout="wide")
    st.title("📊 LSTM-Based Stock Price Predictor")
    stock_symbol = st.text_input("Enter stock ticker (e.g., AAPL, TSLA, INFY):", value="AAPL").upper()

    if stock_symbol:
        st.write(f"Fetching historical data for **{stock_symbol}**...")
        df = fetch_stock_data(stock_symbol)

        if df.empty:
            st.error("❌ Invalid stock symbol or no data found.")
            return

        last_close_price = df['Close'].iloc[-1]
        last_close_date = df['Date'].iloc[-1].strftime('%Y-%m-%d')
        st.success(f"📌 Last Closing Price on {last_close_date}: **${last_close_price:.2f}**")

        X, y, scaler, df_prepared = prepare_lstm_data(df)
        model, history = train_lstm_model(X, y)
        predictions = predict_next_7_days(model, df_prepared, scaler)

        # Show Loss Curve
        st.subheader("📉 Model Training Loss")
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(history.history['loss'], label='Training Loss')
        ax_loss.plot(history.history['val_loss'], label='Validation Loss')
        ax_loss.set_title("LSTM Loss Curve")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()
        ax_loss.grid(True)
        st.pyplot(fig_loss)

        # Show Future Predictions
        st.subheader("🔮 Predicted Prices for Next 7 Trading Days")
        pred_df = pd.DataFrame({
            'Date': list(predictions.keys()),
            'Predicted Price ($)': [f"{p:.2f}" for p in predictions.values()]
        })
        st.table(pred_df)

        # Plotting actual + predicted
        st.subheader("📈 Forecast Visualization")
        fig_pred, ax_pred = plt.subplots(figsize=(10, 5))
        ax_pred.plot(df_prepared['Date'], df_prepared['Close'], label='Historical Prices')
        future_dates_dt = [datetime.strptime(d, "%Y-%m-%d") for d in predictions.keys()]
        future_prices = list(predictions.values())
        ax_pred.plot(future_dates_dt, future_prices, label='Predicted Prices', linestyle='--', marker='o')
        ax_pred.axhline(y=last_close_price, color='gray', linestyle=':', label='Last Close Price')
        ax_pred.set_title(f"{stock_symbol} Closing Price Forecast (LSTM)")
        ax_pred.set_xlabel("Date")
        ax_pred.set_ylabel("Price (USD)")
        ax_pred.legend()
        ax_pred.grid(True)
        st.pyplot(fig_pred)

if __name__ == "__main__":
    main()
