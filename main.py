import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten
import plotly.graph_objects as go
import time
from datetime import datetime as dt
import warnings
warnings.filterwarnings('ignore')

# Function to get live stock data
def get_live_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker,interval='1m')
    return data

# Function to split sequences
def split_sequences(sequences, n_steps):
    x, y = list(), list()
    for i in range(len(sequences)):
        end_idx = i + n_steps
        if end_idx > len(sequences)-1:
            break
        seq_x, seq_y = sequences[i:end_idx], sequences[end_idx]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)

n_steps = 3
n_features = 1
# Function to train LSTM model
def train_lstm_model(df):
    # Prepare data for LSTM
    x, y = split_sequences(df.values, n_steps)
    x = x.reshape((x.shape[0], x.shape[1], n_features))

    # Define and compile LSTM model
    model = Sequential([
        LSTM(256, activation='relu', input_shape=(n_steps, n_features), return_sequences=True),
        Flatten(),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(124, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(x, y, epochs=10, verbose=0)

    return model

# Function to predict future stock prices
def predict_future_prices(model, df, future_time_steps):
    # Prepare input data for forecasting
    latest_prices = df.values
    x1 = np.array([latest_prices[-n_steps:]]).reshape((1, n_steps, n_features)).astype('float32')
    forecasted_prices = []

    # Generate future predictions
    for i in range(future_time_steps):
        # Predict the next price
        p1 = model.predict(x1)
        forecasted_prices.append(p1[0][0])

        # Update input data for the next prediction
        latest_prices = np.append(latest_prices, p1[0][0])
        x1 = np.array([latest_prices[-n_steps:]]).reshape((1, n_steps, n_features)).astype('float32')

    return forecasted_prices
# Main function
def main():
    # Set up Streamlit app
    st.title('Stock Price Forecasting App')

    # Get live stock data


    # Streamlit loop
    fig = go.Figure()
    fig.update_xaxes(title='Time')
    fig.update_yaxes(title='Price')
    fig.update_layout(title='Actual vs Predicted Stock Prices')
    actual_line = fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Actual'))
    predicted_line = fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Predicted'))

    chart_placeholder = st.empty()  # Placeholder to hold the chart

    while True:
        # Predict future prices
        ticker = '^nsebank'
        start_date = '2024-02-01'
        end_date = dt.now()
        data = get_live_stock_data(ticker, start_date, end_date)
        df = data['Close']

        # Train LSTM model
        model = train_lstm_model(df)
        future_time_steps = 5
        forecasted_prices = predict_future_prices(model, df, future_time_steps)

        # Update plot
        actual_prices = np.concatenate((df.values, np.full(future_time_steps, np.nan)))
        predicted_prices = np.concatenate((np.full_like(df.values, np.nan), forecasted_prices))
        fig.data[0].x = np.arange(len(actual_prices))
        fig.data[0].y = actual_prices[-50:]
        fig.data[1].x = np.arange(len(predicted_prices))
        fig.data[1].y = predicted_prices[-50:]

        chart_placeholder.plotly_chart(fig)

        # Pause for a while before updating again
        time.sleep(1)  # Adjust the time interval as needed

if __name__ == "__main__":
    main()

## the final one which is working good
