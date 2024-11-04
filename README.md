
# Stock Price Forecasting App

## 1. Project Overview
This Streamlit app provides a real-time stock price forecasting model using an LSTM neural network. It fetches live stock data, trains a predictive model, and forecasts future stock prices, which are visualized alongside actual prices in real time.

## 2. Problem Statement
The goal of this project is to predict future stock prices based on historical and live market data. The app enables users to visualize actual and predicted stock prices for informed investment decisions.

## 3. Technologies and Libraries Used
- **Streamlit**: For building the interactive web application.
- **yfinance**: For fetching live stock data.
- **NumPy and Pandas**: For data manipulation and preprocessing.
- **TensorFlow**: For building and training the LSTM model.
- **Plotly**: For creating interactive and real-time data visualizations.
- **Warnings**: To suppress unnecessary warnings for a clean app interface.

## 4. Project Workflow
1. **Data Collection**: Fetches live stock data using the `yfinance` library.
2. **Data Preparation**: Splits sequences for time series forecasting using a sliding window approach.
3. **Model Training**: Builds and trains an LSTM model on the stock data.
4. **Forecasting**: Uses the trained LSTM model to predict future stock prices.
5. **Visualization**: Displays actual and predicted stock prices in real time with continuous updates.

## 5. Code Modules and Functions

### 5.1 Data Collection
```python
def get_live_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, interval='1m')
    return data
```
- **Purpose**: Fetches live stock data for a specified ticker and time range.
- **Parameters**:
  - `ticker`: Stock symbol.
  - `start_date` and `end_date`: Date range for data collection.

### 5.2 Data Preparation
```python
def split_sequences(sequences, n_steps):
    x, y = list(), list()
    for i in range(len(sequences)):
        end_idx = i + n_steps
        if end_idx > len(sequences) - 1:
            break
        seq_x, seq_y = sequences[i:end_idx], sequences[end_idx]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)
```
- **Purpose**: Splits data into input-output pairs to train the LSTM model.
- **Parameters**:
  - `sequences`: Time series data.
  - `n_steps`: Number of time steps for each input sequence.

### 5.3 Model Training
```python
def train_lstm_model(df):
    x, y = split_sequences(df.values, n_steps)
    x = x.reshape((x.shape[0], x.shape[1], n_features))
    
    model = Sequential([
        LSTM(256, activation='relu', input_shape=(n_steps, n_features), return_sequences=True),
        Flatten(),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(124, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=10, verbose=0)
    return model
```
- **Purpose**: Defines, compiles, and trains an LSTM model on the historical stock data.
- **Model Architecture**:
  - LSTM layer with 256 units.
  - Dense layers for non-linear transformations.
  - Output layer for predicting the next price point.
- **Loss Function**: Mean Squared Error (MSE), optimized with Adam.

### 5.4 Forecasting
```python
def predict_future_prices(model, df, future_time_steps):
    latest_prices = df.values
    x1 = np.array([latest_prices[-n_steps:]]).reshape((1, n_steps, n_features)).astype('float32')
    forecasted_prices = []
    for i in range(future_time_steps):
        p1 = model.predict(x1)
        forecasted_prices.append(p1[0][0])
        latest_prices = np.append(latest_prices, p1[0][0])
        x1 = np.array([latest_prices[-n_steps:]]).reshape((1, n_steps, n_features)).astype('float32')
    return forecasted_prices
```
- **Purpose**: Generates a specified number of future predictions by feeding the latest available data into the model.
- **Parameters**:
  - `future_time_steps`: Number of time steps to predict into the future.

### 5.5 Real-Time Visualization
```python
def main():
    st.title('Stock Price Forecasting App')
    ticker = '^nsebank'
    start_date = '2024-02-01'
    end_date = dt.now()
    data = get_live_stock_data(ticker, start_date, end_date)
    df = data['Close']
    
    fig = go.Figure()
    fig.update_xaxes(title='Time')
    fig.update_yaxes(title='Price')
    fig.update_layout(title='Actual vs Predicted Stock Prices')
    actual_line = fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Actual'))
    predicted_line = fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Predicted'))

    chart_placeholder = st.empty()

    while True:
        model = train_lstm_model(df)
        future_time_steps = 5
        forecasted_prices = predict_future_prices(model, df, future_time_steps)

        actual_prices = np.concatenate((df.values, np.full(future_time_steps, np.nan)))
        predicted_prices = np.concatenate((np.full_like(df.values, np.nan), forecasted_prices))
        
        fig.data[0].x = np.arange(len(actual_prices))
        fig.data[0].y = actual_prices[-50:]
        fig.data[1].x = np.arange(len(predicted_prices))
        fig.data[1].y = predicted_prices[-50:]

        chart_placeholder.plotly_chart(fig)
        time.sleep(1)
```
- **Purpose**: Sets up Streamlit interface, trains the model, and updates the forecasted prices on the interactive chart.
- **Visualization**: Uses Plotly to create a real-time line chart with actual and predicted stock prices.

## 6. Model Performance
- **Training Parameters**:
  - `n_steps = 3`: Number of previous time steps used for predicting the next step.
  - **Epochs**: 10, allowing the model to learn patterns from historical data.
- **Evaluation**: Mean Squared Error (MSE) is minimized during training, but real-time performance monitoring is challenging in a live forecasting model.

## 7. Results and Observations
- **Real-Time Predictions**: Predicted prices are displayed on the chart alongside actual prices.
- **Model Responsiveness**: The model forecasts future stock prices for the next 5 minutes, updating the chart in real-time.
- **Scalability**: The model’s architecture allows adjustments to predict further into the future or include more historical data points.

## 8. Conclusion
This Stock Price Forecasting App provides a visual, interactive, and real-time approach to stock price prediction, enabling investors to make data-informed decisions. The LSTM model captures temporal dependencies in stock price movements, making it suitable for short-term forecasts.

## 9. Future Work
- Experiment with larger models or additional layers to capture complex stock price trends.
- Extend the prediction window for longer-term forecasting.
- Integrate sentiment analysis from financial news or social media as additional features.

