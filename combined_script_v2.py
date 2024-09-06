print("Script is running ...")
import os
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX  # Import SARIMAX
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 2: Fetch historical data for Silver (symbol: 'SI=F')
end_date = datetime.today().strftime('%Y-%m-%d')
silver_data = yf.download('SI=F', start='2018-01-01', end=end_date)

# Step 3: Process the data
silver_data['Date'] = silver_data.index
silver_data['Date'] = pd.to_datetime(silver_data['Date'])
silver_data.set_index('Date', inplace=True)
silver_data = silver_data.asfreq('D')
silver_data = silver_data.fillna(method='ffill')

# Split data into train and test sets
train_data = silver_data[:'2022']
test_data = silver_data['2023':]

# Exponential Smoothing (ETS) Model
ets_model = ExponentialSmoothing(train_data['Close'], trend='add', seasonal='add', seasonal_periods=365).fit()
ets_pred = ets_model.forecast(len(test_data))

# SARIMA Model
sarima_model = SARIMAX(train_data['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()
sarima_pred = sarima_model.predict(start=test_data.index[0], end=test_data.index[-1], dynamic=False)

# LSTM Model
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

scaler = MinMaxScaler(feature_range=(0,1))
train_data_scaled = scaler.fit_transform(train_data['Close'].values.reshape(-1, 1))

time_step = 100
X_train, y_train = create_dataset(train_data_scaled, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
lstm_model.add(LSTM(50, return_sequences=False))
lstm_model.add(Dense(25))
lstm_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train, y_train, batch_size=1, epochs=1)

test_data_scaled = scaler.transform(test_data['Close'].values.reshape(-1, 1))
X_test, y_test = create_dataset(test_data_scaled, time_step)

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
lstm_pred = lstm_model.predict(X_test)
lstm_pred = scaler.inverse_transform(lstm_pred)

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(test_data.index, test_data['Close'], label='Actual Price')
plt.plot(test_data.index, ets_pred, label='ETS Prediction')
plt.plot(test_data.index, sarima_pred, label='SARIMA Prediction')
plt.plot(test_data.index[time_step+1:], lstm_pred, label='LSTM Prediction')
plt.legend()

# Save the prediction plots to the specified directory with the current date in the filename
plt.savefig("/Users/spicy.kev/Documents/github/kspicer80.github.io/static/images/imgforblogposts/post_35/lstm_and_other_model_predictions.png")
current_date = datetime.today().strftime('%m_%d_%y')
filename = f'plots/lstm_and_other_models_plot{current_date}.png'
plt.savefig(filename)