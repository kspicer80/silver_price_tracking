print("Script is running ...")
# Step 1: Import necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.linear_model import LinearRegression
import numpy as np

# Step 2: Fetch historical data for Silver (symbol: 'SI=F')
end_date = datetime.today().strftime('%Y-%m-%d')
silver_data = yf.download('SI=F', start='2018-01-01', end=end_date)

# Step 3: Process the data
# Display the first few rows of the data
print(silver_data.head())

# ARIMA Model
# Step 4: Fit ARIMA model
# Use the 'Close' prices for modeling
silver_prices = silver_data['Close']

# Define the ARIMA model
arima_model = ARIMA(silver_prices, order=(5, 1, 0))
arima_model_fit = arima_model.fit()

# Step 5: Make predictions with ARIMA
# Forecast the next 180 days (approximately 6 months)
arima_forecast = arima_model_fit.forecast(steps=180)
arima_forecast_dates = pd.date_range(start=silver_prices.index[-1], periods=180, freq='B')

# Prophet Model
# Prepare data for Prophet
silver_data.reset_index(inplace=True)
silver_data = silver_data[['Date', 'Close']]
silver_data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

# Step 4: Fit Prophet model
prophet_model = Prophet()
prophet_model.fit(silver_data)

# Step 5: Make predictions with Prophet
# Forecast the next 180 days (approximately 6 months)
future = prophet_model.make_future_dataframe(periods=180)
prophet_forecast = prophet_model.predict(future)

# Linear Regression Model
# Prepare data for Linear Regression
silver_data['Date_ordinal'] = pd.to_datetime(silver_data['ds']).map(datetime.toordinal)
X = silver_data[['Date_ordinal']]
y = silver_data['y']

# Fit Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X, y)

# Make predictions with Linear Regression
future_dates = pd.date_range(start=silver_data['ds'].iloc[-1], periods=180, freq='B')
future_dates_ordinal = future_dates.map(datetime.toordinal).values.reshape(-1, 1)
linear_forecast = linear_model.predict(future_dates_ordinal)

# Step 6: Visualize the data and predictions
plt.figure(figsize=(8, 11))  # Adjust the figure size to 8x11 inches

# Ensure the 'plots' directory exists
os.makedirs('plots', exist_ok=True)

# Plot ARIMA predictions
plt.subplot(3, 1, 1)
plt.plot(silver_prices, label='Historical Silver Prices')
plt.plot(arima_forecast_dates, arima_forecast, label='Predicted Silver Prices Using ARIMA Model', color='red')
plt.title('Silver Price Prediction Using ARIMA Model')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)

# Plot Prophet predictions
plt.subplot(3, 1, 2)
plt.plot(silver_data['ds'], silver_data['y'], label='Historical Silver Prices')
plt.plot(prophet_forecast['ds'], prophet_forecast['yhat'], label="Predicted Silver Prices Using Facebook's Prophet Model", color='red')
plt.title("Silver Price Prediction Using Facebook's Prophet Model")
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)

# Plot Linear Regression predictions
plt.subplot(3, 1, 3)
plt.plot(silver_data['ds'], silver_data['y'], label='Historical Silver Prices')
plt.plot(future_dates, linear_forecast, label='Predicted Silver Prices Using Linear Regression Model', color='red')
plt.title('Silver Price Prediction Using Linear Regression Model')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)

# Adjust layout and spacing for the plot with subplots
plt.tight_layout()
plt.subplots_adjust(hspace=1.0)  # Increase the space between subplots

# Save the prediction plots to the specified directory with the current date in the filename
plt.savefig("/Users/spicy.kev/Documents/github/kspicer80.github.io/static/images/imgforblogposts/post_35/all_models_silver_prices.png")
current_date = datetime.today().strftime('%m_%d_%y')
filename = f'plots/all_models_silver_prices_{current_date}.png'
plt.savefig(filename)

# Show the prediction plots
#plt.show()

# Plot historical prices for the last couple of months in a new figure
last_couple_months = silver_data[silver_data['ds'] >= (datetime.today() - timedelta(days=60))]
plt.figure(figsize=(8, 6))  # Adjust the figure size to 8x6 inches
plt.plot(last_couple_months['ds'], last_couple_months['y'], color='magenta', label='Historical Silver Prices (Last 2 Months)')
plt.title('Historical Silver Prices (Last 2 Months)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=90) 

# Save the last couple of months plot to the specified directory with the current date in the filename
filename_last_couple_months = f'/Users/spicy.kev/Documents/github/kspicer80.github.io/static/images/imgforblogposts/post_35/last_couple_of_months_plot.png'
plt.tight_layout()
plt.savefig(filename_last_couple_months)

# Show the last couple of months' plot
#plt.show()
print("Script has completed ...")