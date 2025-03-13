import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from math import sqrt

def forecast(data, frequency):

    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Resample data hourly and sum USDT flows
    data = data.resample(frequency).sum()

    # Split the data into train and test sets (80% train, 20% test)
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]

    # Rolling window predictions
    rolling_predictions = []
    history = list(train['USDT'])

    for t in range(len(test)):
        # Fit model for each rolling window
        model = ExponentialSmoothing(history, trend='add', seasonal='add', seasonal_periods=24)
        fit = model.fit()
        pred = fit.forecast(steps=1)[0]
        rolling_predictions.append(pred)
        history.append(test['USDT'].iloc[t])

    # Calculate RMSE for rolling predictions
    rmse = sqrt(mean_squared_error(test['USDT'], rolling_predictions))
    print(f'Rolling Test RMSE: {rmse:.2f}')

    # Forecast for the next 1 day
    model_final = ExponentialSmoothing(train['USDT'], trend='add', seasonal='add', seasonal_periods=7)
    fit_final = model_final.fit()
    forecast_24h = fit_final.forecast(steps=1)

    # Ensure no negative values in the forecast
    forecast_24h = forecast_24h.apply(lambda x: max(x, 0))

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(train['USDT'], label='Train')
    plt.plot(test['USDT'], label='Test')
    plt.plot(test.index, rolling_predictions, label='Rolling Predictions', color='orange')
    plt.plot(forecast_24h, label='1-Day Forecast', color='green')
    plt.xlabel('Date')
    plt.ylabel('USDT Flow')
    plt.title('USDT Flow Forecasting with Rolling Predictions')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print the 1-day forecast values with 95% confidence intervals
    forecast_df = pd.DataFrame({'Forecast': forecast_24h})
    forecast_df['Lower Bound'] = forecast_df['Forecast'] - 1.96 * forecast_df['Forecast'].std()
    forecast_df['Upper Bound'] = forecast_df['Forecast'] + 1.96 * forecast_df['Forecast'].std()

    # Ensure no negative values in confidence intervals
    forecast_df['Lower Bound'] = forecast_df['Lower Bound'].apply(lambda x: max(x, 0))
    forecast_df['Upper Bound'] = forecast_df['Upper Bound'].apply(lambda x: max(x, 0))

    print(forecast_df)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(train['USDT'], label='Train')
    plt.plot(test['USDT'], label='Test')
    plt.plot(test.index, rolling_predictions, label='Rolling Predictions', color='orange')
    plt.plot(forecast_24h, label='1-Day Forecast', color='green')

    # Add shaded area for confidence intervals
    plt.fill_between(
        forecast_df.index,
        forecast_df['Lower Bound'],
        forecast_df['Upper Bound'],
        color='red',
        alpha=0.3,
        label='95% Confidence Interval'
    )

    plt.xlabel('Date')
    plt.ylabel('USDT Flow')
    plt.title('USDT Flow Forecasting with Rolling Predictions')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Save the forecast results to a CSV file
    forecast_df.to_csv('usdt_1day_forecast.csv')
