# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.linear_model import LinearRegression
# import alpha_vantage.timeseries as av

# from dotenv import load_dotenv
# import os
# alpha_vantage_api = os.getenv('alpha_vantage_api')

# app = FastAPI()

# class StockIn(BaseModel):
#     company_name: str
#     start_date: str
#     end_date: str

# class StockOut(BaseModel):
#     company_name: str
#     start_date: str
#     end_date: str
#     predicted_prices: list[float]  # Update the type to list[float]

# # Define the companies and their corresponding Alpha Vantage symbols
# companies = {
#     "Apple": "AAPL",
#     "Microsoft": "MSFT",
#     "Amazon": "AMZN",
#     "Alphabet": "GOOGL",
#     "Facebook": "FB"
# }

# # Download the stock price data for each company
# stock_data = {}
# for company, symbol in companies.items():
#     ts = av.TimeSeries(key=alpha_vantage_api, output_format='pandas')
#     data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
#     data.rename(columns={'date': 'Date'}, inplace=True)  # Rename the date column
#     stock_data[company] = data

# # Create and fit the model for each company
# models = {}
# for company, data in stock_data.items():
#     scaler = MinMaxScaler()
#     data['Close'] = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
#     model = LinearRegression()
#     model.fit(data[['Open']], data['Close'])  # Use 'Open' as the feature
#     models[company] = (scaler, model)

# @app.post("/predict", response_model=StockOut)
# async def predict_stock_price(stock_in: StockIn):
#     company_name = stock_in.company_name
#     start_date = stock_in.start_date
#     end_date = stock_in.end_date

#     # Filter the data by date range
#     data = stock_data[company_name]
#     data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

#     # Make predictions
#     scaler, model = models[company_name]
#     predicted_prices = model.predict(data[['Open']])

#     # Invert predictions
#     predicted_prices = scaler.inverse_transform(predicted_prices)

#     # Return the predicted prices
#     return StockOut(
#         company_name=company_name,
#         start_date=start_date,
#         end_date=end_date,
#         predicted_prices=predicted_prices.flatten().tolist()  # Flatten and convert to list
#     )



import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime

# List of stock symbols
stock_symbols = ["TATAMOTORS.NS", "INFY", "TECHM.NS", "HDB", "RELIANCE.NS"]

start_date = '2018-01-01'
end_date = datetime.now()

for ticker in stock_symbols:
    print(f"\nProcessing {ticker}...")
    
    # Download stock data
    data = yf.download(ticker, start=start_date, end=end_date)

    # Prepare data for Prophet
    df = data.reset_index()[["Date", "Close"]]
    df.columns = ["ds", "y"]

    # Create and fit the model
    model = Prophet()
    model.fit(df)

    # Make future dataframe for predictions
    future = model.make_future_dataframe(periods=365)  # Predict 365 days into the future

    # Make predictions
    forecast = model.predict(future)

    # Plot the results
    fig1 = model.plot(forecast)
    plt.title(f"{ticker} Stock Price Prediction")
    plt.show()

    # Plot the components of the forecast
    fig2 = model.plot_components(forecast)
    plt.show()

    # Print the predicted values for the next 30 days
    print(f"\nPredicted values for {ticker} (next 30 days):")
    print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(30))

    # Calculate RMSE for the historical period
    historical_forecast = forecast[forecast['ds'] <= df['ds'].max()]
    historical_truth = df.merge(historical_forecast[['ds', 'yhat']], on='ds', how='left')
    rmse = ((historical_truth['y'] - historical_truth['yhat']) ** 2).mean() ** 0.5
    print(f"\nRMSE for {ticker}: {rmse}")

    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(df['ds'], df['y'], label='Actual')
    plt.plot(historical_forecast['ds'], historical_forecast['yhat'], label='Predicted')
    plt.title(f"{ticker} Actual vs Predicted Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()

    print("\n" + "="*50)  # Separator between stocks