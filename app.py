import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
CORS(app)

# List of stock symbols
stock_symbols = ["TATAMOTORS.NS", "INFY", "TECHM.NS", "HDB", "RELIANCE.NS"]

def get_stock_data(start_date, end_date, stock_symbol, min_trading_days=3):
    while True:
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
        if len(stock_data) >= min_trading_days:
            return stock_data, start_date, end_date
        end_date += timedelta(days=1)

@app.route('/')
def index():
    return render_template('index.html', stocks=stock_symbols)

@app.route('/get_stock_data', methods=['POST'])
def get_stock_data_route():
    data = request.json
    start_date = datetime.strptime(data['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(data['end_date'], '%Y-%m-%d')
    stock_symbol = data['stock_symbol']

    try:
        # Download historical data up to today
        historical_start = start_date - timedelta(days=365)  # Get one year of historical data for training
        today = datetime.now().date()
        stock_data, actual_start, actual_end = get_stock_data(historical_start, today, stock_symbol)

        # Prepare data for Prophet
        df = stock_data.reset_index()[["Date", "Close"]]
        df.columns = ["ds", "y"]

        # Create and fit the model
        model = Prophet()
        model.fit(df)

        # Make future dataframe for predictions
        future_days = max((end_date.date() - today).days, 0) + 1  # +1 to include the end date
        future = model.make_future_dataframe(periods=future_days)

        # Make predictions
        forecast = model.predict(future)

        # Generate plots
        fig1 = model.plot(forecast)
        plt.title(f"{stock_symbol} Stock Price Prediction")
        img1 = io.BytesIO()
        plt.savefig(img1, format='png')
        img1.seek(0)
        plot1 = base64.b64encode(img1.getvalue()).decode()
        plt.close(fig1)

        fig2 = model.plot_components(forecast)
        img2 = io.BytesIO()
        plt.savefig(img2, format='png')
        img2.seek(0)
        plot2 = base64.b64encode(img2.getvalue()).decode()
        plt.close(fig2)

        # Prepare combined data
        combined_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        combined_data.columns = ['Date', 'Close', 'Lower', 'Upper']
        
        # Filter the forecast to include both past data and future data
        combined_data = combined_data[(combined_data['Date'] >= pd.to_datetime(start_date)) & (combined_data['Date'] <= pd.to_datetime(end_date))]

        combined_data['Date'] = combined_data['Date'].dt.strftime('%Y-%m-%d')

        is_future_prediction = end_date.date() > today

        return jsonify({
            'data': combined_data.to_dict(orient='records'),
            'requested_start': start_date.strftime('%Y-%m-%d'),
            'requested_end': end_date.strftime('%Y-%m-%d'),
            'actual_start': actual_start.strftime('%Y-%m-%d'),
            'actual_end': actual_end.strftime('%Y-%m-%d'),
            'plot1': plot1,
            'plot2': plot2,
            'is_future_prediction': is_future_prediction
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
