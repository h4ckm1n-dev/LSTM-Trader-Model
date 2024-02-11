import ccxt
import csv
import os
from datetime import datetime, timedelta

# Initialize the Binance Exchange
binance = ccxt.binance({
    'enableRateLimit': True,  # important for Binance to avoid IP bans
})

def fetch_ohlcv_in_chunks(symbol, start_date, end_date, timeframe):
    all_data = []
    current_date = start_date
    while current_date < end_date:
        next_date = min(current_date + timedelta(days=365), end_date)  # Adjust the chunk size as needed
        since = binance.parse8601(current_date.strftime('%Y-%m-%d') + 'T00:00:00Z')
        data = binance.fetch_ohlcv(symbol, timeframe, since)
        all_data.extend(data)
        current_date = next_date
    return all_data

# Your desired date range and other parameters
start_date = datetime(2021, 1, 1)  # Example: starting from January 1, 2021
end_date = datetime.now()  # Up to the current date
symbol = 'BTC/USDT'
timeframe = '1d'

# Fetching the data
ohlcv_data = fetch_ohlcv_in_chunks(symbol, start_date, end_date, timeframe)

# Specify your CSV file path
csv_file_path = './bitcoin_price_data_extended.csv'

# Writing to CSV
with open(csv_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    for row in ohlcv_data:
        # Formatting the timestamp to a more readable format
        row[0] = datetime.utcfromtimestamp(row[0] / 1000).strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow(row)

print(f"Data successfully saved to {csv_file_path}")
