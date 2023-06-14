import ccxt
import pandas as pd
import datetime
import time
import math

# Function to fetch OHLCV data
def fetch_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
    num_retries = 0
    try:
        num_retries += 1
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        return ohlcv
    except Exception:
        if num_retries > max_retries:
            raise  # Exception in last attempt

        # Sleep before retrying
        time.sleep(exchange.rateLimit / 1000)
        return fetch_ohlcv(exchange, max_retries, symbol, timeframe, since, limit)

# Binance instance
exchange = ccxt.binance({
    'apiKey': 'PWT9SEKurxBo2p0CMRFe7SOT5eO5KGWDvIeiazsBIUjW6s4KxwyKokRrrnsqoDQj',
    'secret': 'SPmWcX10izRNWlIDITvINjnqGocQ204FqkEZJgOFN7YHfzKsJTAtxfCmDTEn7BXV'
})

max_retries = 5  # Define the maximum retry attempts
symbol = 'BTC/USDT'  # Symbol to fetch
timeframe = '1h'  # Hourly timeframe
since = exchange.parse8601('2013-06-14T00:00:00Z')  # Start date for the BTC data
limit = 500  # Maximum number of results per request

all_data = []  # List to hold all data

# Fetch data in chunks
while True:
    data = fetch_ohlcv(exchange, max_retries, symbol, timeframe, since, limit)
    if len(data) > 0 and exchange.iso8601(since) < '2023-05-01T00:00:00Z':
        since = data[-1][0]  # Get timestamp of last loaded candle
        all_data += data  # Append new data
        print('Fetched', symbol, len(all_data), 'candles in total until', exchange.iso8601(since))
        # Rate limit
        time.sleep(exchange.rateLimit / 1000)
    else:
        break

df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['date'] = [datetime.datetime.fromtimestamp(x/1000) for x in df['timestamp']]
# After your dataframe is ready
df.to_csv('btc_hourly_data.csv', index=False)

print(df)
