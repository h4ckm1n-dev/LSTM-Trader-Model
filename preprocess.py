import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from ta import add_all_ta_features
import h5py
import warnings

# Suppress warnings about invalid values encountered in scalar divide during technical indicator calculation
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

def dynamic_threshold(prices, window=30, multiplier=1.5):
    """Calculate a dynamic threshold based on recent volatility."""
    daily_returns = prices.pct_change()
    rolling_std = daily_returns.rolling(window=window).std()
    return rolling_std * multiplier

def create_sequences_with_signals(data, prices, seq_length, predict_ahead=1, dynamic_thresh=False):
    xs, ys, future_prices = [], [], []
    dynamic_thresh_values = dynamic_threshold(prices) if dynamic_thresh else None

    for i in range(len(data) - seq_length - predict_ahead):
        x = data[i:(i + seq_length)]
        current_price = prices[i + seq_length - 1]
        future_price = prices[i + seq_length + predict_ahead - 1]

        # Determine the signal based on price change
        price_change = (future_price - current_price) / current_price
        threshold = dynamic_thresh_values[i + seq_length - 1] if dynamic_thresh else 0.01
        if price_change > threshold:
            y = 1  # Long signal
        elif price_change < -threshold:
            y = 0  # Short signal
        else:
            continue  # Skip if the change is within the threshold

        xs.append(x)
        ys.append(y)
        future_prices.append(future_price)

    return np.array(xs), np.array(ys), np.array(future_prices)

# Load and preprocess the data
df = pd.read_csv('./bitcoin_price_data_extended.csv', parse_dates=['Timestamp'], index_col='Timestamp')

df['Daily_Returns'] = df['Close'].pct_change()
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

# Feature selection
features = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Returns',
    'momentum_rsi', 'trend_macd', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl'
]
df_selected = df[features].copy()

# Data splitting
train_size = int(len(df_selected) * 0.8)
df_train, df_test = df_selected.iloc[:train_size], df_selected.iloc[train_size:]

# Data normalization
scaler = RobustScaler()
df_train_scaled = scaler.fit_transform(df_train)
df_test_scaled = scaler.transform(df_test)

# Generate sequences with signals
seq_length = 60
predict_ahead = 3
dynamic_thresh = True  # Use dynamic threshold based on recent volatility

X_train, y_train, future_prices_train = create_sequences_with_signals(df_train_scaled, df_train['Close'].values, seq_length, predict_ahead, dynamic_thresh)
X_test, y_test, future_prices_test = create_sequences_with_signals(df_test_scaled, df_test['Close'].values, seq_length, predict_ahead, dynamic_thresh)

# Save processed data to HDF5
with h5py.File('./data.h5', 'w') as hf:
    hf.create_dataset('X_train', data=X_train)
    hf.create_dataset('y_train', data=y_train)
    hf.create_dataset('X_test', data=X_test)
    hf.create_dataset('y_test', data=y_test)
    hf.create_dataset('future_prices_train', data=future_prices_train)
    hf.create_dataset('future_prices_test', data=future_prices_test)
    print("HDF5 File saved successfully.")
