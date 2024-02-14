import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from ta import add_all_ta_features
import h5py
import warnings
from joblib import dump

# Suppress warnings about invalid values encountered in scalar divide during technical indicator calculation
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

def dynamic_threshold(prices, window=30, multiplier=1.5):
    """Calculate a dynamic threshold based on recent volatility."""
    # Calculate daily returns as percentage changes
    daily_returns = np.diff(prices) / prices[:-1]
    # Calculate rolling standard deviation
    rolling_std = pd.Series(daily_returns).rolling(window=window).std().to_numpy()
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

    print(f"Generated {len(xs)} sequences with signals.")
    return np.array(xs), np.array(ys), np.array(future_prices)

# Load and preprocess the data
df = pd.read_csv('./bitcoin_price_data_extended.csv', parse_dates=['Timestamp'], index_col='Timestamp')
print("Data loaded successfully.\n", df.head())

df['Daily_Returns'] = df['Close'].pct_change()
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)
print("Data cleaned and missing values handled.\n", df.head())

df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
print("Technical indicators added.\n", df.head())

# Calculate Exponential Moving Averages and add to the DataFrame
df['ema10'] = df['Close'].ewm(span=10, adjust=False).mean()
df['ema50'] = df['Close'].ewm(span=50, adjust=False).mean()
df['ema200'] = df['Close'].ewm(span=200, adjust=False).mean()

features = [
    'Close',  # Primary focus on closing price for simplicity
    'Volume',  # Indicator of trading activity
    'momentum_rsi',  # Momentum indicator
    'trend_macd',  # Trend indicator
    'volatility_bbm', 'volatility_bbh', 'volatility_bbl',  # Volatility indicators
    'ema50', 'ema200',  # Exponential Moving Averages for trend analysis
]


df_selected = df[features].copy()
print("Features selected.\n", df_selected.head())

# Data splitting
train_size = int(len(df_selected) * 0.8)
df_train, df_test = df_selected.iloc[:train_size], df_selected.iloc[train_size:]
print(f"Data split into training and testing sets.\nTraining set shape: {df_train.shape}\nTesting set shape: {df_test.shape}")

# Data normalization
scaler = RobustScaler()
df_train_scaled = scaler.fit_transform(df_train)
df_test_scaled = scaler.transform(df_test)
print("Data normalized.\nSample of scaled training data:\n", df_train_scaled[:5])

# Save the scaler for later use
dump(scaler, 'robust_scaler.joblib')

# Generate sequences with signals
seq_length = 60
predict_ahead = 3
dynamic_thresh = True  # Use dynamic threshold based on recent volatility

X_train, y_train, future_prices_train = create_sequences_with_signals(df_train_scaled, df_train['Close'].values, seq_length, predict_ahead, dynamic_thresh)
X_test, y_test, future_prices_test = create_sequences_with_signals(df_test_scaled, df_test['Close'].values, seq_length, predict_ahead, dynamic_thresh)
print("Sequences with signals generated for training and testing.")

# Save processed data to HDF5
with h5py.File('./data.h5', 'w') as hf:
    hf.create_dataset('X_train', data=X_train)
    hf.create_dataset('y_train', data=y_train)
    hf.create_dataset('X_test', data=X_test)
    hf.create_dataset('y_test', data=y_test)
    hf.create_dataset('future_prices_train', data=future_prices_train)
    hf.create_dataset('future_prices_test', data=future_prices_test)
    print("HDF5 File saved successfully.")
