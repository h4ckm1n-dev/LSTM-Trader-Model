import pandas as pd
import warnings
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features  # Ensure the 'ta' library is installed
import h5py

# Load the data
df = pd.read_csv('./bitcoin_price_data_extended.csv', parse_dates=['Timestamp'], index_col='Timestamp')
print("Initial Data Loaded:")
print(df.head())  # Show the first few rows of the loaded data

# Preprocess and add technical indicators
df['Daily_Returns'] = df['Close'].pct_change()
# Use forward fill for missing values, then backfill as a secondary measure
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)
print("\nData after filling missing values:")
print(df.head())  # Check data after forward and backward fill

# Suppressing the warning
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
print("\nData after adding technical indicators:")
print(df.head())  # Check data after adding technical indicators

# Refine feature selection
features = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Returns',
    'momentum_rsi', 'trend_macd', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl'
]
df_selected = df[features].copy()
print("\nSelected Features:")
print(df_selected.head())  # Show the first few rows after feature selection

# Split the data before normalization to prevent data leakage
train_size = int(len(df_selected) * 0.8)
df_train, df_test = df_selected.iloc[:train_size], df_selected.iloc[train_size:]
print("\nTraining Data Shape:", df_train.shape)
print("Test Data Shape:", df_test.shape)

# Normalization using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df_train_scaled = scaler.fit_transform(df_train)
df_test_scaled = scaler.transform(df_test)  # Apply the same transformation to the test data
print("\nFirst 5 rows of scaled training data:")
print(df_train_scaled[:5])  # Show the first few rows of scaled training data

inverse_scaled_data = scaler.inverse_transform(df_train_scaled)
print("\nCheck inverse scaling on the first 5 rows of training data:")
print(inverse_scaled_data[:5])

# Adjust sequence creation for predictive modeling
def create_sequences_with_future_price(data, prices, seq_length, predict_ahead=1):
    xs, ys, future_prices = [], [], []
    for i in range(len(data) - seq_length - predict_ahead):
        x = data[i:(i + seq_length)]
        y = prices[i + seq_length + predict_ahead - 1]
        future_price = prices[i + seq_length + predict_ahead] if (i + seq_length + predict_ahead) < len(prices) else np.nan
        xs.append(x)
        ys.append(y)
        future_prices.append(future_price)
    return np.array(xs), np.array(ys), np.array(future_prices)

seq_length = 60
predict_ahead = 3  # Adjusted for clearer future price prediction
X_train, y_train, future_prices_train = create_sequences_with_future_price(df_train_scaled, df_train['Close'].values, seq_length, predict_ahead)
X_test, y_test, future_prices_test = create_sequences_with_future_price(df_test_scaled, df_test['Close'].values, seq_length, predict_ahead)

# Save data
with h5py.File('/home/h4ckm1n/OpenInterpreter/data.h5', 'w') as hf:
    hf.create_dataset('X_train', data=X_train)
    hf.create_dataset('y_train', data=y_train)
    hf.create_dataset('X_test', data=X_test)
    hf.create_dataset('y_test', data=y_test)
    # Future prices can be useful for evaluation but are not used during training
    hf.create_dataset('future_prices_train', data=future_prices_train)
    hf.create_dataset('future_prices_test', data=future_prices_test)
    print("HDF5 File saved successfully.")
