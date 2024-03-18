import pandas as pd
import numpy as np
import glob  # For loading all CSV files in a directory
from sklearn.preprocessing import RobustScaler
from ta import add_all_ta_features
from keras.utils import to_categorical
import h5py
import warnings
from joblib import dump

# Suppress specific warnings
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")
# Suppress the specific FutureWarning related to the 'ta' library
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Series.__setitem__ treating keys as positions is deprecated.*",
)


def dynamic_threshold(prices, vix_values, window=30, base_multiplier=1.5):
    """
    Calculate a dynamic threshold based on recent volatility and market conditions.
    Adjusts the threshold multiplier based on VIX levels.
    """
    if prices is None or len(prices) == 0 or vix_values is None or len(vix_values) == 0:
        return np.array([])
    daily_returns = np.diff(prices) / prices[:-1]
    rolling_std = pd.Series(daily_returns).rolling(window=window).std().to_numpy()
    adjusted_multiplier = base_multiplier * (
        1 + (vix_values - vix_values.mean()) / vix_values.std()
    )
    return rolling_std * adjusted_multiplier


def create_sequences_with_signals(
    data, prices, vix_values, seq_length, predict_ahead=1, dynamic_thresh=False
):
    xs, ys, future_prices = [], [], []
    dynamic_thresh_values = (
        dynamic_threshold(prices, vix_values) if dynamic_thresh else None
    )

    # Ensure dynamic_thresh_values is not None and is subscriptable
    if dynamic_thresh_values is None:
        dynamic_thresh_values = [0.01] * (
            len(prices) - seq_length - predict_ahead
        )  # Default threshold

    for i in range(len(data) - seq_length - predict_ahead):
        x = data[i : (i + seq_length)]
        current_price = prices[i + seq_length - 1]
        future_price = prices[i + seq_length + predict_ahead - 1]
        price_change = (future_price - current_price) / current_price

        # Safely access dynamic_thresh_values with a default fallback
        threshold = dynamic_thresh_values[i] if dynamic_thresh else 0.01

        if price_change > threshold:
            y = 1  # Long signal
        elif price_change < -threshold:
            y = 0  # Short signal
        else:
            y = 2  # Hold signal

        xs.append(x)
        ys.append([y])
        future_prices.append(future_price)

    print(f"Generated {len(xs)} sequences with signals.")
    return np.array(xs), np.array(ys).flatten(), np.array(future_prices)


# Adjust this path to where your CSV files are stored
path_to_csv_files = "./market-data/*.csv"  # Example: './data/*.csv'

# Use glob to list all CSV files in the directory
csv_files = glob.glob(path_to_csv_files)

# Load and concatenate all CSV files into a single DataFrame
df_list = []
for file in csv_files:
    df_temp = pd.read_csv(file, parse_dates=["Timestamp"], index_col="Timestamp")
    df_list.append(df_temp)

df = pd.concat(df_list)
print("Data from all CSV files loaded and concatenated successfully.\n", df.head())

df["Daily_Returns"] = df["Close"].pct_change()
df.ffill(inplace=True)  # Forward fill to handle initial missing values
df.bfill(inplace=True)  # Backward fill to handle remaining missing values
print("Data cleaned and missing values handled.\n", df.head())

# Add technical indicators
df = add_all_ta_features(
    df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
)
print("Technical indicators added.\n", df.head())

# Calculate and add Exponential Moving Averages
df["ema10"] = df["Close"].ewm(span=10, adjust=False).mean()
df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()
df["ema200"] = df["Close"].ewm(span=200, adjust=False).mean()

# Select features for the model
features = [
    "Close",
    "trend_sma_fast",
    "trend_sma_slow",
    "momentum_rsi",
    "trend_macd_diff",
    "volatility_bbm",
    "volatility_bbh",
    "volatility_bbl",
    "ema50",
    "ema200",
]
df_selected = df[features].copy()
print("Features selected.\n", df_selected.head())

# Data splitting
train_size = int(len(df_selected) * 0.8)
df_train, df_test = df_selected.iloc[:train_size], df_selected.iloc[train_size:]
print(
    f"Data split into training and testing sets.\nTraining set shape: {df_train.shape}\nTesting set shape: {df_test.shape}"
)

# Normalize data using the RobustScaler and save it for later use
scaler = RobustScaler()
df_train_scaled = scaler.fit_transform(df_train)
df_test_scaled = scaler.transform(df_test)
dump(scaler, "robust_scaler.joblib")

# Generate sequences with signals for both training and testing sets using the create_sequences_with_signals function
seq_length = 240
predict_ahead = 1
dynamic_thresh = True

# Generate sequences with signals for both training and testing sets
X_train, y_train, future_prices_train = create_sequences_with_signals(
    df_train_scaled, df_train["Close"].values, seq_length, predict_ahead, dynamic_thresh
)
X_test, y_test, future_prices_test = create_sequences_with_signals(
    df_test_scaled, df_test["Close"].values, seq_length, predict_ahead, dynamic_thresh
)

# Correctly apply to_categorical now that y_train and y_test are properly flattened
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

# Ensure data is saved correctly
with h5py.File("./data.h5", "w") as hf:
    hf.create_dataset("X_train", data=X_train)
    hf.create_dataset("y_train", data=y_train_categorical)
    hf.create_dataset("X_test", data=X_test)
    hf.create_dataset("y_test", data=y_test_categorical)
    hf.create_dataset("future_prices_train", data=future_prices_train)
    hf.create_dataset("future_prices_test", data=future_prices_test)
    print("HDF5 File saved successfully.")
