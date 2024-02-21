import backtrader as bt
import numpy as np
import pandas as pd
from joblib import load, dump
from sklearn.preprocessing import RobustScaler
import tensorflow as tf

# Custom Indicator that uses the saved model and scalerclass 
class PredictiveIndicator(bt.Indicator):
    lines = ('signal',)
    params = (('period', 240), ('scaler', None), ('model', None), ('feature_names', None))

    def __init__(self):
        self.addminperiod(self.params.period)

    def next(self):
        size = self.params.period
        
        # Debugging: Print feature names
        print("Feature Names:", self.params.feature_names)
        
        values = np.empty((size, len(self.params.feature_names)))
        for i in range(size):
            for j, feat in enumerate(self.params.feature_names):
                values[i, j] = getattr(self.data, feat)[-i-1][0]

        if self.params.scaler:
            # Set feature_names_in_ to None to avoid the warning
            self.params.scaler.feature_names_in_ = None
            scaled_values = self.params.scaler.transform(values)
            scaled_values = scaled_values.reshape(1, size, len(self.params.feature_names))  # Reshape for the model
        else:
            scaled_values = values.reshape(1, size, len(self.params.feature_names))

        prediction = self.params.model.predict(scaled_values)
        self.lines.signal[0] = (prediction > 0.5) - (prediction <= 0.5)


# Custom Strategy that uses the PredictiveIndicator
class LSTMStrategy(bt.Strategy):
    def __init__(self):
        # Load scaler and model
        scaler = load('robust_scaler.joblib')
        model = tf.keras.models.load_model('best_lstm_model')
        feature_names = ['Close', 'trend_sma_fast', 'trend_sma_slow', 'momentum_rsi', 'trend_macd_diff', 
                         'volatility_bbm', 'volatility_bbh', 'volatility_bbl', 'ema50', 'ema200']
        self.predictive_signal = PredictiveIndicator(period=240, scaler=scaler, model=model, feature_names=feature_names)

    def next(self):
        if self.predictive_signal[0] == 1:
            self.buy(size=1)
        elif self.predictive_signal[0] == -1:
            self.sell(size=1)

# Load your data
# Note: Adjust this part to load your specific data
dataframe = pd.read_csv('./bitcoin_price_data_extended.csv', parse_dates=['Timestamp'])
dataframe.set_index('Timestamp', inplace=True)

# Split data into train and test sets
train_size = int(0.8 * len(dataframe))
train_data, test_data = dataframe[:train_size], dataframe[train_size:]

# Fit scaler only with the training data
scaler = RobustScaler()
scaler.fit(train_data)

# Save the scaler for later use
dump(scaler, 'robust_scaler.joblib')

# Convert the Pandas dataframe to a Backtrader data feed
data = bt.feeds.PandasData(dataname=train_data)  # Use only training data for fitting

# Set up Backtrader cerebro
cerebro = bt.Cerebro()
cerebro.addstrategy(LSTMStrategy)
cerebro.adddata(data)
cerebro.broker.setcash(100000.0)  # Initial cash
cerebro.addsizer(bt.sizers.FixedSize, stake=1)  # Stake size

# Run the backtest
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
cerebro.run()
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
