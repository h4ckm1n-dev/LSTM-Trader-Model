import backtrader as bt
import numpy as np
from joblib import load
import tensorflow as tf
import pandas as pd


# Custom Indicator that uses the saved model and scaler
class PredictiveIndicator(bt.Indicator):
    lines = ('signal',)
    params = (('period', 60), ('scaler', None), ('model', None))

    def __init__(self):
        self.addminperiod(self.params.period)
        # Scaler and model are set via params

    def next(self):
        size = self.params.period
        values = np.array([self.data.close[-i] for i in range(size, 0, -1)])
        values = values.reshape(-1, 1)  # Reshape for the scaler

        scaled_values = self.params.scaler.transform(values)
        scaled_values = scaled_values.reshape(1, size, 1)  # Reshape for the model
        
        prediction = self.params.model.predict(scaled_values)
        self.lines.signal[0] = (prediction > 0.5) - (prediction <= 0.5)

# Custom Strategy that uses the PredictiveIndicator
class LSTMStrategy(bt.Strategy):
    def __init__(self):
        self.predictive_signal = PredictiveIndicator(period=60, 
                                                      scaler=load('robust_scaler.joblib'), 
                                                      model=tf.keras.models.load_model('best_lstm_model'))

    def next(self):
        if self.predictive_signal[0] == 1:
            self.buy(size=1)
        elif self.predictive_signal[0] == -1:
            self.sell(size=1)

# Load your data
# Note: Adjust this part to load your specific data
dataframe = pd.read_csv('./bitcoin_price_data_extended.csv', parse_dates=['Timestamp'])
dataframe.set_index('Timestamp', inplace=True)

# Convert the Pandas dataframe to a Backtrader data feed
data = bt.feeds.PandasData(dataname=dataframe)

# Set up Backtrader cerebro
cerebro = bt.Cerebro()
cerebro.addstrategy(LSTMStrategy)
cerebro.adddata(data)
cerebro.broker.setcash(100000.0)  # Initial cash
cerebro.addsizer(bt.sizers.FixedSize, stake=10)  # Stake size

# Run the backtest
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
cerebro.run()
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
