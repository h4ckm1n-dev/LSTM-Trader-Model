import backtrader as bt
import numpy as np
import h5py
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib

# Assuming the model and scaler are trained with the specifics you provided
model_file = './trained_model.keras'
scaler_file = './scaler.pkl'

# Load the trained LSTM model
model = tf.keras.models.load_model(model_file)

# Load the scaler used during preprocessing
scaler = joblib.load(scaler_file)

class MyStrategy(bt.Strategy):
    params = (
        ('model', model),
        ('scaler', scaler),
        ('sequence_length', 60),  # Match this with your training setup
        ('predict_ahead', 1),     # Match this with your training setup
        ('features', ['feature1', 'feature2', 'featureN']),  # Specify the exact features used
    )

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def __init__(self):
        self.dataclose = self.datas[0].close    # Keep a reference to the "close" line in the data[0] dataseries
        self.order = None                       # To keep track of pending orders

    def next(self):
        # Skip the loop if an order is pending
        if self.order:
            return

        # Prepare the data array for prediction
        if len(self.data) >= self.params.sequence_length + self.params.predict_ahead:
            data_window = np.array([[
                self.datas[0].close.get(size=self.params.sequence_length),
                # Add other feature lines as needed, matching the training feature set
            ]])
            data_window = data_window.reshape(self.params.sequence_length, len(self.params.features))
            scaled_data = self.params.scaler.transform(data_window)
            scaled_data = scaled_data.reshape(1, self.params.sequence_length, len(self.params.features))  # Reshape for the LSTM

            # Predict the signal: buy (1), sell (0)
            predicted_signal = np.argmax(self.params.model.predict(scaled_data), axis=-1)

            # Execute orders
            if predicted_signal == 1 and not self.position:  # Buy signal and we don't hold a position
                self.order = self.buy()
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
            elif predicted_signal == 0 and self.position:    # Sell signal and we hold a position
                self.order = self.sell()
                self.log('SELL CREATE, %.2f' % self.dataclose[0])

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Order has been submitted/accepted - no further action required
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price}, Cost: {order.executed.value}, Comm: {order.executed.comm}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price}, Cost: {order.executed.value}, Comm: {order.executed.comm}')
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Reset orders
        self.order = None

    def stop(self):
        self.log(f'Ending Value: {self.broker.getvalue()}, Cash: {self.broker.getcash()}')
