import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import joblib

def load_sequenced_data_from_h5(filepath):
    with h5py.File(filepath, 'r') as hf:
        X_train = hf['X_train'][:]
        y_train = hf['y_train'][:]
        X_test = hf['X_test'][:]
        y_test = hf['y_test'][:]
        future_prices_test = hf['future_prices_test'][:]
    return X_train, X_test, y_train, y_test, future_prices_test

class LSTMHyperModel(kt.HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential([
            LSTM(units=hp.Int('units', min_value=32, max_value=256, step=32),
                 activation='relu',
                 input_shape=self.input_shape,
                 return_sequences=True,
                 kernel_regularizer=l1_l2(l1=hp.Float('l1', min_value=1e-5, max_value=1e-3, sampling='LOG'),
                                          l2=hp.Float('l2', min_value=1e-4, max_value=1e-2, sampling='LOG'))),
            Dropout(rate=hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.05)),
            BatchNormalization(),

            LSTM(units=hp.Int('units', min_value=32, max_value=256, step=32),
                 activation='relu',
                 return_sequences=False,
                 kernel_regularizer=l1_l2(l1=hp.Float('l1', min_value=1e-5, max_value=1e-3, sampling='LOG'),
                                          l2=hp.Float('l2', min_value=1e-4, max_value=1e-2, sampling='LOG'))),
            Dropout(rate=hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.05)),
            BatchNormalization(),

            Dense(units=hp.Int('dense_units', min_value=16, max_value=128, step=16),
                  activation='relu',
                  kernel_regularizer=l1_l2(l1=hp.Float('l1', min_value=1e-5, max_value=1e-3, sampling='LOG'),
                                           l2=hp.Float('l2', min_value=1e-4, max_value=1e-2, sampling='LOG'))),
            Dropout(rate=hp.Float('dropout_3', min_value=0.2, max_value=0.5, step=0.05)),

            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=Adam(hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

def main():
    filepath = './data.h5'
    X_train, X_test, y_train, y_test, future_prices_test = load_sequenced_data_from_h5(filepath)

    kf = KFold(n_splits=5, shuffle=True)

    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        hypermodel = LSTMHyperModel(input_shape=(X_train_fold.shape[1], X_train_fold.shape[2]))

        early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

        tuner = kt.RandomSearch(hypermodel,
                                objective='val_accuracy',
                                max_trials=10,
                                executions_per_trial=2,
                                directory='keras_tuner_dir',
                                project_name='lstm_hyper_tuning')

        tuner.search(X_train_fold, y_train_fold, epochs=50, validation_data=(X_val_fold, y_val_fold), callbacks=[early_stopping], verbose=1)

        best_model = tuner.get_best_models(num_models=1)[0]
        best_hyperparameters = tuner.get_best_hyperparameters()[0]

        print(f"Best hyperparameters: {best_hyperparameters.values}")

        history = best_model.fit(X_train_fold, y_train_fold,
                                 epochs=100,
                                 batch_size=32,
                                 validation_data=(X_val_fold, y_val_fold),
                                 callbacks=[early_stopping],
                                 verbose=1)

        test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=1)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        y_pred_probs = best_model.predict(X_test)
        y_pred_classes = (y_pred_probs > 0.5).astype(int)

        # Calculate and print long & short proportion
        long_signals = np.sum(y_pred_classes == 0)
        short_signals = np.sum(y_pred_classes == 1)
        total_signals = long_signals + short_signals
        print(f"Long Signals: {long_signals} ({long_signals / total_signals * 100:.2f}%)")
        print(f"Short Signals: {short_signals} ({short_signals / total_signals * 100:.2f}%)")

        plt.figure(figsize=(12, 7))
        plt.plot(range(len(future_prices_test)), future_prices_test, label='Actual Future Prices', color='blue', alpha=0.6)

        # Initialize lists to collect legend information
        markers = {'Long Correct': ('^', 'green'), 'Short Correct': ('v', 'green'),
                   'Long Incorrect': ('^', 'red'), 'Short Incorrect': ('v', 'red')}
        for label, (marker, color) in markers.items():
            plt.scatter([], [], color=color, marker=marker, label=label, s=100, alpha=0.7)

        for i, (pred, actual, prob) in enumerate(zip(y_pred_classes.flatten(), y_test, y_pred_probs.flatten())):
            # Determine marker and color based on prediction correctness and type
            is_correct = pred == actual
            marker = '^' if pred == 1 else 'v'
            color = 'green' if is_correct else 'red'
            size = 100 * prob  # Scale size by confidence (optional)
            plt.scatter(i, future_prices_test[i], color=color, alpha=0.7, marker=marker, s=size)

        plt.title('Actual Future Prices with Predicted Buy/Sell Signals')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend(loc='best', fontsize=8)
        plt.tight_layout()
        plt.savefig('result.png', dpi=300)
        
        history_df = pd.DataFrame(history.history)
        history_df.to_csv('./model_training_history.csv', index=False)

        test_results_df = pd.DataFrame({'Test Loss': [test_loss], 'Test Accuracy': [test_accuracy]})
        test_results_df.to_csv('./model_test_evaluation.csv', index=False)

        best_model.save('./trained_model.keras')

if __name__ == "__main__":
    main()
