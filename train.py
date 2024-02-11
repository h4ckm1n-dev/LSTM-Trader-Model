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
from sklearn.preprocessing import MinMaxScaler
import joblib  

# Load preprocessed and sequenced data from HDF5
def load_sequenced_data_from_h5(filepath):
    with h5py.File(filepath, 'r') as hf:
        X_train = hf['X_train'][:]
        y_train = hf['y_train'][:]
        X_test = hf['X_test'][:]
        y_test = hf['y_test'][:]
        future_prices_test = hf['future_prices_test'][:]
    return X_train, X_test, y_train, y_test, future_prices_test

# HyperModel class for keras-tuner
class LSTMHyperModel(kt.HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
    
    def build(self, hp):
        model = Sequential([
            LSTM(
                units=hp.Int('units', min_value=32, max_value=256, step=32),
                activation='relu',
                input_shape=self.input_shape,
                return_sequences=True,
                kernel_regularizer=l1_l2(
                    l1=hp.Float('l1', min_value=1e-5, max_value=1e-3, sampling='LOG'),
                    l2=hp.Float('l2', min_value=1e-4, max_value=1e-2, sampling='LOG')
                )
            ),
            Dropout(rate=hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.05)),
            BatchNormalization(),
            
            LSTM(
                units=hp.Int('units', min_value=32, max_value=256, step=32),
                activation='relu',
                return_sequences=False,
                kernel_regularizer=l1_l2(
                    l1=hp.Float('l1', min_value=1e-5, max_value=1e-3, sampling='LOG'),
                    l2=hp.Float('l2', min_value=1e-4, max_value=1e-2, sampling='LOG')
                )
            ),
            Dropout(rate=hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.05)),
            BatchNormalization(),

            Dense(
                units=hp.Int('dense_units', min_value=16, max_value=128, step=16),
                activation='relu',
                kernel_regularizer=l1_l2(
                    l1=hp.Float('l1', min_value=1e-5, max_value=1e-3, sampling='LOG'),
                    l2=hp.Float('l2', min_value=1e-4, max_value=1e-2, sampling='LOG')
                )
            ),
            Dropout(rate=hp.Float('dropout_3', min_value=0.2, max_value=0.5, step=0.05)),
            
            Dense(2, activation='softmax')  
        ])
        
        model.compile(
            optimizer=Adam(hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')),
            loss='categorical_crossentropy',  
            metrics=['accuracy']  
        )
        
        return model

def main():
    filepath = './data.h5'
    X_train, X_test, y_train, y_test, future_prices_test = load_sequenced_data_from_h5(filepath)
    
    # Convert y_train and y_test to represent classes (e.g., 0 for sell, 1 for buy)
    y_train_classes = np.where(y_train > 0, 1, 0)
    y_test_classes = np.where(y_test > 0, 1, 0)
    
    # Perform one-hot encoding if necessary
    y_train_encoded = tf.keras.utils.to_categorical(y_train_classes, num_classes=2)
    y_test_encoded = tf.keras.utils.to_categorical(y_test_classes, num_classes=2)
    
    # Initialize MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit the scaler to the training data
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))  # Reshape the training data for scaling
    
    # Save the scaler using joblib
    joblib.dump(scaler, './scaler.pkl')
    
    hypermodel = LSTMHyperModel(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',  
        patience=10,
        restore_best_weights=True
    )
    
    tuner = kt.RandomSearch(
        hypermodel,
        objective='val_accuracy',  
        max_trials=10,
        executions_per_trial=2,
        directory='keras_tuner_dir',
        project_name='lstm_hyper_tuning'
    )
    
    tuner.search(X_train, y_train_encoded, epochs=50, validation_split=0.1, callbacks=[early_stopping], verbose=1)
    
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters()[0]
    
    print(f"Best hyperparameters: {best_hyperparameters.values}")
    
    history = best_model.fit(
        X_train, y_train_encoded,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )
    
    test_loss, test_accuracy = best_model.evaluate(X_test, y_test_encoded, verbose=1)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    # Plotting comparison of actual and predicted future prices
    y_pred_probs = best_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)  

    # Convert continuous values to discrete classes
    y_test_classes_discrete = np.where(y_test_classes > 0, 1, 0)
    y_pred_classes_discrete = (y_pred_probs[:, 1] > 0.5).astype(int)

    # Plot actual future prices
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(future_prices_test)), future_prices_test, label='Actual Future Prices', color='blue')

    # Plot predicted signals
    for i in range(len(future_prices_test)):
        if y_pred_classes_discrete[i] == y_test_classes_discrete[i]:
            color = 'green'  # Correct prediction
        else:
            color = 'red'  # Incorrect prediction
        plt.scatter(i, future_prices_test[i], color=color)

    plt.title('Actual Future Prices with Predicted Buy/Sell Signals')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('actual_vs_predicted_prices_with_signals.png', dpi=300)

    history_df = pd.DataFrame(history.history)
    history_df.to_csv('./model_training_history.csv', index=False)
    
    test_results_df = pd.DataFrame({'Test Loss': [test_loss], 'Test Accuracy': [test_accuracy]})
    test_results_df.to_csv('./model_test_evaluation.csv', index=False)

    best_model.save('./trained_model.keras')

if __name__ == "__main__":
    main()
