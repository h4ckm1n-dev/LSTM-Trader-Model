import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall, AUC  # Corrected import
import keras_tuner as kt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def load_sequenced_data_from_h5(filepath):
    with h5py.File(filepath, 'r') as hf:
        X_train = hf['X_train'][:]
        y_train = hf['y_train'][:]
        X_test = hf['X_test'][:]
        y_test = hf['y_test'][:]
    return X_train, X_test, y_train, y_test

class LSTMHyperModel(kt.HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = Sequential([
            Bidirectional(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
                               activation='tanh',
                               input_shape=self.input_shape,
                               return_sequences=True,
                               kernel_regularizer=l1_l2(l1=hp.Float('l1', min_value=1e-5, max_value=1e-2, sampling='log'),  # Adjusted min_value
                                                        l2=hp.Float('l2', min_value=1e-5, max_value=1e-2, sampling='log')))),  # Adjusted min_value
            BatchNormalization(),
            Dropout(rate=hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.05)),
            LSTM(units=hp.Int('units_l2', min_value=32, max_value=512, step=32),
                 activation='tanh',
                 return_sequences=False,
                 kernel_regularizer=l1_l2(l1=hp.Float('l1_l2', min_value=1e-5, max_value=1e-3, sampling='log'),  # Adjusted min_value
                                          l2=hp.Float('l2_l2', min_value=1e-5, max_value=1e-2, sampling='log'))),  # Adjusted min_value
            BatchNormalization(),
            Dropout(rate=hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.05)),
            Dense(units=hp.Int('dense_units', min_value=16, max_value=128, step=16),
                  activation='relu',
                  kernel_regularizer=l1_l2(l1=hp.Float('l1_dense', min_value=1e-5, max_value=1e-3, sampling='log'),  # Adjusted min_value
                                           l2=hp.Float('l2_dense', min_value=1e-5, max_value=1e-2, sampling='log'))),  # Adjusted min_value
            Dropout(rate=hp.Float('dropout_3', min_value=0.2, max_value=0.5, step=0.05)),
            Dense(1, activation='sigmoid')
        ])

        optimizer_choice = hp.Choice('optimizer', values=['adam', 'rmsprop', 'nadam'])
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')

        if optimizer_choice == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_choice == 'rmsprop':
            optimizer = RMSprop(learning_rate=learning_rate)
        else:
            optimizer = Nadam(learning_rate=learning_rate)

        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy', Precision(), Recall(), AUC()])
        return model

def main():
    filepath = './data.h5'
    X_train, X_test, y_train, y_test = load_sequenced_data_from_h5(filepath)
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = 1

    hypermodel = LSTMHyperModel(input_shape=input_shape, num_classes=num_classes)

    tuner = kt.BayesianOptimization(
        hypermodel,
        objective='val_accuracy',
        max_trials=50,
        executions_per_trial=2,
        directory='keras_tuner_dir',
        project_name='lstm_hyper_tuning'
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    tuner.search(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping, reduce_lr], verbose=1)

    best_hyperparameters = tuner.get_best_hyperparameters()[0]
    best_model = tuner.hypermodel.build(best_hyperparameters)

    checkpoint_path = "training_1/cp.ckpt"
    cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    best_model.fit(X_train, y_train, epochs=200, batch_size=64, validation_split=0.2, callbacks=[cp_callback, early_stopping, reduce_lr], verbose=1)

    test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    y_pred = best_model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)

    print("Classification Report:\n", classification_report(y_test, y_pred_classes))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_classes))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

    best_model.save('best_lstm_model.h5')

if __name__ == "__main__":
    main()
