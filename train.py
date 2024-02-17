import numpy as np
import pandas as pd
import h5py
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import keras_tuner as kt
from sklearn.model_selection import KFold

def load_sequenced_data_from_h5(filepath):
    with h5py.File(filepath, 'r') as hf:
        X_train = hf['X_train'][:]
        y_train = hf['y_train'][:]
        X_test = hf['X_test'][:]
        y_test = hf['y_test'][:]
        future_prices_test = hf['future_prices_test'][:]
    return X_train, X_test, y_train, y_test, future_prices_test

class LSTMHyperModel(kt.HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = Sequential([
            LSTM(units=hp.Int('units', min_value=32, max_value=256, step=32),
                 activation='tanh',
                 input_shape=self.input_shape,
                 return_sequences=True,
                 kernel_regularizer=l1_l2(l1=hp.Float('l1', min_value=1e-5, max_value=1e-3, sampling='LOG'),
                                          l2=hp.Float('l2', min_value=1e-4, max_value=1e-2, sampling='LOG'))),
            Dropout(rate=hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.05)),

            LSTM(units=hp.Int('units_l2', min_value=32, max_value=256, step=32),
                 activation='tanh',
                 return_sequences=False,
                 kernel_regularizer=l1_l2(l1=hp.Float('l1_l2', min_value=1e-5, max_value=1e-3, sampling='LOG'),
                                          l2=hp.Float('l2_l2', min_value=1e-4, max_value=1e-2, sampling='LOG'))),
            Dropout(rate=hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.05)),

            Dense(units=hp.Int('dense_units', min_value=16, max_value=128, step=16),
                  activation='relu',
                  kernel_regularizer=l1_l2(l1=hp.Float('l1_dense', min_value=1e-5, max_value=1e-3, sampling='LOG'),
                                           l2=hp.Float('l2_dense', min_value=1e-4, max_value=1e-2, sampling='LOG'))),
            Dropout(rate=hp.Float('dropout_3', min_value=0.2, max_value=0.5, step=0.05)),

            Dense(1, activation='sigmoid')
        ])

        # Add epochs as a hyperparameter
        epochs = hp.Int('epochs', min_value=50, max_value=200, step=10)

        model.compile(optimizer=Adam(hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

def main():
    filepath = './data.h5'
    X_train, X_test, y_train, y_test, future_prices_test = load_sequenced_data_from_h5(filepath)

    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = 1  # Binary classification
    hypermodel = LSTMHyperModel(input_shape=input_shape, num_classes=num_classes)

    tuner = kt.RandomSearch(
        hypermodel,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='keras_tuner_dir',
        project_name='lstm_hyper_tuning'
    )

    # Learning rate scheduler options
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))  # Example of a cyclic LR scheduler

    kf = KFold(n_splits=5, shuffle=True)

    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Pass validation data to tuner.search
        tuner.search(X_train_fold, y_train_fold, epochs=50, validation_data=(X_val_fold, y_val_fold), callbacks=[lr_scheduler], verbose=1)

    best_hyperparameters = tuner.get_best_hyperparameters()[0]
    best_model = tuner.hypermodel.build(best_hyperparameters)

    # ModelCheckpoint for saving the best model during the final training
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                  save_weights_only=True,
                                  verbose=1)

    # Final training with the best hyperparameters
    epochs = best_hyperparameters.get('epochs')  # Get the value of the 'epochs' hyperparameter
    history = best_model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, callbacks=[cp_callback], verbose=1)

    # Evaluate the best model
    test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    
    # Predict classes
    y_pred = best_model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype("int32")

    # Compute additional metrics
    print("Classification Report:\n", classification_report(y_test, y_pred_classes))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_classes))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

    # Save the fully trained model for later use or inference
    best_model.save('best_lstm_model')

if __name__ == "__main__":
    main()