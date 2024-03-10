import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall, AUC
import tensorflow.keras.backend as K
import keras_tuner as kt
from sklearn.metrics import classification_report, confusion_matrix

# Custom Attention Layer
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(Attention, self).get_config()

def load_sequenced_data_from_h5(filepath):
    with h5py.File(filepath, 'r') as hf:
        X_train = hf['X_train'][:]
        y_train = hf['y_train'][:]
        X_test = hf['X_test'][:]
        y_test = hf['y_test'][:]
    return X_train, X_test, to_categorical(y_train), to_categorical(y_test)

class LSTMHyperModel(kt.HyperModel):
    def __init__(self, input_shape, num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = Sequential([
            Bidirectional(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
                               activation='tanh',
                               input_shape=self.input_shape,
                               return_sequences=True,  # For attention
                               kernel_regularizer=l1_l2(l1=hp.Float('l1', min_value=1e-5, max_value=1e-2, sampling='log'),
                                                        l2=hp.Float('l2', min_value=1e-5, max_value=1e-2, sampling='log')))),
            BatchNormalization(),
            Dropout(rate=hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.05)),
            Attention(),
            Dense(units=hp.Int('dense_units', min_value=16, max_value=128, step=16),
                  activation='relu',
                  kernel_regularizer=l1_l2(l1=hp.Float('l1_dense', min_value=1e-5, max_value=1e-3, sampling='log'),
                                           l2=hp.Float('l2_dense', min_value=1e-5, max_value=1e-2, sampling='log'))),
            Dropout(rate=hp.Float('dropout_3', min_value=0.2, max_value=0.5, step=0.05)),
            Dense(self.num_classes, activation='softmax')
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
                      loss='categorical_crossentropy',
                      metrics=['accuracy', Precision(), Recall(), AUC()])
        return model

def main():
    # Load data
    filepath = './data.h5'
    X_train, X_test, y_train, y_test = load_sequenced_data_from_h5(filepath)
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = 3

    # Define hypermodel
    hypermodel = LSTMHyperModel(input_shape=input_shape, num_classes=num_classes)

    # Hyperparameter tuning
    tuner = kt.BayesianOptimization(
        hypermodel,
        objective='val_accuracy',
        max_trials=50,
        executions_per_trial=2,
        directory='keras_tuner_dir',
        project_name='lstm_hyper_tuning'
    )

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    # Perform hyperparameter search
    tuner.search(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping, reduce_lr], verbose=1)

    # Get best hyperparameters and build the model
    best_hyperparameters = tuner.get_best_hyperparameters()[0]
    best_model = tuner.hypermodel.build(best_hyperparameters)

    # Checkpoint callback
    checkpoint_path = "training_1/cp.ckpt"
    cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    # Train the best model
    best_model.fit(X_train, y_train, epochs=200, batch_size=64, validation_split=0.2, callbacks=[cp_callback, early_stopping, reduce_lr], verbose=1)

    # Evaluate model on test set
    results = best_model.evaluate(X_test, y_test, verbose=1)
    test_loss, test_accuracy, _, _, _ = results
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    # Generate predictions on test set
    y_pred = best_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Convert one-hot encoded labels to class labels
    y_test_classes = np.argmax(y_test, axis=1)

    # Print classification report and confusion matrix
    print("Classification Report:\n", classification_report(y_test_classes, y_pred_classes))
    print("Confusion Matrix:\n", confusion_matrix(y_test_classes, y_pred_classes))

    # Save the best model
    best_model.save('best_lstm_model')

if __name__ == "__main__":
    main()
