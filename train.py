import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K
import keras_tuner as kt
from sklearn.metrics import classification_report, confusion_matrix

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("CUDA is enabled. GPU device(s) found:")
    for gpu in gpus:
        print(gpu)
else:
    print("CUDA is not enabled. Using CPU for computation.")
    
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

def load_sequenced_data_from_h5(filepath):
    with h5py.File(filepath, 'r') as hf:
        X_train = hf['X_train'][:]
        y_train = hf['y_train'][:]
        X_test = hf['X_test'][:]
        y_test = hf['y_test'][:]
        print("Shape of y_train:", y_train.shape)
        print("Shape of y_test:", y_test.shape)
        # Reshape y_train and y_test to remove the extra dimension
    y_train = y_train.argmax(axis=1)
    y_test = y_test.argmax(axis=1)
    
    # Ensure the shape of y_train and y_test is correct
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    
    # Ensure the labels are integers
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    
    # Convert labels to one-hot encoding
    y_train_categorical = to_categorical(y_train)
    y_test_categorical = to_categorical(y_test)
    return X_train, X_test, y_train_categorical, y_test_categorical

class LSTMHyperModel(kt.HyperModel):
    def __init__(self, input_shape, num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = Sequential([
            Input(shape=self.input_shape),  # Explicitly define input shape
            Bidirectional(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
                               activation='tanh',
                               return_sequences=True)),
            BatchNormalization(),
            Dropout(rate=hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.05)),
            Attention(),
            Dense(units=hp.Int('dense_units', min_value=16, max_value=128, step=16), activation='relu'),
            Dropout(rate=hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.05)),
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
                      metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])
        return model

def main():
    filepath = './data.h5'
    X_train, X_test, y_train_categorical, y_test_categorical = load_sequenced_data_from_h5(filepath)

    # Shape verification
    print("Shape of y_train_categorical:", y_train_categorical.shape)  # Should output (N, 3)
    print("Shape of y_test_categorical:", y_test_categorical.shape)    # Should output (M, 3)

    input_shape = (X_train.shape[1], X_train.shape[2])
    hypermodel = LSTMHyperModel(input_shape=input_shape, num_classes=3)

    tuner = kt.BayesianOptimization(
        hypermodel,
        objective='val_loss',
        max_trials=10,
        executions_per_trial=2,
        directory='keras_tuner_dir',
        project_name='lstm_hyper_tuning'
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    tuner.search(X_train, y_train_categorical, epochs=100, validation_split=0.2, callbacks=[early_stopping, reduce_lr], verbose=1)

    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hyperparameters)

    checkpoint_path = "best_model_cp.weights.h5"
    cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    best_model.fit(X_train, y_train_categorical, epochs=200, batch_size=64, validation_split=0.2, callbacks=[cp_callback, early_stopping, reduce_lr], verbose=1)

    results = best_model.evaluate(X_test, y_test_categorical, verbose=1)
    print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

    y_pred_probs = best_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_test_classes = np.argmax(y_test_categorical, axis=1)

    print("Classification Report:\n", classification_report(y_test_classes, y_pred_classes))
    print("Confusion Matrix:\n", confusion_matrix(y_test_classes, y_pred_classes))

    best_model.save('best_lstm_model.h5')

if __name__ == "__main__":
    main()
