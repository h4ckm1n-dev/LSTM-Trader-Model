# Bitcoin LSTM Trader Model

## Overview
This project aims to develop a predictive model for Bitcoin trading signals using Long Short-Term Memory (LSTM) networks. It focuses on preprocessing historical Bitcoin price data, feature engineering with technical indicators, and preparing the data for LSTM model training.

## Getting Started

### Prerequisites
- Python 3.x
- Pandas, Numpy, TensorFlow, Keras, Scikit-learn, TA-Lib
- HDF5 and h5py for data storage

### Installation
Clone the repository to your local machine:
```bash
git clone https://github.com/yourusername/LSTM-Trader-Model.git
```

### Data Preprocessing
1. **Data Loading**: Load Bitcoin price data from CSV.
2. **Cleaning**: Handle missing values and calculate daily returns.
3. **Feature Engineering**: Add technical indicators and EMAs.
4. **Normalization**: Use RobustScaler for data normalization.

### Sequence Generation
Generate sequences with dynamic thresholding to create signals for model training.

### Model Training
1. `load_sequenced_data_from_h5`: Retrieves pre-processed training and testing datasets, including features, labels, and future prices, from an HDF5 file.
2. `LSTMHyperModel`: A class to define and build the LSTM model architecture, allowing customization of hyperparameters for tuning purposes.
3. `main`: Manages the training process, including hyperparameter tuning with keras-tuner, implementing callbacks for training optimization, training with the best hyperparameters, evaluating model performance, and saving the trained model.

## Usage
- Run the preprocessing script to prepare your data.
- Use the model training script to train your LSTM model.
- Evaluate the model's performance with the provided testing data.

## Contributing
Contributions are welcome!

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments
- Thanks to the contributors of the used open-source libraries.
- Inspired by recent advances in machine learning.

