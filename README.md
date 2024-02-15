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
Model Definition: Create an LSTM model with dynamically tuned hyperparameters.
Hyperparameter Tuning: Use keras-tuner to find the optimal model configuration.
Training Process: Employ callbacks like EarlyStopping and ModelCheckpoint for efficient training.
Evaluation: Assess model performance with metrics such as accuracy, confusion matrix, and ROC-AUC score.
Model Saving: Save the trained model for future inference.
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

