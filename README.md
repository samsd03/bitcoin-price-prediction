# Bitcoin Price Prediction

## Overview
This repository contains multiple machine learning and deep learning models for Bitcoin price prediction using historical price data. The models include:
- **LSTM (Long Short-Term Memory)**
- **XGBoost (Extreme Gradient Boosting)**
- **Other ML Algorithms** (e.g., Random Forest, Linear Regression)

## Features
- Data preprocessing and feature engineering
- Multiple predictive models for comparison
- Model evaluation using metrics like RMSE and MAPE
- Visualization of actual vs predicted prices
- Scalable and modular code structure

## Technologies Used
- **Python**
- **TensorFlow/Keras** (for LSTM)
- **XGBoost**
- **Scikit-learn**
- **Matplotlib & Seaborn** (for data visualization)
- **Pandas & NumPy** (for data manipulation)

## Dataset
The dataset used for training the models consists of historical Bitcoin price data.

## Installation
Clone the repository:
```bash
git clone https://github.com/yourusername/bitcoin-price-prediction.git
cd bitcoin-price-prediction
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### Train the Model
Run the training script to train a model of your choice:
```bash
python train_lstm.py  # For LSTM
python train_xgboost.py  # For XGBoost
```

### Predict Bitcoin Prices
```bash
python predict.py --model lstm --input latest_price.csv
```

## Results
Model performance is evaluated using RMSE and visualization:
- LSTM: **XX% accuracy**
- XGBoost: **XX% accuracy**

## Contributing
Feel free to contribute by adding new models, improving existing ones, or optimizing performance.

## License
This project is licensed under the MIT License.

