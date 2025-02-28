# Bitcoin Price Prediction

## Overview
This repository contains multiple machine learning and deep learning models for Bitcoin price prediction using historical price data. The models include:
- **LSTM (Long Short-Term Memory)**
- **XGBoost (Extreme Gradient Boosting)**
- **Other ML Algorithms** (Need to be added in future) 

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
The dataset used for training the models consists of historical Bitcoin price data. The dataset is fetched from https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data .

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

### Predict Bitcoin Prices (Still Need to create this script)
```bash
python predict.py --model lstm --input latest_price.csv
```

## Results
Model performance is evaluated using RMSE and visualization:
- LSTM Model Evaluation:

        RMSE: 13.37
        MAPE: 0.57%
        Accuracy: 99.43%
  
![lstm](https://github.com/user-attachments/assets/049352d2-2d4a-4c83-af40-322cf2275283)

- XGBOOST Model Evaluation:

        RMSE: 89.14 %
        MAPE: 5.06 %
        Accuracy: 94.94 %
![xgboost](https://github.com/user-attachments/assets/ea7219ab-998c-47f1-aa75-0ccc2df36f5b)



## Contributing
Feel free to contribute by adding new models, improving existing ones, or optimizing performance.

## License
This project is licensed under the MIT License.

