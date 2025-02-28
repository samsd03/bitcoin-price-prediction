import argparse
from src.data_preprocessing import prepare_data
from src.lstm_model import build_lstm_model, train_lstm_model, save_lstm_model
from src.xgboost_model import train_xgboost, save_xgboost_model, predict_xgboost
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess data
file_path = "data/raw/btcusd_1-min_data.csv"
X_train, y_train, X_test, y_test, scaler = prepare_data(file_path, time_steps=20)

# Argument parser for selecting model
parser = argparse.ArgumentParser(description="Train and test a Bitcoin prediction model.")
parser.add_argument("--model", choices=["lstm", "xgboost"], required=True, help="Choose model: 'lstm' or 'xgboost'")
args = parser.parse_args()

if args.model == "lstm":
    print("Training LSTM model...")
    model = build_lstm_model(time_steps=20)
    model = train_lstm_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=128)
    save_lstm_model(model, "models/lstm_model.h5")
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

elif args.model == "xgboost":
    print("Training XGBoost model...")
    model = train_xgboost(X_train, y_train)
    save_xgboost_model(model, "models/xgboost_model.json")
    predictions = predict_xgboost(model, X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

# Transform y_test back to original scale
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(y_test_original, color='blue', label="Actual Price")
plt.plot(predictions, color='red', label="Predicted Price")
plt.title(f"Bitcoin Price Prediction using {args.model.upper()}")
plt.legend()
plt.show()
