import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

def evaluate_model(model, X_test, y_test, scaler, model_type="lstm"):
    """
    Evaluates a model (LSTM or XGBoost) using RMSE, MAPE, and Accuracy.

    Parameters:
    - model: Trained model (LSTM or XGBoost)
    - X_test: Test feature set
    - y_test: True test values
    - scaler: Scaler used for inverse transformation
    - model_type: "lstm" or "xgboost" (default: "lstm")

    Returns:
    - RMSE, MAPE, Accuracy
    """

    # Predict based on model type
    if model_type == "lstm":
        predictions = model.predict(X_test)
    elif model_type == "xgboost":
        predictions = model.predict(X_test.reshape(X_test.shape[0], -1))
    else:
        raise ValueError("Invalid model_type. Choose 'lstm' or 'xgboost'.")

    # Convert predictions and actual values back to original scale
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
    predictions_original = scaler.inverse_transform(predictions.reshape(-1, 1))

    # Compute Metrics
    rmse = np.sqrt(mean_squared_error(y_test_original, predictions_original))
    mape = mean_absolute_percentage_error(y_test_original, predictions_original) * 100
    accuracy = 100 - mape  # Higher MAPE means lower accuracy

    print(f"{model_type.upper()} Model Evaluation:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Accuracy: {accuracy:.2f}%")

    return rmse, mape, accuracy
