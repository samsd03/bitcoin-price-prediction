import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """
    Load the Bitcoin dataset from a CSV file.
    
    Args:
        file_path (str): Path to the dataset CSV file.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with required columns.
    """
    df = pd.read_csv(file_path)
    df = df.dropna()  # Remove missing values
    return df[['Close']]  # Use only the closing price

def normalize_data(data):
    """
    Normalize data using MinMaxScaler (scales between 0 and 1).
    
    Args:
        data (pd.DataFrame): DataFrame containing the closing price.
    
    Returns:
        np.array: Scaled data.
        MinMaxScaler: Scaler instance to inverse transform predictions.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

def create_sequences(data, time_steps=20):
    """
    Create input sequences and labels for LSTM training.
    
    Args:
        data (np.array): Scaled data.
        time_steps (int): Number of previous time steps to consider.
    
    Returns:
        tuple: (X, y) where X is the input sequences and y is the target values.
    """
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

def prepare_data(file_path, time_steps=20, train_split=0.8):
    """
    Complete data preparation pipeline: load, normalize, and split data.
    
    Args:
        file_path (str): Path to the dataset CSV file.
        time_steps (int): Number of time steps to consider for input sequences.
        train_split (float): Percentage of data to use for training.
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test, scaler)
    """
    df = load_data(file_path)
    data_scaled, scaler = normalize_data(df)
    X, y = create_sequences(data_scaled, time_steps)
    
    train_size = int(len(X) * train_split)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    return X_train, y_train, X_test, y_test, scaler
