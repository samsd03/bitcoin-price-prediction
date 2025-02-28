import sys
import os
import pytest
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Add the src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_preprocessing import load_data, normalize_data, create_sequences, prepare_data


# Fixture to provide sample data for testing
@pytest.fixture
def sample_data():
    data = {
        'Close': [10000, 10100, 10200, 10300, 10400, 10500, 10600, 10700, 10800, 10900]
    }
    return pd.DataFrame(data)


# Test for the load_data function
def test_load_data(mocker):
    # Mock the pandas read_csv function to return a predefined DataFrame
    mocker.patch('pandas.read_csv', return_value=pd.DataFrame({
        'Close': [10000, 10100, 10200, 10300, 10400]
    }))
    df = load_data('dummy_path')
    # Check that there are no null values and 'Close' column exists
    assert not df.isnull().values.any()
    assert 'Close' in df.columns


# Test for the normalize_data function
def test_normalize_data(sample_data):
    data_scaled, scaler = normalize_data(sample_data)
    # Check that the data is scaled between 0 and 1
    assert data_scaled.min() == 0
    assert data_scaled.max() == 1
    # Check that the scaler is an instance of MinMaxScaler
    assert isinstance(scaler, MinMaxScaler)


# Test for the create_sequences function
def test_create_sequences(sample_data):
    data_scaled, _ = normalize_data(sample_data)
    X, y = create_sequences(data_scaled, time_steps=3)
    # Check the shape of the sequences
    assert X.shape == (7, 3, 1)
    assert y.shape == (7, 1)


# Test for the prepare_data function
def test_prepare_data(mocker):
    # Mock the pandas read_csv function to return a predefined DataFrame
    mocker.patch('pandas.read_csv', return_value=pd.DataFrame({
        'Close': [10000, 10100, 10200, 10300, 10400, 10500, 10600, 10700, 10800, 10900]
    }))
    X_train, y_train, X_test, y_test, scaler = prepare_data('dummy_path', time_steps=3, train_split=0.8)
    # Check the shape of the training and testing data
    assert X_train.shape == (5, 3, 1)
    assert y_train.shape == (5, 1)
    assert X_test.shape == (2, 3, 1)
    assert y_test.shape == (2, 1)
    # Check that the scaler is an instance of MinMaxScaler
    assert isinstance(scaler, MinMaxScaler)
