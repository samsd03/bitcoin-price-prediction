from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model

def build_lstm_model(time_steps=20):
    model = Sequential([
        LSTM(40, return_sequences=False, input_shape=(time_steps, 1)),
        Dense(20, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=128):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    return model

def save_lstm_model(model, path):
    model.save(path)

def load_lstm_model(path):
    return load_model(path)
