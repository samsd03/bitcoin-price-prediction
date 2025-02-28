import xgboost as xgb
import joblib

def train_xgboost(X_train, y_train):
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    return model

def save_xgboost_model(model, path):
    joblib.dump(model, path)

def load_xgboost_model(path):
    return joblib.load(path)

def predict_xgboost(model, X_test):
    return model.predict(X_test.reshape(X_test.shape[0], -1))
