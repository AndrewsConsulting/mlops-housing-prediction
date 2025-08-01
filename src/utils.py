import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

def load_data():
    """Load and split California Housing dataset"""
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def save_model(model, filepath):
    """Save model using joblib"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)

def load_model(filepath):
    """Load model using joblib"""
    return joblib.load(filepath)

def calculate_metrics(y_true, y_pred):
    """Calculate R2, MAE, and MSE"""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return r2, mae, mse

def quantize_weights(weights, bits=8):
    """Quantize weights to specified bits"""
    weights = np.array(weights)
    if bits == 8:
        # Scale to 0-255 range for uint8
        w_min, w_max = weights.min(), weights.max()
        # Handle edge case where all weights are the same
        if abs(w_max - w_min) < 1e-8:
            scale = 1.0
            zero_point = w_min
            quantized = np.zeros_like(weights, dtype=np.uint8)
        else:
            scale = (w_max - w_min) / 255.0
            zero_point = w_min
            quantized = np.round((weights - zero_point) / scale).astype(np.uint8)
        return quantized, scale, zero_point
    elif bits == 16:
        # Scale to 0-65535 range for uint16
        w_min, w_max = weights.min(), weights.max()
        # Handle edge case where all weights are the same
        if abs(w_max - w_min) < 1e-8:
            scale = 1.0
            zero_point = w_min
            quantized = np.zeros_like(weights, dtype=np.uint16)
        else:
            scale = (w_max - w_min) / 65535.0
            zero_point = w_min
            quantized = np.round((weights - zero_point) / scale).astype(np.uint16)
        return quantized, scale, zero_point

def dequantize_weights(quantized_weights, scale, zero_point):
    """Dequantize weights back to float32"""
    return quantized_weights.astype(np.float32) * scale + zero_point