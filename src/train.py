import numpy as np
from sklearn.linear_model import LinearRegression
from utils import load_data, save_model, calculate_metrics

def train_model():
    """Train Linear Regression model on California Housing dataset"""
    print("Loading California Housing dataset...")
    X_train, X_test, y_train, y_test = load_data()
    
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_r2, train_mae, train_mse = calculate_metrics(y_train, y_pred_train)
    test_r2, test_mae, test_mse = calculate_metrics(y_test, y_pred_test)
    
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Training MAE: {train_mae:.4f}")
    print(f"Training MSE: {train_mse:.4f}")
    
    print(f"Test R² Score: {test_r2:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    
    # Save model
    save_model(model, '../models/linear_regression_model.joblib')
    print("Model saved successfully!")
    
    return model, test_r2

if __name__ == "__main__":
    train_model()
