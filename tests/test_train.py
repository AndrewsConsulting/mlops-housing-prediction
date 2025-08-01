import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_data, load_model
from train import train_model

class TestTraining:
    def test_dataset_loading(self):
        """Test if dataset loads correctly"""
        X_train, X_test, y_train, y_test = load_data()
        
        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None
        
        # Check shapes
        assert X_train.shape[1] == 8  # California housing has 8 features
        assert X_test.shape[1] == 8
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        
        print("✓ Dataset loading test passed")
    
    def test_model_creation(self):
        """Test if model is LinearRegression instance"""
        model, _ = train_model()
        
        assert isinstance(model, LinearRegression)
        print("✓ Model creation test passed")
    
    def test_model_training(self):
        """Test if model was trained (coefficients exist)"""
        model, _ = train_model()
        
        # Check if coefficients exist (model is fitted)
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        assert model.coef_ is not None
        assert model.intercept_ is not None
        
        # Check coefficient shape
        assert len(model.coef_) == 8  # 8 features
        
        print("✓ Model training test passed")
    
    def test_r2_threshold(self):
        """Test if R² score exceeds minimum threshold"""
        _, r2_score = train_model()
        
        # Minimum threshold for California housing dataset
        min_threshold = 0.5
        assert r2_score > min_threshold, f"R² score {r2_score} is below threshold {min_threshold}"
        
        print(f"✓ R² threshold test passed (R² = {r2_score:.4f})")
    
    def test_model_saved(self):
        """Test if model file is saved"""
        train_model()
        
        model_path = '../models/linear_regression_model.joblib'
        assert os.path.exists(model_path), "Model file was not saved"
        
        # Test if saved model can be loaded
        loaded_model = load_model(model_path)
        assert isinstance(loaded_model, LinearRegression)
        
        print("✓ Model saving test passed")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
