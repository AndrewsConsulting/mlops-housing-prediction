import numpy as np
import pandas as pd
from utils import load_model, load_data, calculate_metrics
from quantize import predict_with_quantized
import joblib

def run_predictions():
    """Run predictions and create comparison table"""
    print("Loading data and models...")
    X_train, X_test, y_train, y_test = load_data()
    
    # Load original model
    original_model = load_model('../models/linear_regression_model.joblib')
    
    # Load quantized parameters
    quant_params_8bit = joblib.load('../models/quant_params.joblib')
    
    # Take first 10 samples for comparison
    X_sample = X_test[:10]
    y_sample = y_test[:10]
    
    print("Making predictions...")
    
    # Original predictions
    y_pred_original = original_model.predict(X_test)
    y_pred_original_sample = original_model.predict(X_sample)
    
    # 8-bit quantized predictions
    y_pred_8bit = predict_with_quantized(X_test, quant_params_8bit)
    y_pred_8bit_sample = predict_with_quantized(X_sample, quant_params_8bit)
    
    # Calculate metrics for full test set
    r2_orig, mae_orig, mse_orig = calculate_metrics(y_test, y_pred_original)
    r2_8bit, mae_8bit, mse_8bit = calculate_metrics(y_test, y_pred_8bit)
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON TABLE")
    print("="*60)
    
    print("\nOverall Test Set Metrics:")
    print("-" * 50)
    print(f"{'Metric':<15} {'Original':<15} {'8-bit Quantized':<15}")
    print("-" * 50)
    print(f"{'R² Score':<15} {r2_orig:<15.6f} {r2_8bit:<15.6f}")
    print(f"{'MAE':<15} {mae_orig:<15.6f} {mae_8bit:<15.6f}")
    print(f"{'MSE':<15} {mse_orig:<15.6f} {mse_8bit:<15.6f}")
    
    print("\n10 Sample Predictions Comparison:")
    print("-" * 80)
    print(f"{'Sample':<8} {'True Value':<12} {'Original':<12} {'8-bit Quant':<12} {'Error':<12}")
    print("-" * 80)
    
    for i in range(10):
        error_8bit = abs(y_sample[i] - y_pred_8bit_sample[i])
        print(f"{i+1:<8} {y_sample[i]:<12.6f} {y_pred_original_sample[i]:<12.6f} {y_pred_8bit_sample[i]:<12.6f} {error_8bit:<12.6f}")
    
    # Create DataFrame for easier analysis
    comparison_df = pd.DataFrame({
        'Sample': range(1, 11),
        'True_Value': y_sample,
        'Original_Pred': y_pred_original_sample,
        '8bit_Quant_Pred': y_pred_8bit_sample,
        '8bit_Error': np.abs(y_sample - y_pred_8bit_sample)
    })
    
    print(f"\nAverage Absolute Error on 10 samples:")
    print(f"8-bit Quantization: {comparison_df['8bit_Error'].mean():.6f}")
    
    # Save comparison results
    comparison_df.to_csv('../models/prediction_comparison.csv', index=False)
    print(f"\nComparison results saved to '../models/prediction_comparison.csv'")
    
    # Print sample outputs (as required by assignment)
    print(f"\nSample Outputs:")
    print(f"First 5 original predictions: {y_pred_original_sample[:5]}")
    print(f"First 5 quantized predictions: {y_pred_8bit_sample[:5]}")

if __name__ == "__main__":
    run_predictions()