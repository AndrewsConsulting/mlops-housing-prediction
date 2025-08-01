import numpy as np
import joblib
from utils import load_model, save_model, load_data, calculate_metrics, quantize_weights, dequantize_weights

def quantize_model():
    """Quantize the trained model to 8-bit as per assignment requirements"""
    print("Loading trained model...")
    model = load_model('../models/linear_regression_model.joblib')
    
    # Extract coefficients and intercept
    coef = model.coef_
    intercept = model.intercept_
    
    print(f"Original coefficients shape: {coef.shape}")
    print(f"Original intercept: {intercept}")
    
    # Save raw parameters (as per assignment requirement)
    raw_params = {'coef': coef, 'intercept': intercept}
    joblib.dump(raw_params, '../models/unquant_params.joblib')
    print("Raw parameters saved!")
    
    # 8-bit quantization only (as per assignment)
    print("\n8-bit Quantization:")
    coef_8bit, coef_scale_8, coef_zero_8 = quantize_weights(coef, bits=8)
    intercept_8bit, int_scale_8, int_zero_8 = quantize_weights(np.array([intercept]), bits=8)
    
    quant_params = {
        'coef': coef_8bit,
        'intercept': intercept_8bit[0],
        'coef_scale': coef_scale_8,
        'coef_zero_point': coef_zero_8,
        'intercept_scale': int_scale_8,
        'intercept_zero_point': int_zero_8,
        'bits': 8
    }
    joblib.dump(quant_params, '../models/quant_params.joblib')
    print("8-bit quantized parameters saved!")
    
    return quant_params

def predict_with_quantized(X, quant_params):
    """Make predictions using quantized parameters"""
    dequant_coef = dequantize_weights(
        quant_params['coef'], 
        quant_params['coef_scale'], 
        quant_params['coef_zero_point']
    )
    dequant_intercept = dequantize_weights(
        np.array([quant_params['intercept']]), 
        quant_params['intercept_scale'], 
        quant_params['intercept_zero_point']
    )[0]
    
    predictions = np.dot(X, dequant_coef) + dequant_intercept
    return predictions

if __name__ == "__main__":
    quantize_model()