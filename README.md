This project implements a complete MLOps pipeline for Linear Regression on the California Housing dataset, including training, testing, 8-bit quantization, Dockerization, and CI/CD.
Project Structure
mlops-housing-prediction/
├── .github/workflows/ci.yml    
├── src/                        
│   ├── train.py               
│   ├── quantize.py            
│   ├── predict.py             
│   └── utils.py               
├── tests/                      
│   └── test_train.py          
├── models/                     
├── Dockerfile                  
├── requirements.txt            
└── README.md
Setup and Usage
Local Development
bashgit clone <repository-url>
cd mlops-housing-prediction

python -m venv venv
source venv/bin/activate  

pip install -r requirements.txt

cd src
python train.py
python quantize.py
python predict.py

python -m pytest ../tests/ -v
Docker Usage
bashdocker build -t mlops-housing .
docker run --rm mlops-housing
Model Training Results
The model was trained on the California Housing dataset with the following results:
Training R² Score: 0.6126
Training MAE: 0.5286
Training MSE: 0.5179
Test R² Score: 0.5758
Test MAE: 0.5332
Test MSE: 0.5559
Quantization Implementation
The quantization process extracts model coefficients and intercept, then converts them to 8-bit unsigned integers. The original coefficients have shape (8,) with intercept value -37.02327770606391.
Raw parameters are saved in models/unquant_params.joblib and quantized parameters in models/quant_params.joblib.
Performance Comparison Table
Overall Test Set Performance:
Metric          Original        8-bit Quantized     Impact
R² Score        0.575788        -0.179884          -131% degradation
MAE             0.533200        1.016263           +90.6% increase
MSE             0.555892        1.546130           +178% increase

Sample Predictions Analysis (First 10 test samples):
Sample  True Value   Original     8-bit Quant   Error
1       0.477000     0.719123     1.496208      1.019208
2       0.458000     1.764017     2.637593      2.179593
3       5.000010     2.709659     3.455221      1.544789
4       2.186000     2.838926     3.784162      1.598162
5       2.780000     2.604657     3.208015      0.428015
6       1.587000     2.011754     3.332973      1.745973
7       1.982000     2.645500     3.034960      1.052960
8       1.575000     2.168755     2.743414      1.168414
9       3.400000     2.740746     3.327787      0.072213
10      4.466000     3.915615     4.458153      0.007847
Average Absolute Error: 1.081717

First 5 original predictions: [0.71912309 1.76401703 2.70965947 2.83892622 2.60465716]
First 5 quantized predictions: [1.49620815 2.63759298 3.45522099 3.78416231 3.20801544]
Analysis
The 8-bit quantization shows significant performance degradation with negative R² score indicating predictions worse than using the mean. This demonstrates that 8-bit compression is too aggressive for linear regression coefficients which are sensitive to precision loss.
The quantization maps continuous coefficient values to a 0-255 range causing substantial information loss. While this achieves approximately 75% model size reduction, the accuracy cost makes it unsuitable for production deployment.
CI/CD Pipeline
The pipeline has three sequential jobs:

test_suite - Runs pytest validation, must pass before others execute
train_and_quantize - Trains model, performs quantization, uploads artifacts
build_and_test_container - Builds Docker image, runs container with predict.py execution

All unit tests passed including dataset loading, model creation, training validation, R² threshold check, and model persistence verification.
Technical Implementation
Dataset: California Housing from sklearn.datasets (20,640 samples, 8 features)
Model: scikit-learn LinearRegression
Split: 80% training, 20% testing with random_state=42
The quantization process extracts coefficients and intercept, saves raw parameters, then applies min-max normalization to scale values to 0-255 range for uint8 conversion. Scaling factors and zero points are stored for dequantization during inference.
Files Generated
models/linear_regression_model.joblib - Original trained model
models/unquant_params.joblib - Raw model parameters
models/quant_params.joblib - 8-bit quantized parameters
models/prediction_comparison.csv - Performance comparison data
Docker Implementation
The container uses python:3.9-slim base image and executes the complete pipeline sequence of train.py, quantize.py, and predict.py. The containerized execution successfully generates all required files and produces the performance comparison results.