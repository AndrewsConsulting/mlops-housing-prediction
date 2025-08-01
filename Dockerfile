# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create models directory
RUN mkdir -p models

# Set Python path
ENV PYTHONPATH=/app/src

# Run the full pipeline: train -> quantize -> predict
CMD ["sh", "-c", "cd src && python train.py && python quantize.py && python predict.py"]
