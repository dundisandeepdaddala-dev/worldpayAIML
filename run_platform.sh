#!/bin/bash
# run_platform.sh

echo "Starting AI Observability Platform..."
echo "=========================================="

# Create directories
mkdir -p logs models features

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Start the main platform
echo "Starting AI Observability Platform..."
python main.py

echo "=========================================="
echo "Platform is running!"
echo "Press Ctrl+C to stop"