#!/bin/bash

echo "Starting AI Observability Demo..."
echo "=========================================="

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Start Docker containers
echo "Starting Docker containers..."
docker compose up -d

# Wait for services to start
echo "Waiting for services to be ready..."
sleep 15

# Pull the LLM model (if not already available)
echo "Checking for LLM model..."
if ! ollama list | grep -q "llama3.1"; then
    echo "Pulling LLM model..."
    ollama pull llama3.1
fi

# Run the main platform
echo "Starting AI Observability Platform..."
python main.py

echo "=========================================="
echo "Demo is running!"
echo "Grafana: http://localhost:3000 (admin/admin)"
echo "Prometheus: http://localhost:9090"
echo "Press Ctrl+C to stop the demo"

# Function to cleanup on exit
cleanup() {
    echo "Stopping Docker stack..."
    docker compose down
    exit
}

# Set trap to cleanup on script exit
trap cleanup INT TERM EXIT

# Wait indefinitely until interrupted
wait