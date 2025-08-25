#!/bin/bash
# run_platform.sh

echo "Starting AI Observability Platform..."
echo "=========================================="

# Create directories
mkdir -p logs models features

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Start Docker containers
echo "Starting Docker containers..."
docker-compose up -d

# Wait for services
echo "Waiting for services to be ready..."
sleep 15

# Pull LLM model
echo "Pulling LLM model..."
curl -X POST http://localhost:11434/api/pull -d '{"name": "llama3.1"}'

# Start the main platform
echo "Starting AI Observability Platform..."
python main.py

echo "=========================================="
echo "Platform is running!"
echo "Press Ctrl+C to stop"

cleanup() {
    echo "Stopping Docker stack..."
    docker-compose down
    exit
}

trap cleanup INT TERM EXIT
wait