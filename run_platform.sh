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

# Pull the LLM model and wait for Ollama to be ready
echo "Checking for LLM model..."
max_retries=10
retry_count=0

while [ $retry_count -lt $max_retries ]; do
    if curl -s http://localhost:11434/api/tags > /dev/null; then
        echo "Ollama is ready!"
        if ! ollama list | grep -q "llama3.1"; then
            echo "Pulling LLM model..."
            ollama pull llama3.1
        fi
        break
    else
        echo "Waiting for Ollama to be ready... (Attempt $((retry_count+1))/$max_retries)"
        sleep 10
        ((retry_count++))
    fi
done

if [ $retry_count -eq $max_retries ]; then
    echo "Warning: Ollama not ready after $max_retries attempts. Continuing without LLM."
fi

# Run the main platform
echo "Starting AI Observability Platform..."
python main.py

echo "=========================================="
echo "Demo is running!"
echo "Grafana: http://localhost:3000 (admin/admin)"
echo "Prometheus: http://localhost:9090"
echo "Ollama: http://localhost:11434"
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