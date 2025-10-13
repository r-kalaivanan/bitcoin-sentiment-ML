#!/bin/bash
# Docker deployment script
# Build and run the Bitcoin Sentiment ML dashboard using Docker

set -e

IMAGE_NAME="bitcoin-sentiment-ml"
CONTAINER_NAME="bitcoin-dashboard"
PORT=8501

echo "🐳 Building and deploying Bitcoin Sentiment ML with Docker..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Stop and remove existing container if it exists
echo "🔄 Cleaning up existing containers..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t $IMAGE_NAME .

# Run the container
echo "🚀 Starting container..."
docker run -d \
    --name $CONTAINER_NAME \
    -p $PORT:8501 \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/models:/app/models" \
    -v "$(pwd)/logs:/app/logs" \
    --restart unless-stopped \
    $IMAGE_NAME

# Wait for container to start
echo "⏳ Waiting for container to start..."
sleep 10

# Check if container is running
if docker ps | grep -q $CONTAINER_NAME; then
    echo "✅ Container started successfully!"
    echo "🌐 Dashboard available at: http://localhost:$PORT"
    echo ""
    echo "📋 Useful commands:"
    echo "   View logs: docker logs -f $CONTAINER_NAME"
    echo "   Stop container: docker stop $CONTAINER_NAME"
    echo "   Restart container: docker restart $CONTAINER_NAME"
    echo "   Remove container: docker rm -f $CONTAINER_NAME"
    
    # Open browser (optional)
    if command -v xdg-open > /dev/null; then
        xdg-open http://localhost:$PORT
    elif command -v open > /dev/null; then
        open http://localhost:$PORT
    fi
else
    echo "❌ Container failed to start. Check logs:"
    docker logs $CONTAINER_NAME
    exit 1
fi