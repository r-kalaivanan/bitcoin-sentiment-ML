# Docker deployment script for Windows PowerShell
# Build and run the Bitcoin Sentiment ML dashboard using Docker

param(
    [string]$ImageName = "bitcoin-sentiment-ml",
    [string]$ContainerName = "bitcoin-dashboard",
    [int]$Port = 8501
)

Write-Host "🐳 Building and deploying Bitcoin Sentiment ML with Docker..." -ForegroundColor Green

# Check if Docker is running
try {
    docker info | Out-Null
} catch {
    Write-Host "❌ Docker is not running. Please start Docker first." -ForegroundColor Red
    exit 1
}

# Stop and remove existing container if it exists
Write-Host "🔄 Cleaning up existing containers..." -ForegroundColor Yellow
docker stop $ContainerName 2>$null
docker rm $ContainerName 2>$null

# Build the Docker image
Write-Host "🔨 Building Docker image..." -ForegroundColor Cyan
docker build -t $ImageName .

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker build failed!" -ForegroundColor Red
    exit 1
}

# Run the container
Write-Host "🚀 Starting container..." -ForegroundColor Green
$CurrentPath = (Get-Location).Path
docker run -d `
    --name $ContainerName `
    -p "${Port}:8501" `
    -v "${CurrentPath}/data:/app/data" `
    -v "${CurrentPath}/models:/app/models" `
    -v "${CurrentPath}/logs:/app/logs" `
    --restart unless-stopped `
    $ImageName

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to start container!" -ForegroundColor Red
    exit 1
}

# Wait for container to start
Write-Host "⏳ Waiting for container to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check if container is running
$ContainerStatus = docker ps --filter "name=$ContainerName" --format "table {{.Names}}"
if ($ContainerStatus -match $ContainerName) {
    Write-Host "✅ Container started successfully!" -ForegroundColor Green
    Write-Host "🌐 Dashboard available at: http://localhost:$Port" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "📋 Useful commands:" -ForegroundColor Yellow
    Write-Host "   View logs: docker logs -f $ContainerName"
    Write-Host "   Stop container: docker stop $ContainerName"
    Write-Host "   Restart container: docker restart $ContainerName"
    Write-Host "   Remove container: docker rm -f $ContainerName"
    
    # Open browser
    Start-Process "http://localhost:$Port"
} else {
    Write-Host "❌ Container failed to start. Check logs:" -ForegroundColor Red
    docker logs $ContainerName
    exit 1
}