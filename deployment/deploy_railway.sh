#!/bin/bash
# Railway deployment script
# Deploy to Railway.app

set -e

echo "🚂 Deploying Bitcoin Sentiment ML to Railway..."

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "📦 Installing Railway CLI..."
    curl -fsSL https://railway.app/install.sh | sh
fi

# Check if logged in to Railway
if ! railway whoami &> /dev/null; then
    echo "🔐 Please log in to Railway:"
    railway login
fi

# Initialize Railway project (if not already done)
if [[ ! -f "railway.json" ]]; then
    echo "⚙️  Initializing Railway project..."
    railway init
fi

# Set environment variables
echo "🔧 Setting environment variables..."
railway variables set STREAMLIT_SERVER_HEADLESS=true
railway variables set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
railway variables set PYTHONPATH=/app

# Deploy to Railway
echo "🚀 Deploying to Railway..."
railway up

echo "✅ Deployment initiated!"
echo "🌐 Your app will be available at the Railway-provided URL"
echo "📊 Check deployment status: railway status"
echo "📋 View logs: railway logs"