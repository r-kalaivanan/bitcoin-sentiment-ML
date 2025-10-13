#!/bin/bash
# Deploy to Heroku
# Complete Heroku deployment script

set -e

APP_NAME="bitcoin-sentiment-ml-$(date +%s)"
echo "🚀 Deploying Bitcoin Sentiment ML to Heroku..."

# Check if Heroku CLI is installed
if ! command -v heroku &> /dev/null; then
    echo "❌ Heroku CLI not found. Please install it first:"
    echo "   https://devcenter.heroku.com/articles/heroku-cli"
    exit 1
fi

# Check if logged in to Heroku
if ! heroku auth:whoami &> /dev/null; then
    echo "🔐 Please log in to Heroku first:"
    heroku login
fi

# Create Heroku app
echo "📱 Creating Heroku app: $APP_NAME"
heroku create $APP_NAME

# Set buildpack
echo "🔧 Setting Python buildpack..."
heroku buildpacks:set heroku/python -a $APP_NAME

# Set environment variables
echo "⚙️  Setting environment variables..."
heroku config:set STREAMLIT_SERVER_HEADLESS=true -a $APP_NAME
heroku config:set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false -a $APP_NAME
heroku config:set PYTHONPATH=/app -a $APP_NAME

# Add Redis addon (optional, for caching)
echo "📊 Adding Redis addon for caching..."
heroku addons:create heroku-redis:mini -a $APP_NAME || echo "⚠️  Redis addon skipped (requires paid account)"

# Deploy to Heroku
echo "🚀 Deploying to Heroku..."
git add .
git commit -m "Deploy to Heroku" || echo "No changes to commit"
git push heroku main || git push heroku master

# Open the app
echo "✅ Deployment complete!"
echo "🌐 Opening your app..."
heroku open -a $APP_NAME

echo ""
echo "📊 App URL: https://$APP_NAME.herokuapp.com"
echo "📋 App logs: heroku logs --tail -a $APP_NAME"
echo "⚙️  App config: heroku config -a $APP_NAME"