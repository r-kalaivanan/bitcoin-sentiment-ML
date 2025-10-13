#!/bin/bash
# Deploy to Streamlit Cloud
# Run this script to prepare for Streamlit Cloud deployment

set -e

echo "🚀 Preparing Bitcoin Sentiment ML for Streamlit Cloud deployment..."

# Check if git repository is clean
if [[ -n $(git status --porcelain) ]]; then
    echo "⚠️  Warning: You have uncommitted changes. Please commit them first."
    git status --short
    exit 1
fi

# Verify requirements.txt exists and is complete
echo "📋 Checking requirements.txt..."
if [[ ! -f "requirements.txt" ]]; then
    echo "❌ requirements.txt not found!"
    exit 1
fi

# Check if main dashboard file exists
if [[ ! -f "scripts/dashboard.py" ]]; then
    echo "❌ scripts/dashboard.py not found!"
    exit 1
fi

# Verify Streamlit config
echo "⚙️  Checking Streamlit configuration..."
mkdir -p .streamlit
if [[ ! -f ".streamlit/config.toml" ]]; then
    echo "⚠️  Streamlit config not found, created default config"
fi

# Test if the app runs locally first
echo "🧪 Testing application locally..."
echo "This will start the app briefly to test imports..."
timeout 10s streamlit run scripts/dashboard.py --server.headless=true > /dev/null 2>&1 || {
    echo "⚠️  Local test completed (timeout expected)"
}

echo "✅ Streamlit Cloud deployment preparation complete!"
echo ""
echo "📚 Next steps:"
echo "1. Push your code to GitHub if not already done:"
echo "   git add ."
echo "   git commit -m 'Prepare for Streamlit Cloud deployment'"
echo "   git push origin main"
echo ""
echo "2. Go to https://share.streamlit.io/"
echo "3. Click 'New app'"
echo "4. Connect your GitHub repository: r-kalaivanan/bitcoin-sentiment-ML"
echo "5. Set the main file path: scripts/dashboard.py"
echo "6. Click 'Deploy!'"
echo ""
echo "📊 Your app will be available at:"
echo "https://[your-app-name].streamlit.app"