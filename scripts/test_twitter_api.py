#!/usr/bin/env python3
"""
Test Twitter API connection and sentiment extraction capability.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sentiment_analysis import SentimentAnalyzer
from scrape import TwitterScraper
from datetime import datetime

def test_twitter_api():
    """Test Twitter API connection and data extraction."""
    
    print("🔍 TWITTER API TESTING")
    print("=" * 50)
    
    # Test 1: Check API credentials
    print("\n1️⃣ Checking API Credentials...")
    bearer_token = os.getenv("X_BEARER_TOKEN")
    
    if bearer_token:
        print(f"✅ Bearer Token found: {bearer_token[:20]}...")
        if bearer_token.startswith("AAAAAAAAAA"):
            print("⚠️  This looks like a placeholder token")
        else:
            print("✅ Token format looks valid")
    else:
        print("❌ No Bearer Token found")
    
    # Test 2: Test TwitterScraper
    print("\n2️⃣ Testing TwitterScraper...")
    try:
        scraper = TwitterScraper()
        api_status = scraper.check_api_status()
        print(f"📊 API Status: {api_status}")
        
        if api_status["status"] == "active":
            print("🐦 Attempting to scrape live tweets...")
            tweets = scraper.get_live_sentiment_data()
            print(f"✅ Retrieved {len(tweets)} tweets")
            
            if len(tweets) > 0:
                print("📝 Sample tweet sentiment:")
                sample = tweets.iloc[0]
                print(f"   Text: {sample.get('text', 'N/A')[:100]}...")
                print(f"   Sentiment: {sample.get('compound', 'N/A')}")
        else:
            print(f"❌ API not active: {api_status['message']}")
            
    except Exception as e:
        print(f"❌ TwitterScraper error: {e}")
    
    # Test 3: Test SentimentAnalyzer
    print("\n3️⃣ Testing SentimentAnalyzer...")
    try:
        analyzer = SentimentAnalyzer()
        sentiment_features = analyzer.create_sentiment_for_prediction()
        
        print(f"✅ Generated sentiment features: {len(sentiment_features)} rows")
        
        if not sentiment_features.empty:
            print("📊 Sentiment Summary:")
            print(f"   Compound Score: {sentiment_features['compound_mean'].iloc[0]:.3f}")
            print(f"   Positive Ratio: {sentiment_features.get('sentiment_positive_ratio', [0]).iloc[0]:.3f}")
            print(f"   Negative Ratio: {sentiment_features.get('sentiment_negative_ratio', [0]).iloc[0]:.3f}")
            
    except Exception as e:
        print(f"❌ SentimentAnalyzer error: {e}")
    
    # Test 4: Overall system status
    print("\n4️⃣ System Status Summary")
    print("=" * 30)
    
    if bearer_token and not bearer_token.startswith("AAAAAAAAAA"):
        print("🟢 Ready for REAL Twitter sentiment extraction")
        print("🎯 Next: Run `python scripts/live_sentiment_pipeline.py`")
    else:
        print("🟡 Using SIMULATED sentiment (fallback mode)")
        print("🔧 To enable real Twitter data:")
        print("   1. Get Twitter Developer Account")
        print("   2. Update X_BEARER_TOKEN in .env")
        print("   3. See TWITTER_API_SETUP.md for details")

if __name__ == "__main__":
    test_twitter_api()
