#!/usr/bin/env python3
"""
Direct Twitter API test - bypasses caching issues
"""

import os
import tweepy
from dotenv import load_dotenv

def test_direct_twitter_api():
    """Test Twitter API directly with explicit token loading."""
    
    print("🔍 DIRECT TWITTER API TEST")
    print("="*50)
    
    # Force reload environment
    load_dotenv(override=True)
    
    # Get token directly
    bearer_token = os.getenv("X_BEARER_TOKEN")
    
    print(f"✅ Token loaded: {bearer_token[:50] + '...' if bearer_token else 'None'}")
    
    if not bearer_token:
        print("❌ No Bearer Token found")
        return False
    
    # Check token format
    if bearer_token.startswith("AAAAAAAAAA"):
        if len(bearer_token) > 30 and "7HenjV8UVwh7urKW3X" in bearer_token:
            print("✅ This looks like your REAL token!")
        else:
            print("⚠️ This looks like a placeholder token")
    
    try:
        # Create Twitter client
        client = tweepy.Client(bearer_token=bearer_token)
        
        # Test with a simple API call
        print("🐦 Testing Twitter API connection...")
        
        # Search for recent tweets (minimal call)
        tweets = client.search_recent_tweets(
            query="bitcoin", 
            max_results=10,
            tweet_fields=["created_at", "text"]
        )
        
        if tweets and tweets.data:
            print(f"✅ SUCCESS! Retrieved {len(tweets.data)} tweets")
            print(f"📝 Sample tweet: {tweets.data[0].text[:100]}...")
            return True
        else:
            print("⚠️ API call succeeded but no tweets returned")
            return False
            
    except Exception as e:
        print(f"❌ Twitter API Error: {e}")
        
        # Check specific error types
        if "401" in str(e) or "Unauthorized" in str(e):
            print("🔑 Error: Unauthorized - Token might be invalid")
        elif "403" in str(e) or "Forbidden" in str(e):
            print("🚫 Error: Forbidden - App permissions issue")
        elif "Connection" in str(e):
            print("🌐 Error: Network connection issue")
        
        return False

if __name__ == "__main__":
    success = test_direct_twitter_api()
    
    if success:
        print("\n🎉 TWITTER API IS WORKING!")
        print("✅ Real sentiment extraction is now possible")
    else:
        print("\n❌ TWITTER API NOT WORKING")
        print("🔄 System will continue using simulated sentiment")
