#!/usr/bin/env python3
"""
Twitter API Diagnostic Tool
Comprehensive testing of Twitter API credentials and access
"""

import tweepy
import os
from dotenv import load_dotenv
import time

load_dotenv()

def test_bearer_token():
    """Test Bearer Token with detailed error reporting."""
    print("🔍 TESTING BEARER TOKEN")
    print("=" * 40)
    
    bearer_token = os.getenv("X_BEARER_TOKEN")
    
    if not bearer_token:
        print("❌ No Bearer Token found in .env file")
        return False
    
    print(f"✅ Bearer Token found: {bearer_token[:20]}...{bearer_token[-10:]}")
    print(f"📏 Token length: {len(bearer_token)} characters")
    
    # Initialize client
    try:
        client = tweepy.Client(bearer_token=bearer_token)
        print("✅ Twitter client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return False
    
    # Test 1: Simple search
    print("\n🧪 TEST 1: Simple Search")
    try:
        response = client.search_recent_tweets(
            query="hello",
            max_results=10
        )
        
        if response.data:
            print(f"✅ Simple search successful - Found {len(response.data)} tweets")
        else:
            print("⚠️ Search successful but no tweets returned")
            
    except tweepy.Unauthorized as e:
        print(f"❌ Authentication failed: {e}")
        print("🔍 Your Bearer Token is invalid or expired")
        return False
    except tweepy.Forbidden as e:
        print(f"❌ Access forbidden: {e}")
        print("🔍 Your account may be suspended or lacks permissions")
        return False
    except tweepy.TooManyRequests as e:
        print(f"❌ Rate limit exceeded immediately: {e}")
        print("🔍 This suggests account issues or quota exhaustion")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    # Test 2: Get user info (if possible)
    print("\n🧪 TEST 2: User Info")
    try:
        me = client.get_me()
        if me.data:
            print(f"✅ User info retrieved: @{me.data.username}")
        else:
            print("⚠️ Could not retrieve user info (Bearer Token only)")
    except tweepy.Unauthorized:
        print("⚠️ Cannot get user info with Bearer Token only (normal)")
    except Exception as e:
        print(f"⚠️ User info error: {e}")
    
    # Test 3: Bitcoin search with more parameters
    print("\n🧪 TEST 3: Bitcoin Search")
    try:
        response = client.search_recent_tweets(
            query="bitcoin -is:retweet lang:en",
            max_results=10,
            tweet_fields=["created_at", "public_metrics"]
        )
        
        if response.data:
            print(f"✅ Bitcoin search successful - Found {len(response.data)} tweets")
            for tweet in response.data[:2]:
                print(f"  📝 {tweet.text[:50]}...")
        else:
            print("⚠️ Bitcoin search successful but no tweets returned")
            
    except Exception as e:
        print(f"❌ Bitcoin search failed: {e}")
        return False
    
    print("\n✅ All Bearer Token tests passed!")
    return True

def test_api_v1_credentials():
    """Test API v1.1 credentials if available."""
    print("\n🔍 TESTING API v1.1 CREDENTIALS")
    print("=" * 40)
    
    api_key = os.getenv("TWITTER_API_KEY")
    api_secret = os.getenv("TWITTER_API_SECRET")
    access_token = os.getenv("TWITTER_ACCESS_TOKEN")
    access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
    
    if not all([api_key, api_secret, access_token, access_token_secret]):
        print("⚠️ Not all v1.1 credentials found - skipping v1.1 tests")
        return True
    
    print("✅ All v1.1 credentials found")
    
    try:
        auth = tweepy.OAuthHandler(api_key, api_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True)
        
        # Test authentication
        user = api.verify_credentials()
        if user:
            print(f"✅ v1.1 API authenticated as: @{user.screen_name}")
        else:
            print("❌ v1.1 API authentication failed")
            return False
            
    except Exception as e:
        print(f"❌ v1.1 API error: {e}")
        return False
    
    print("✅ API v1.1 tests passed!")
    return True

def check_rate_limits():
    """Check current rate limit status."""
    print("\n🔍 CHECKING RATE LIMITS")
    print("=" * 40)
    
    bearer_token = os.getenv("X_BEARER_TOKEN")
    if not bearer_token:
        print("❌ No Bearer Token for rate limit check")
        return
    
    try:
        client = tweepy.Client(bearer_token=bearer_token)
        
        # Make a simple request to check limits
        response = client.search_recent_tweets(query="test", max_results=10)
        
        # Check rate limit headers if available
        if hasattr(response, 'meta') and response.meta:
            print("📊 Rate limit info from response:")
            print(f"  Remaining: {response.meta.get('remaining', 'Unknown')}")
            print(f"  Reset time: {response.meta.get('reset_time', 'Unknown')}")
        else:
            print("ℹ️ Rate limit headers not available in response")
        
    except tweepy.TooManyRequests as e:
        print(f"❌ Rate limit exceeded: {e}")
        print("🔍 Your account may have exhausted its quota")
    except Exception as e:
        print(f"⚠️ Could not check rate limits: {e}")

def main():
    """Run all diagnostic tests."""
    print("🚀 TWITTER API DIAGNOSTIC TOOL")
    print("=" * 50)
    
    # Test Bearer Token (most important)
    bearer_success = test_bearer_token()
    
    if not bearer_success:
        print("\n❌ DIAGNOSTIC FAILED")
        print("🔍 Primary issue: Bearer Token authentication")
        print("\n💡 RECOMMENDED ACTIONS:")
        print("1. Verify your Twitter Developer account is active")
        print("2. Check if your app has been suspended")
        print("3. Regenerate your Bearer Token")
        print("4. Ensure you have API v2 access")
        return
    
    # Test v1.1 credentials (optional)
    test_api_v1_credentials()
    
    # Check rate limits
    check_rate_limits()
    
    print("\n✅ DIAGNOSTICS COMPLETE")
    print("🎉 Your Twitter API setup appears to be working!")

if __name__ == "__main__":
    main()
