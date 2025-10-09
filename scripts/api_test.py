#!/usr/bin/env python3
"""
API Connection Test Script
Tests all API connections for the Bitcoin Sentiment ML system
"""

import os
import sys
from dotenv import load_dotenv
import tweepy
import praw
import requests

# Load environment variables
load_dotenv()

def test_twitter_api():
    """Test Twitter API connection"""
    print("🐦 Testing Twitter API...")
    
    try:
        # Try both possible environment variable names
        bearer_token = os.getenv('X_BEARER_TOKEN') or os.getenv('TWITTER_BEARER_TOKEN')
        
        if not bearer_token:
            print("❌ Twitter Bearer Token not found in .env file")
            return False
            
        client = tweepy.Client(bearer_token=bearer_token)
        
        # Test with a simple search
        tweets = client.search_recent_tweets(
            query="bitcoin OR BTC -is:retweet", 
            max_results=10
        )
        
        if tweets.data:
            print(f"✅ Twitter API working! Found {len(tweets.data)} tweets")
            print(f"   Sample tweet: {tweets.data[0].text[:100]}...")
            return True
        else:
            print("⚠️ Twitter API connected but no tweets found")
            return False
            
    except Exception as e:
        print(f"❌ Twitter API error: {e}")
        return False

def test_reddit_api():
    """Test Reddit API connection"""
    print("\n📱 Testing Reddit API...")
    
    try:
        client_id = os.getenv('REDDIT_CLIENT_ID')
        client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        
        if not client_id or not client_secret:
            print("❌ Reddit credentials not found in .env file")
            print("   Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET")
            return False
            
        if client_id == 'your_reddit_client_id_here':
            print("❌ Reddit credentials not updated from template")
            print("   Please follow the Reddit API setup guide")
            return False
            
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent='bitcoin_sentiment_analyzer_test'
        )
        
        # Test by accessing a subreddit
        subreddit = reddit.subreddit('bitcoin')
        posts = list(subreddit.hot(limit=5))
        
        if posts:
            print(f"✅ Reddit API working! Found {len(posts)} posts from r/bitcoin")
            print(f"   Sample post: {posts[0].title[:100]}...")
            return True
        else:
            print("⚠️ Reddit API connected but no posts found")
            return False
            
    except Exception as e:
        print(f"❌ Reddit API error: {e}")
        if "received 401 HTTP response" in str(e):
            print("   This usually means invalid credentials")
        return False

def test_news_api():
    """Test News API connection"""
    print("\n📰 Testing News API...")
    
    try:
        api_key = os.getenv('NEWS_API_KEY')
        
        if not api_key or api_key == 'your_news_api_key_here':
            print("⚠️ News API key not configured - this is optional")
            return True  # Not a failure since it's optional
            
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': 'bitcoin',
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 5,
            'apiKey': api_key
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            
            if articles:
                print(f"✅ News API working! Found {len(articles)} articles")
                print(f"   Sample headline: {articles[0]['title'][:100]}...")
                return True
            else:
                print("⚠️ News API connected but no articles found")
                return False
        else:
            print(f"❌ News API error: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ News API error: {e}")
        return False

def test_database_connection():
    """Test database connection"""
    print("\n💾 Testing Database Connection...")
    
    try:
        import sqlite3
        
        # Test connection to sentiment database
        conn = sqlite3.connect('data/sentiment_data.db')
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sentiment_scores'")
        table_exists = cursor.fetchone()
        
        if table_exists:
            # Count records
            cursor.execute("SELECT COUNT(*) FROM sentiment_scores")
            count = cursor.fetchone()[0]
            print(f"✅ Sentiment database working! Contains {count} records")
        else:
            print("⚠️ Sentiment database table not found - will be created on first run")
            
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def main():
    """Run all API tests"""
    print("🚀 Bitcoin Sentiment ML - API Connection Test")
    print("=" * 60)
    
    results = {
        'twitter': test_twitter_api(),
        'reddit': test_reddit_api(),
        'news': test_news_api(),
        'database': test_database_connection()
    }
    
    print("\n📊 Test Summary:")
    print("=" * 30)
    
    working_apis = 0
    total_apis = len(results)
    
    for api, status in results.items():
        status_emoji = "✅" if status else "❌"
        print(f"{status_emoji} {api.title()}: {'Working' if status else 'Failed'}")
        if status:
            working_apis += 1
    
    print(f"\n🎯 {working_apis}/{total_apis} services are working properly")
    
    if results['twitter'] and results['database']:
        print("\n🎉 Minimum requirements met! You can collect real sentiment data.")
    elif results['database']:
        print("\n⚠️ Using mock data for sentiment analysis (APIs not configured)")
    else:
        print("\n❌ Database issues detected. Please check your setup.")
    
    print("\n📚 Next Steps:")
    if not results['twitter']:
        print("  - Check your Twitter API credentials in .env file")
    if not results['reddit']:
        print("  - Set up Reddit API following the guide in API_SETUP_GUIDE.md")
    if not results['news']:
        print("  - (Optional) Set up News API for additional sentiment sources")
    
    print("\n🔄 To start collecting real sentiment data, run:")
    print("  python scripts/enhanced_sentiment_analyzer.py")

if __name__ == "__main__":
    main()