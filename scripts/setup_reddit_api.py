#!/usr/bin/env python3
"""
Reddit API Setup Helper
This script helps you set up Reddit API credentials step by step
"""

import os
import sys
from pathlib import Path

def update_env_file(client_id, client_secret):
    """Update the .env file with Reddit credentials"""
    env_path = Path('.env')
    
    if not env_path.exists():
        print("❌ .env file not found. Please run: copy .env.template .env")
        return False
    
    # Read current content
    with open(env_path, 'r') as f:
        content = f.read()
    
    # Replace Reddit credentials
    content = content.replace('REDDIT_CLIENT_ID=your_reddit_client_id_here', f'REDDIT_CLIENT_ID={client_id}')
    content = content.replace('REDDIT_CLIENT_SECRET=your_reddit_client_secret_here', f'REDDIT_CLIENT_SECRET={client_secret}')
    
    # Write updated content
    with open(env_path, 'w') as f:
        f.write(content)
    
    print("✅ Reddit credentials updated in .env file")
    return True

def test_reddit_credentials(client_id, client_secret):
    """Test Reddit credentials"""
    try:
        import praw
        
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent='bitcoin_sentiment_setup_test'
        )
        
        # Test by accessing a subreddit
        subreddit = reddit.subreddit('bitcoin')
        posts = list(subreddit.hot(limit=1))
        
        if posts:
            print("✅ Reddit API credentials are working!")
            print(f"   Test post: {posts[0].title[:50]}...")
            return True
        else:
            print("⚠️ Reddit API connected but no posts found")
            return False
            
    except Exception as e:
        print(f"❌ Reddit API test failed: {e}")
        if "received 401 HTTP response" in str(e):
            print("   This usually means invalid credentials")
        return False

def main():
    """Main setup process"""
    print("🚀 Reddit API Setup Helper")
    print("=" * 40)
    print()
    
    print("📚 First, you need to create a Reddit app:")
    print("1. Go to: https://www.reddit.com/prefs/apps")
    print("2. Click 'Create App' or 'Create Another App'")
    print("3. Fill in:")
    print("   - Name: Bitcoin Sentiment Analyzer")
    print("   - App type: script")
    print("   - Description: ML sentiment analysis")
    print("   - Redirect URI: http://localhost:8080")
    print()
    
    input("Press Enter after you've created the Reddit app...")
    print()
    
    print("🔑 Now enter your Reddit app credentials:")
    print("(You can find these on the app page you just created)")
    print()
    
    while True:
        client_id = input("📝 Client ID (14 characters under 'personal use script'): ").strip()
        
        if len(client_id) != 14:
            print("⚠️ Client ID should be 14 characters. Please check and try again.")
            continue
        break
    
    while True:
        client_secret = input("🔐 Client Secret (27 characters in 'secret' field): ").strip()
        
        if len(client_secret) != 27:
            print("⚠️ Client Secret should be 27 characters. Please check and try again.")
            continue
        break
    
    print()
    print("🧪 Testing your Reddit credentials...")
    
    if test_reddit_credentials(client_id, client_secret):
        print()
        print("💾 Saving credentials to .env file...")
        
        if update_env_file(client_id, client_secret):
            print()
            print("🎉 Reddit API setup complete!")
            print()
            print("🔄 You can now run the sentiment analyzer to get real Reddit data:")
            print("   python scripts/enhanced_sentiment_analyzer.py")
            print()
            print("🧪 Or test all your APIs:")
            print("   python scripts/api_test.py")
        else:
            print("❌ Failed to update .env file")
    else:
        print()
        print("❌ Reddit credentials are not working. Please check:")
        print("1. Client ID is correct (14 characters)")
        print("2. Client Secret is correct (27 characters)")
        print("3. App type is set to 'script'")
        print()
        print("🔄 Run this script again to retry.")

if __name__ == "__main__":
    main()