#!/usr/bin/env python3
"""
Quick Reddit Credential Tester
Paste your credentials here to test them immediately
"""

# PASTE YOUR CREDENTIALS HERE:
CLIENT_ID = "g_jvTMNIFhUrx?caD5NXg"  # Replace with exact copy from Reddit
CLIENT_SECRET = "vIfb_p4n0FQQ2fzC-LpIAO5xgjWCw"  # Replace with exact copy from Reddit

def test_reddit_quickly():
    try:
        import praw
        
        print(f"Testing Client ID: {CLIENT_ID}")
        print(f"Testing Secret: {CLIENT_SECRET[:10]}...")
        
        reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent='quick_test'
        )
        
        # Simple test
        subreddit = reddit.subreddit('bitcoin')
        posts = list(subreddit.hot(limit=1))
        
        if posts:
            print("✅ SUCCESS! Reddit API is working!")
            print(f"Sample post: {posts[0].title}")
            
            # Update .env file
            with open('.env', 'r') as f:
                content = f.read()
            
            content = content.replace('REDDIT_CLIENT_ID=g_jvTMNIFhUrx?caD5NXg', f'REDDIT_CLIENT_ID={CLIENT_ID}')
            content = content.replace('REDDIT_CLIENT_SECRET=vIfb_p4n0FQQ2fzC-LpIAO5xgjWCw', f'REDDIT_CLIENT_SECRET={CLIENT_SECRET}')
            
            with open('.env', 'w') as f:
                f.write(content)
                
            print("✅ Credentials updated in .env file!")
            return True
        else:
            print("❌ No posts found")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        if "401" in str(e):
            print("This means the credentials are incorrect.")
            print("Please double-check the Client ID and Secret from Reddit.")
        return False

if __name__ == "__main__":
    print("🧪 Quick Reddit API Test")
    print("=" * 30)
    test_reddit_quickly()