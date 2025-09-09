#!/usr/bin/env python3
"""
FREE REDDIT SENTIMENT SCRAPER
Alternative to Twitter API for crypto sentiment analysis
Uses PRAW (Reddit API) - 1000 requests per minute, FREE!
"""

import praw
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import os
from dotenv import load_dotenv

load_dotenv()

class RedditCryptoScraper:
    """Free Reddit scraper for crypto sentiment."""
    
    def __init__(self):
        print("🔧 Initializing Reddit scraper (FREE alternative to Twitter)")
        
        # Reddit API is much more generous - 1000 requests/minute!
        self.reddit = None
        self.subreddits = [
            'bitcoin', 'cryptocurrency', 'CryptoCurrency', 
            'btc', 'BitcoinMarkets', 'CryptoMarkets'
        ]
        
        # Try to initialize Reddit (optional - can work without credentials)
        try:
            # You can get these FREE at https://www.reddit.com/prefs/apps
            self.reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID', 'your_client_id'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET', 'your_secret'),
                user_agent='bitcoin_sentiment_scraper'
            )
            print("✅ Reddit API initialized")
        except:
            print("⚠️ Reddit credentials not set - will use web scraping fallback")
    
    def scrape_reddit_posts(self, limit_per_sub=50):
        """Scrape recent Bitcoin posts from multiple subreddits."""
        if not self.reddit:
            return self.scrape_reddit_fallback()
        
        all_posts = []
        
        for subreddit_name in self.subreddits:
            try:
                print(f"🔍 Scraping r/{subreddit_name}...")
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Get hot posts
                for post in subreddit.hot(limit=limit_per_sub):
                    if self.is_bitcoin_related(post.title + " " + post.selftext):
                        all_posts.append({
                            'timestamp': datetime.fromtimestamp(post.created_utc),
                            'title': post.title,
                            'text': post.selftext,
                            'score': post.score,
                            'num_comments': post.num_comments,
                            'subreddit': subreddit_name,
                            'url': post.url,
                            'id': post.id
                        })
                
                print(f"✅ Found {len([p for p in all_posts if p['subreddit'] == subreddit_name])} Bitcoin posts")
                
            except Exception as e:
                print(f"⚠️ Error scraping r/{subreddit_name}: {e}")
        
        return pd.DataFrame(all_posts)
    
    def scrape_reddit_fallback(self):
        """Fallback web scraping method (no API needed)."""
        print("🌐 Using web scraping fallback method...")
        
        try:
            import requests
            from bs4 import BeautifulSoup
            
            all_posts = []
            
            for subreddit in ['bitcoin', 'cryptocurrency']:
                url = f"https://www.reddit.com/r/{subreddit}/hot.json"
                
                headers = {
                    'User-Agent': 'bitcoin-sentiment-analysis-bot'
                }
                
                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for post in data['data']['children'][:25]:  # First 25 posts
                        post_data = post['data']
                        
                        if self.is_bitcoin_related(post_data.get('title', '') + post_data.get('selftext', '')):
                            all_posts.append({
                                'timestamp': datetime.fromtimestamp(post_data['created_utc']),
                                'title': post_data['title'],
                                'text': post_data.get('selftext', ''),
                                'score': post_data['score'],
                                'num_comments': post_data['num_comments'],
                                'subreddit': subreddit,
                                'url': f"https://reddit.com{post_data['permalink']}",
                                'id': post_data['id']
                            })
                    
                    print(f"✅ Web scraped r/{subreddit}: {len([p for p in all_posts if p['subreddit'] == subreddit])} posts")
            
            return pd.DataFrame(all_posts)
            
        except ImportError:
            print("❌ requests/beautifulsoup not available. Install with: pip install requests beautifulsoup4")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Web scraping failed: {e}")
            return pd.DataFrame()
    
    def is_bitcoin_related(self, text):
        """Check if text is Bitcoin/crypto related."""
        bitcoin_keywords = [
            'bitcoin', 'btc', 'cryptocurrency', 'crypto', 'blockchain',
            'satoshi', 'hodl', 'mining', 'wallet', 'exchange'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in bitcoin_keywords)
    
    def clean_text(self, text):
        """Clean Reddit text for sentiment analysis."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove Reddit formatting
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.+?)\*', r'\1', text)      # Italic
        # Clean up
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def analyze_sentiment(self, df):
        """Add sentiment analysis to Reddit posts."""
        if df.empty:
            return df
        
        try:
            from textblob import TextBlob
            
            sentiments = []
            for _, row in df.iterrows():
                full_text = f"{row['title']} {row['text']}"
                cleaned_text = self.clean_text(full_text)
                
                blob = TextBlob(cleaned_text)
                sentiments.append({
                    'sentiment_polarity': blob.sentiment.polarity,
                    'sentiment_subjectivity': blob.sentiment.subjectivity,
                    'cleaned_text': cleaned_text
                })
            
            sentiment_df = pd.DataFrame(sentiments)
            result_df = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)
            
            print(f"✅ Added sentiment analysis to {len(result_df)} posts")
            return result_df
            
        except ImportError:
            print("⚠️ TextBlob not available. Install with: pip install textblob")
            return df
    
    def save_reddit_data(self, df, filename="reddit_crypto_sentiment.csv"):
        """Save Reddit data."""
        if df.empty:
            print("⚠️ No Reddit data to save")
            return False
        
        os.makedirs("data", exist_ok=True)
        filepath = f"data/{filename}"
        
        df['scraped_at'] = datetime.now()
        df.to_csv(filepath, index=False)
        
        print(f"✅ Saved {len(df)} Reddit posts → {filepath}")
        return True

def get_free_reddit_sentiment():
    """Main function to get Reddit sentiment data."""
    print("🚀 FREE REDDIT CRYPTO SENTIMENT SCRAPER")
    print("=" * 50)
    print("💡 Reddit API allows 1000 requests/minute - much better than Twitter!")
    
    scraper = RedditCryptoScraper()
    
    # Scrape posts
    posts_df = scraper.scrape_reddit_posts(limit_per_sub=30)
    
    if posts_df.empty:
        print("❌ No Reddit posts found")
        return None
    
    print(f"\n📊 REDDIT SCRAPING RESULTS:")
    print(f"Total posts: {len(posts_df)}")
    print(f"Subreddits: {posts_df['subreddit'].nunique()}")
    print(f"Date range: {posts_df['timestamp'].min()} to {posts_df['timestamp'].max()}")
    print(f"Avg score: {posts_df['score'].mean():.1f}")
    
    # Add sentiment analysis
    posts_with_sentiment = scraper.analyze_sentiment(posts_df)
    
    if 'sentiment_polarity' in posts_with_sentiment.columns:
        avg_sentiment = posts_with_sentiment['sentiment_polarity'].mean()
        print(f"Avg sentiment: {avg_sentiment:.3f}")
        
        # Sentiment distribution
        positive = len(posts_with_sentiment[posts_with_sentiment['sentiment_polarity'] > 0.1])
        negative = len(posts_with_sentiment[posts_with_sentiment['sentiment_polarity'] < -0.1])
        neutral = len(posts_with_sentiment) - positive - negative
        
        print(f"Sentiment distribution: {positive} positive, {neutral} neutral, {negative} negative")
    
    # Save data
    scraper.save_reddit_data(posts_with_sentiment)
    
    return posts_with_sentiment

if __name__ == "__main__":
    reddit_data = get_free_reddit_sentiment()
    
    if reddit_data is not None:
        print(f"\n🎉 SUCCESS! Reddit sentiment data collected")
        print(f"💡 This is a FREE alternative while waiting for Twitter quota reset")
        print(f"📅 Twitter quota resets: August 24, 2025")
        print(f"🔄 You can run this Reddit scraper daily for continuous sentiment data")
    else:
        print(f"❌ Reddit scraping failed. Check internet connection.")
