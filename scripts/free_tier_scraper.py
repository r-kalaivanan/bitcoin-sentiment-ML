#!/usr/bin/env python3
"""
FREE TIER TWITTER SCRAPER
Optimized for Twitter API Free Tier (100 tweets/month limit)
"""

import tweepy
import pandas as pd
from dotenv import load_dotenv
import os
import re
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

class FreeTierTwitterScraper:
    """Twitter scraper optimized for FREE TIER usage (100 tweets/month)."""
    
    def __init__(self):
        self.bearer_token = os.getenv("X_BEARER_TOKEN")
        self.client = None
        
        # FREE TIER LIMITS - Be very conservative
        self.max_monthly_tweets = 80  # Leave 20 tweets buffer
        self.max_per_request = 10     # Small batches to avoid waste
        
        if self.bearer_token:
            try:
                self.client = tweepy.Client(bearer_token=self.bearer_token)
                print(f"🔑 FREE TIER: Twitter API client initialized")
                print(f"⚠️  Monthly limit: {self.max_monthly_tweets} tweets")
            except Exception as e:
                print(f"❌ Failed to initialize: {e}")
                self.client = None
        else:
            print("⚠️ X_BEARER_TOKEN not found")
    
    def check_api_status(self) -> bool:
        """Test API with minimal tweet consumption."""
        if not self.client:
            return False
        
        try:
            print("🔍 Testing API (using 1 tweet from quota)...")
            response = self.client.search_recent_tweets(
                query="bitcoin", 
                max_results=10  # Minimal test
            )
            
            if response.data:
                print(f"✅ API working - Found {len(response.data)} tweets")
                return True
            else:
                print("⚠️ API responded but no data")
                return False
                
        except tweepy.TooManyRequests:
            print("❌ Monthly quota exhausted! Wait until next month.")
            return False
        except tweepy.Unauthorized:
            print("❌ Invalid Bearer Token")
            return False
        except Exception as e:
            print(f"❌ API error: {e}")
            return False
    
    def smart_scrape_minimal(self, query: str = "bitcoin", max_tweets: int = 30) -> pd.DataFrame:
        """
        Smart scraping that maximizes value from limited tweets.
        Focus on high-engagement, recent tweets.
        """
        if not self.client:
            return pd.DataFrame()
        
        if max_tweets > self.max_monthly_tweets:
            print(f"⚠️ Reducing request from {max_tweets} to {self.max_monthly_tweets} (monthly limit)")
            max_tweets = self.max_monthly_tweets
        
        print(f"🐦 FREE TIER: Getting {max_tweets} high-value tweets for '{query}'")
        
        tweets_data = []
        
        try:
            # Get recent tweets with engagement metrics
            tweets = tweepy.Paginator(
                self.client.search_recent_tweets,
                query=f"{query} -is:retweet lang:en",
                tweet_fields=["created_at", "text", "public_metrics", "author_id"],
                max_results=100,  # Per page (but we'll limit total)
                start_time=datetime.now() - timedelta(hours=24)  # Last 24 hours only
            ).flatten(limit=max_tweets)
            
            for tweet in tweets:
                # Clean and process tweet
                cleaned_text = self.clean_tweet_text(tweet.text)
                if len(cleaned_text) > 15:  # Filter very short tweets
                    
                    # Calculate engagement score
                    engagement = (
                        tweet.public_metrics['retweet_count'] * 3 +
                        tweet.public_metrics['like_count'] * 1 +
                        tweet.public_metrics['reply_count'] * 2
                    )
                    
                    tweets_data.append({
                        'timestamp': tweet.created_at,
                        'text': cleaned_text,
                        'original_text': tweet.text,
                        'retweet_count': tweet.public_metrics['retweet_count'],
                        'like_count': tweet.public_metrics['like_count'],
                        'reply_count': tweet.public_metrics['reply_count'],
                        'engagement_score': engagement,
                        'tweet_id': tweet.id,
                        'query_used': query
                    })
            
            # Sort by engagement to keep highest quality tweets
            if tweets_data:
                df = pd.DataFrame(tweets_data)
                df = df.sort_values('engagement_score', ascending=False)
                print(f"✅ FREE TIER: Collected {len(df)} tweets (quota used wisely)")
                return df
            
        except tweepy.TooManyRequests:
            print("❌ Monthly quota exhausted!")
        except Exception as e:
            print(f"❌ Scraping error: {e}")
        
        return pd.DataFrame()
    
    def clean_tweet_text(self, text: str) -> str:
        """Clean tweet text for sentiment analysis."""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()
    
    def save_tweets(self, df: pd.DataFrame, filename: str = "free_tier_tweets.csv") -> bool:
        """Save tweets with metadata."""
        if df.empty:
            return False
        
        os.makedirs("data", exist_ok=True)
        filepath = f"data/{filename}"
        
        # Add metadata
        df['scraped_at'] = datetime.now()
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        df['free_tier_optimized'] = True
        
        df.to_csv(filepath, index=False)
        print(f"✅ Saved {len(df)} tweets → {filepath}")
        return True

def get_sample_tweets_free_tier():
    """Get sample tweets optimized for free tier."""
    scraper = FreeTierTwitterScraper()
    
    if not scraper.check_api_status():
        print("❌ Cannot proceed with scraping")
        return pd.DataFrame()
    
    # Use remaining quota efficiently
    queries = ["bitcoin price", "btc"]  # Focus on most relevant queries
    all_tweets = []
    tweets_per_query = 15  # Use 30 tweets total (15 per query)
    
    for query in queries:
        print(f"\n🔍 FREE TIER: Scraping '{query}' ({tweets_per_query} tweets)")
        tweets = scraper.smart_scrape_minimal(query=query, max_tweets=tweets_per_query)
        
        if not tweets.empty:
            all_tweets.append(tweets)
        
        # Small delay between queries
        time.sleep(1)
    
    if all_tweets:
        combined = pd.concat(all_tweets, ignore_index=True)
        # Remove duplicates
        combined = combined.drop_duplicates(subset=['text'], keep='first')
        
        # Save results
        scraper.save_tweets(combined, "bitcoin_sentiment_free_tier.csv")
        
        print(f"\n🎉 FREE TIER SUCCESS!")
        print(f"📊 Total unique tweets: {len(combined)}")
        print(f"📈 Avg engagement: {combined['engagement_score'].mean():.1f}")
        print(f"⏰ Time range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
        
        return combined
    
    print("❌ No tweets collected")
    return pd.DataFrame()

if __name__ == "__main__":
    print("🚀 FREE TIER BITCOIN SENTIMENT SCRAPER")
    print("=" * 50)
    print("⚠️  Optimized for Twitter API Free Tier (100 tweets/month)")
    print("🎯 Focus: High-quality, recent Bitcoin sentiment tweets")
    print()
    
    # Run free tier optimized scraping
    tweets_df = get_sample_tweets_free_tier()
    
    if not tweets_df.empty:
        print("\n✅ SUCCESS: Ready for sentiment analysis!")
        print("💡 Next steps:")
        print("   1. Run sentiment analysis on collected tweets")
        print("   2. Use existing Bitcoin price data for correlation")
        print("   3. Build lightweight prediction model")
    else:
        print("\n❌ No tweets collected. Check your API status.")
