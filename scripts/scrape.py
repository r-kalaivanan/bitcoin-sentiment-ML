# scripts/scrape.py - Enhanced Tweet Scraping

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

class TwitterScraper:
    """Enhanced Twitter scraper for Bitcoin-related tweets."""
    
    def __init__(self):
        self.bearer_token = os.getenv("X_BEARER_TOKEN")
        self.client = tweepy.Client(bearer_token=self.bearer_token) if self.bearer_token else None
    
    def clean_tweet_text(self, text):
        """Clean tweet text for better sentiment analysis."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def scrape_recent_tweets(self, query="bitcoin OR btc OR cryptocurrency", max_results=1000, days_back=7):
        """
        Scrape recent tweets with enhanced features.
        
        Args:
            query: Search query
            max_results: Maximum tweets to collect
            days_back: Days to look back
        """
        if not self.client:
            print("❌ Twitter API credentials not found. Set X_BEARER_TOKEN in .env file")
            return pd.DataFrame()
        
        print(f"🐦 Scraping tweets: '{query}' (last {days_back} days)")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        tweets_data = []
        
        try:
            tweets = tweepy.Paginator(
                self.client.search_recent_tweets,
                query=f"{query} -is:retweet lang:en",
                tweet_fields=["created_at", "text", "public_metrics", "lang"],
                max_results=100,
                start_time=start_time,
                end_time=end_time
            ).flatten(limit=max_results)
            
            for tweet in tweets:
                if tweet.lang == "en":
                    cleaned_text = self.clean_tweet_text(tweet.text)
                    if len(cleaned_text) > 10:
                        tweets_data.append({
                            'timestamp': tweet.created_at,
                            'text': cleaned_text,
                            'original_text': tweet.text,
                            'retweet_count': tweet.public_metrics['retweet_count'],
                            'like_count': tweet.public_metrics['like_count'],
                            'reply_count': tweet.public_metrics['reply_count'],
                            'quote_count': tweet.public_metrics['quote_count']
                        })
            
            print(f"✅ Scraped {len(tweets_data)} tweets")
            
        except Exception as e:
            print(f"❌ Error scraping tweets: {str(e)}")
        
        return pd.DataFrame(tweets_data)
    
    def save_tweets(self, tweets_df, filename="raw_tweets.csv"):
        """Save tweets to CSV file."""
        if tweets_df.empty:
            print("⚠️ No tweets to save")
            return False
        
        os.makedirs("data", exist_ok=True)
        filepath = f"data/{filename}"
        tweets_df.to_csv(filepath, index=False)
        print(f"✅ Saved {len(tweets_df)} tweets → {filepath}")
        return True

def scrape_recent_tweets(query="bitcoin OR btc", max_results=100):
    """Legacy function for backward compatibility."""
    scraper = TwitterScraper()
    return scraper.scrape_recent_tweets(query, max_results)

if __name__ == "__main__":
    # Enhanced scraping
    scraper = TwitterScraper()
    
    # Scrape with different queries
    queries = [
        "bitcoin OR btc",
        "cryptocurrency OR crypto",
        "#bitcoin OR #btc OR #cryptocurrency"
    ]
    
    all_tweets = []
    
    for query in queries:
        print(f"\n📡 Scraping: {query}")
        tweets = scraper.scrape_recent_tweets(query=query, max_results=500, days_back=3)
        if not tweets.empty:
            tweets['query_used'] = query
            all_tweets.append(tweets)
        time.sleep(1)  # Rate limiting
    
    if all_tweets:
        # Combine all tweets
        combined_tweets = pd.concat(all_tweets, ignore_index=True)
        
        # Remove duplicates
        combined_tweets = combined_tweets.drop_duplicates(subset=['text'])
        
        # Save combined tweets
        scraper.save_tweets(combined_tweets, "combined_raw_tweets.csv")
        
        print(f"\n📊 SCRAPING SUMMARY:")
        print(f"Total unique tweets: {len(combined_tweets)}")
        print(f"Date range: {combined_tweets['timestamp'].min()} to {combined_tweets['timestamp'].max()}")
        print(f"Average engagement: {combined_tweets[['like_count', 'retweet_count']].mean().sum():.1f}")
    else:
        print("⚠️ No tweets collected")
