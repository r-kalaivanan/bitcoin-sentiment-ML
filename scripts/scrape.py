# scripts/scrape.py - Enhanced Tweet Scraping for Real-time Sentiment

import tweepy
import pandas as pd
from dotenv import load_dotenv
import os
import re
from datetime import datetime, timedelta
import time
import warnings
import json
from typing import Optional, List, Dict
warnings.filterwarnings('ignore')

load_dotenv()

class TwitterScraper:
    """Enhanced Twitter scraper for Bitcoin-related tweets with production features."""
    
    def __init__(self):
        self.bearer_token = os.getenv("X_BEARER_TOKEN")
        self.client = None
        self.max_results = int(os.getenv("TWITTER_MAX_RESULTS", 1000))
        self.days_back = int(os.getenv("TWITTER_DAYS_BACK", 7))
        
        # Initialize Twitter client with Bearer Token only (API v2)
        if self.bearer_token:
            try:
                # Validate Bearer Token format
                if not self.bearer_token.startswith('AAAAAAAAAAAAAAAAAAAAAA'):
                    print("⚠️ Bearer Token doesn't match expected format")
                
                self.client = tweepy.Client(bearer_token=self.bearer_token)
                print(f"🔑 Twitter API client initialized with Bearer Token")
                print(f"🔍 Token preview: {self.bearer_token[:20]}...")
            except Exception as e:
                print(f"❌ Failed to initialize Twitter client: {e}")
                self.client = None
        else:
            print("⚠️ X_BEARER_TOKEN not found in environment variables")
        
        # Rate limiting tracking
        self.last_request_time = 0
        self.requests_made = 0
        
    def check_api_status(self) -> bool:
        """Check if Twitter API is accessible with retry logic."""
        if not self.client:
            print("❌ Twitter API client not initialized. Check X_BEARER_TOKEN in .env file")
            return False
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"🔍 Testing Twitter API connection (attempt {attempt + 1}/{max_retries})...")
                
                # Simple test query with minimal parameters
                test_response = self.client.search_recent_tweets(
                    query="bitcoin", 
                    max_results=10
                )
                
                if test_response.data:
                    print("✅ Twitter API connection successful")
                    print(f"📊 Test returned {len(test_response.data)} tweets")
                    return True
                else:
                    print("⚠️ API responded but no data returned")
                    
            except tweepy.Unauthorized as e:
                print(f"❌ Twitter API authentication failed: {str(e)}")
                print("🔍 Check your Bearer Token in .env file")
                return False
            except tweepy.Forbidden as e:
                print(f"❌ Twitter API access forbidden: {str(e)}")
                print("🔍 Check your API permissions and account status")
                return False
            except tweepy.TooManyRequests as e:
                print(f"⚠️ Rate limit exceeded: {str(e)}")
                print("🔍 This might indicate account suspension or quota exhaustion")
                if attempt < max_retries - 1:
                    print("⏳ Waiting 60 seconds before retry...")
                    time.sleep(60)
                continue
            except tweepy.BadRequest as e:
                print(f"❌ Bad request error: {str(e)}")
                print("🔍 The API request format might be incorrect")
                return False
            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1} failed: {str(e)}")
                print(f"🔍 Error type: {type(e).__name__}")
                if attempt < max_retries - 1:
                    print("🔄 Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print("❌ All connection attempts failed")
                    return False
        
        return False
    
    def clean_tweet_text(self, text: str) -> str:
        """Clean tweet text for better sentiment analysis."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions and hashtags (but preserve the content)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)  # Keep hashtag content
        # Remove extra whitespace and special characters
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()
    
    def rate_limit_handler(self):
        """Handle API rate limiting."""
        current_time = time.time()
        if current_time - self.last_request_time < 1:  # Wait at least 1 second between requests
            time.sleep(1 - (current_time - self.last_request_time))
        self.last_request_time = time.time()
        self.requests_made += 1
    
    def scrape_recent_tweets(self, query: str = "bitcoin OR btc OR cryptocurrency", 
                           max_results: int = None, days_back: int = None) -> pd.DataFrame:
        """
        Scrape recent tweets with enhanced features and error handling.
        
        Args:
            query: Search query for tweets
            max_results: Maximum tweets to collect (defaults to env setting)
            days_back: Days to look back (defaults to env setting)
        
        Returns:
            DataFrame with tweet data
        """
        if not self.client:
            print("❌ Twitter API credentials not found. Set X_BEARER_TOKEN in .env file")
            return pd.DataFrame()
        
        max_results = max_results or self.max_results
        days_back = days_back or self.days_back
        
        print(f"🐦 Scraping tweets: '{query}' (last {days_back} days, max {max_results})")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        tweets_data = []
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                self.rate_limit_handler()
                
                tweets = tweepy.Paginator(
                    self.client.search_recent_tweets,
                    query=f"{query} -is:retweet lang:en",
                    tweet_fields=["created_at", "text", "public_metrics", "lang", "context_annotations"],
                    max_results=100,  # Per page
                    start_time=start_time,
                    end_time=end_time
                ).flatten(limit=max_results)
                
                for tweet in tweets:
                    if tweet.lang == "en":
                        cleaned_text = self.clean_tweet_text(tweet.text)
                        if len(cleaned_text) > 10:  # Filter out very short tweets
                            # Calculate engagement score
                            engagement = (
                                tweet.public_metrics['retweet_count'] * 3 +
                                tweet.public_metrics['like_count'] * 1 +
                                tweet.public_metrics['reply_count'] * 2 +
                                tweet.public_metrics.get('quote_count', 0) * 2
                            )
                            
                            tweets_data.append({
                                'timestamp': tweet.created_at,
                                'text': cleaned_text,
                                'original_text': tweet.text,
                                'retweet_count': tweet.public_metrics['retweet_count'],
                                'like_count': tweet.public_metrics['like_count'],
                                'reply_count': tweet.public_metrics['reply_count'],
                                'quote_count': tweet.public_metrics.get('quote_count', 0),
                                'engagement_score': engagement,
                                'tweet_id': tweet.id
                            })
                
                print(f"✅ Scraped {len(tweets_data)} tweets successfully")
                break  # Success, exit retry loop
                
            except tweepy.TooManyRequests:
                print(f"⚠️ Rate limit hit. Waiting 15 minutes... (Retry {retry_count + 1}/{max_retries})")
                time.sleep(900)  # Wait 15 minutes
                retry_count += 1
                
            except Exception as e:
                print(f"❌ Error scraping tweets (Retry {retry_count + 1}/{max_retries}): {str(e)}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(10)  # Wait 10 seconds before retry
        
        if retry_count >= max_retries:
            print("❌ Max retries reached. Returning empty DataFrame.")
            return pd.DataFrame()
        
        return pd.DataFrame(tweets_data)
    
    def save_tweets(self, tweets_df: pd.DataFrame, filename: str = "raw_tweets.csv") -> bool:
        """Save tweets to CSV file with metadata."""
        if tweets_df.empty:
            print("⚠️ No tweets to save")
            return False
        
        os.makedirs("data", exist_ok=True)
        filepath = f"data/{filename}"
        
        # Add metadata
        tweets_df['scraped_at'] = datetime.now()
        tweets_df['date'] = pd.to_datetime(tweets_df['timestamp']).dt.date
        
        tweets_df.to_csv(filepath, index=False)
        print(f"✅ Saved {len(tweets_df)} tweets → {filepath}")
        return True
    
    def scrape_for_date_range(self, start_date: str, end_date: str, 
                             query: str = "bitcoin OR btc") -> pd.DataFrame:
        """
        Scrape tweets for a specific date range (for historical analysis).
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            query: Search query
        """
        if not self.client:
            print("❌ Twitter API credentials not found")
            return pd.DataFrame()
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        print(f"🐦 Scraping tweets from {start_date} to {end_date}")
        
        tweets_data = []
        
        try:
            self.rate_limit_handler()
            
            tweets = tweepy.Paginator(
                self.client.search_recent_tweets,
                query=f"{query} -is:retweet lang:en",
                tweet_fields=["created_at", "text", "public_metrics", "lang"],
                max_results=100,
                start_time=start_dt,
                end_time=end_dt
            ).flatten(limit=self.max_results)
            
            for tweet in tweets:
                if tweet.lang == "en":
                    cleaned_text = self.clean_tweet_text(tweet.text)
                    if len(cleaned_text) > 10:
                        engagement = (
                            tweet.public_metrics['retweet_count'] * 3 +
                            tweet.public_metrics['like_count'] * 1 +
                            tweet.public_metrics['reply_count'] * 2 +
                            tweet.public_metrics.get('quote_count', 0) * 2
                        )
                        
                        tweets_data.append({
                            'timestamp': tweet.created_at,
                            'text': cleaned_text,
                            'original_text': tweet.text,
                            'retweet_count': tweet.public_metrics['retweet_count'],
                            'like_count': tweet.public_metrics['like_count'],
                            'reply_count': tweet.public_metrics['reply_count'],
                            'quote_count': tweet.public_metrics.get('quote_count', 0),
                            'engagement_score': engagement,
                            'tweet_id': tweet.id
                        })
            
            print(f"✅ Scraped {len(tweets_data)} tweets for date range")
            
        except Exception as e:
            print(f"❌ Error scraping tweets for date range: {str(e)}")
        
        return pd.DataFrame(tweets_data)
    
    def get_live_sentiment_data(self, hours_back: int = 24) -> pd.DataFrame:
        """
        Get recent tweets for live sentiment analysis.
        
        Args:
            hours_back: Hours to look back for tweets
        """
        print(f"📡 Getting live sentiment data (last {hours_back} hours)")
        
        # Use multiple Bitcoin-related queries for better coverage
        queries = [
            "bitcoin price OR btc price",
            "bitcoin bull OR btc bull OR bitcoin bear OR btc bear", 
            "#bitcoin OR #btc",
            "cryptocurrency market OR crypto market"
        ]
        
        all_tweets = []
        
        for query in queries:
            print(f"🔍 Querying: {query}")
            tweets = self.scrape_recent_tweets(
                query=query, 
                max_results=250,  # Smaller batches per query
                days_back=max(1, hours_back // 24)  # Convert hours to days
            )
            
            if not tweets.empty:
                tweets['query_used'] = query
                all_tweets.append(tweets)
            
            # Rate limiting between queries
            time.sleep(2)
        
        if all_tweets:
            combined_tweets = pd.concat(all_tweets, ignore_index=True)
            # Remove duplicates based on tweet text
            combined_tweets = combined_tweets.drop_duplicates(subset=['text'], keep='first')
            
            # Filter to actual time range requested
            if hours_back < 24:
                cutoff_time = datetime.now() - timedelta(hours=hours_back)
                combined_tweets = combined_tweets[
                    pd.to_datetime(combined_tweets['timestamp']) >= cutoff_time
                ]
            
            print(f"✅ Collected {len(combined_tweets)} unique tweets for sentiment analysis")
            return combined_tweets
        
        print("⚠️ No tweets collected for live sentiment")
        return pd.DataFrame()

def scrape_recent_tweets(query="bitcoin OR btc", max_results=100):
    """Legacy function for backward compatibility."""
    scraper = TwitterScraper()
    return scraper.scrape_recent_tweets(query, max_results)

def get_live_tweets_for_prediction():
    """Get live tweets specifically for prediction pipeline."""
    scraper = TwitterScraper()
    
    if not scraper.check_api_status():
        return pd.DataFrame()
    
    # Get recent tweets for prediction
    live_tweets = scraper.get_live_sentiment_data(hours_back=6)  # Last 6 hours
    
    if not live_tweets.empty:
        # Save for record keeping
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scraper.save_tweets(live_tweets, f"live_tweets_{timestamp}.csv")
    
    return live_tweets

if __name__ == "__main__":
    print("🚀 BITCOIN SENTIMENT ML - TWITTER SCRAPER")
    print("=" * 50)
    
    scraper = TwitterScraper()
    
    # Check API status first
    if not scraper.check_api_status():
        print("❌ Cannot proceed without Twitter API access")
        print("📝 Please set up your Twitter API credentials in .env file")
        print("🔗 Get credentials at: https://developer.twitter.com/en/portal/dashboard")
        exit(1)
    
    # Choose scraping mode
    print("\n🔧 SCRAPING OPTIONS:")
    print("1. Live sentiment data (last 6 hours) - for predictions")
    print("2. Historical data (last 3 days) - for training")
    print("3. Custom date range")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        # Live sentiment scraping
        print("\n📡 LIVE SENTIMENT SCRAPING")
        live_tweets = scraper.get_live_sentiment_data(hours_back=6)
        
        if not live_tweets.empty:
            scraper.save_tweets(live_tweets, "live_sentiment_tweets.csv")
            print(f"\n📊 LIVE SCRAPING SUMMARY:")
            print(f"Total tweets: {len(live_tweets)}")
            print(f"Time range: {live_tweets['timestamp'].min()} to {live_tweets['timestamp'].max()}")
            print(f"Average engagement: {live_tweets['engagement_score'].mean():.1f}")
        else:
            print("⚠️ No live tweets collected")
    
    elif choice == "2":
        # Historical training data
        print("\n📚 HISTORICAL DATA SCRAPING")
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
            time.sleep(2)  # Rate limiting
        
        if all_tweets:
            combined_tweets = pd.concat(all_tweets, ignore_index=True)
            combined_tweets = combined_tweets.drop_duplicates(subset=['text'])
            scraper.save_tweets(combined_tweets, "historical_training_tweets.csv")
            
            print(f"\n📊 HISTORICAL SCRAPING SUMMARY:")
            print(f"Total unique tweets: {len(combined_tweets)}")
            print(f"Date range: {combined_tweets['timestamp'].min()} to {combined_tweets['timestamp'].max()}")
            print(f"Average engagement: {combined_tweets['engagement_score'].mean():.1f}")
        else:
            print("⚠️ No historical tweets collected")
    
    elif choice == "3":
        # Custom date range
        print("\n📅 CUSTOM DATE RANGE SCRAPING")
        start_date = input("Start date (YYYY-MM-DD): ").strip()
        end_date = input("End date (YYYY-MM-DD): ").strip()
        
        try:
            tweets = scraper.scrape_for_date_range(start_date, end_date)
            if not tweets.empty:
                filename = f"tweets_{start_date}_to_{end_date}.csv"
                scraper.save_tweets(tweets, filename)
                
                print(f"\n📊 CUSTOM RANGE SCRAPING SUMMARY:")
                print(f"Total tweets: {len(tweets)}")
                print(f"Average engagement: {tweets['engagement_score'].mean():.1f}")
            else:
                print("⚠️ No tweets found for the specified date range")
        except ValueError:
            print("❌ Invalid date format. Please use YYYY-MM-DD")
    
    else:
        print("❌ Invalid choice")
    
    print("\n✅ Scraping complete!")
