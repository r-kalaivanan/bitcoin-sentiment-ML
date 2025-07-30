# scripts/sentiment_analysis.py

import pandas as pd
import numpy as np
import tweepy
import os
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import re
import time
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

class SentimentAnalyzer:
    """
    Comprehensive sentiment analysis for Bitcoin-related tweets.
    Scrapes tweets, analyzes sentiment, and creates daily sentiment features.
    """
    
    def __init__(self):
        self.bearer_token = os.getenv("X_BEARER_TOKEN")
        self.client = tweepy.Client(bearer_token=self.bearer_token) if self.bearer_token else None
        self.analyzer = SentimentIntensityAnalyzer()
        
    def clean_tweet_text(self, text):
        """Clean and preprocess tweet text."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters but keep relevant ones for sentiment
        text = re.sub(r'[^\w\s!?.,]', '', text)
        return text.strip()
    
    def scrape_recent_tweets(self, query="bitcoin OR btc OR cryptocurrency", max_results=1000, days_back=7):
        """
        Scrape recent tweets about Bitcoin.
        
        Args:
            query: Search query for tweets
            max_results: Maximum number of tweets to retrieve
            days_back: Number of days to go back for tweet collection
        """
        if not self.client:
            print("❌ Twitter API credentials not found. Please set X_BEARER_TOKEN in .env file")
            return pd.DataFrame()
        
        print(f"🐦 Scraping tweets with query: '{query}'")
        
        # Calculate date range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        tweets_data = []
        
        try:
            # Search for tweets
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
                    if len(cleaned_text) > 10:  # Filter out very short tweets
                        tweets_data.append({
                            'timestamp': tweet.created_at,
                            'text': cleaned_text,
                            'retweet_count': tweet.public_metrics['retweet_count'],
                            'like_count': tweet.public_metrics['like_count'],
                            'reply_count': tweet.public_metrics['reply_count']
                        })
            
            print(f"✅ Scraped {len(tweets_data)} clean tweets")
            
        except Exception as e:
            print(f"❌ Error scraping tweets: {str(e)}")
            return pd.DataFrame()
        
        return pd.DataFrame(tweets_data)
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of a single text using VADER.
        
        Returns:
            dict: Sentiment scores (positive, negative, neutral, compound)
        """
        scores = self.analyzer.polarity_scores(text)
        return {
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': scores['compound']
        }
    
    def classify_sentiment(self, compound_score):
        """
        Classify sentiment based on compound score.
        
        Args:
            compound_score: VADER compound sentiment score
            
        Returns:
            str: 'positive', 'negative', or 'neutral'
        """
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def process_tweets_sentiment(self, tweets_df):
        """
        Process sentiment for all tweets in the dataframe.
        
        Args:
            tweets_df: DataFrame with tweet data
            
        Returns:
            DataFrame: Enhanced with sentiment scores
        """
        if tweets_df.empty:
            print("❌ No tweets to process")
            return tweets_df
        
        print(f"🔍 Analyzing sentiment for {len(tweets_df)} tweets...")
        
        # Analyze sentiment for each tweet
        sentiment_results = []
        for _, tweet in tweets_df.iterrows():
            sentiment = self.analyze_sentiment(tweet['text'])
            sentiment_results.append(sentiment)
        
        # Add sentiment columns
        sentiment_df = pd.DataFrame(sentiment_results)
        tweets_df = pd.concat([tweets_df, sentiment_df], axis=1)
        
        # Add sentiment classification
        tweets_df['sentiment_class'] = tweets_df['compound'].apply(self.classify_sentiment)
        
        # Add engagement weight (tweets with more engagement get higher weight)
        tweets_df['engagement_score'] = (
            tweets_df['like_count'] + 
            tweets_df['retweet_count'] * 2 + 
            tweets_df['reply_count']
        )
        
        print("✅ Sentiment analysis complete")
        return tweets_df
    
    def create_daily_sentiment_features(self, tweets_df):
        """
        Create daily aggregated sentiment features.
        
        Args:
            tweets_df: DataFrame with tweet sentiment data
            
        Returns:
            DataFrame: Daily sentiment features
        """
        if tweets_df.empty:
            print("❌ No tweets to aggregate")
            return pd.DataFrame()
        
        print("📊 Creating daily sentiment aggregations...")
        
        # Convert timestamp to date
        tweets_df['date'] = pd.to_datetime(tweets_df['timestamp']).dt.date
        
        # Basic sentiment aggregations
        daily_sentiment = tweets_df.groupby('date').agg({
            'positive': ['mean', 'std', 'count'],
            'negative': ['mean', 'std'],
            'neutral': ['mean', 'std'],
            'compound': ['mean', 'std', 'min', 'max'],
            'engagement_score': ['sum', 'mean']
        }).round(4)
        
        # Flatten column names
        daily_sentiment.columns = ['_'.join(col).strip() for col in daily_sentiment.columns]
        daily_sentiment = daily_sentiment.reset_index()
        
        # Sentiment class distribution
        sentiment_dist = tweets_df.groupby(['date', 'sentiment_class']).size().unstack(fill_value=0)
        sentiment_dist = sentiment_dist.div(sentiment_dist.sum(axis=1), axis=0).round(4)
        sentiment_dist.columns = [f'sentiment_{col}_ratio' for col in sentiment_dist.columns]
        sentiment_dist = sentiment_dist.reset_index()
        
        # Merge aggregations
        daily_features = pd.merge(daily_sentiment, sentiment_dist, on='date', how='left')
        
        # Additional features
        daily_features['sentiment_volatility'] = daily_features['compound_std']
        daily_features['sentiment_range'] = daily_features['compound_max'] - daily_features['compound_min']
        daily_features['total_tweets'] = daily_features['positive_count']
        daily_features['weighted_sentiment'] = (
            daily_features['compound_mean'] * np.log1p(daily_features['total_tweets'])
        )
        
        # Clean column names
        daily_features.columns = [col.replace('__', '_') for col in daily_features.columns]
        
        print(f"✅ Created daily sentiment features for {len(daily_features)} days")
        return daily_features
    
    def create_sentiment_features_historical(self, start_date='2020-01-01', end_date=None):
        """
        Create historical sentiment features (simulated for demonstration).
        In practice, you would have historical tweet data or use alternative sources.
        """
        print("📈 Creating historical sentiment features (simulated)...")
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Simulate sentiment data (replace with real historical data when available)
        np.random.seed(42)  # For reproducible results
        
        sentiment_data = []
        for date in date_range:
            # Simulate daily sentiment with some realistic patterns
            base_sentiment = 0.1 + 0.3 * np.sin(len(sentiment_data) * 0.01)  # Long-term trend
            daily_noise = np.random.normal(0, 0.2)  # Daily variation
            
            compound_mean = np.clip(base_sentiment + daily_noise, -1, 1)
            
            sentiment_data.append({
                'date': date.date(),
                'compound_mean': compound_mean,
                'compound_std': np.random.uniform(0.1, 0.4),
                'positive_mean': max(0, compound_mean + np.random.uniform(0, 0.3)),
                'negative_mean': max(0, -compound_mean + np.random.uniform(0, 0.3)),
                'neutral_mean': np.random.uniform(0.3, 0.7),
                'total_tweets': np.random.randint(50, 500),
                'engagement_score_sum': np.random.randint(1000, 10000),
                'sentiment_positive_ratio': max(0, compound_mean + 0.5) / 1.5,
                'sentiment_negative_ratio': max(0, -compound_mean + 0.5) / 1.5,
                'sentiment_neutral_ratio': np.random.uniform(0.2, 0.5),
                'sentiment_volatility': np.random.uniform(0.1, 0.4),
                'weighted_sentiment': compound_mean * np.log1p(np.random.randint(50, 500))
            })
        
        sentiment_df = pd.DataFrame(sentiment_data)
        
        # Add rolling features
        sentiment_df['sentiment_ma_7'] = sentiment_df['compound_mean'].rolling(7).mean()
        sentiment_df['sentiment_ma_14'] = sentiment_df['compound_mean'].rolling(14).mean()
        sentiment_df['sentiment_momentum'] = sentiment_df['compound_mean'] - sentiment_df['sentiment_ma_7']
        
        print(f"✅ Created {len(sentiment_df)} days of historical sentiment features")
        return sentiment_df
    
    def run_sentiment_pipeline(self, historical=True, scrape_recent=False):
        """
        Run the complete sentiment analysis pipeline.
        
        Args:
            historical: Whether to create historical sentiment features
            scrape_recent: Whether to scrape recent tweets
        """
        print("🚀 Starting sentiment analysis pipeline...")
        
        all_sentiment_data = []
        
        # Create historical sentiment features
        if historical:
            historical_sentiment = self.create_sentiment_features_historical()
            all_sentiment_data.append(historical_sentiment)
        
        # Scrape and analyze recent tweets
        if scrape_recent and self.client:
            recent_tweets = self.scrape_recent_tweets()
            if not recent_tweets.empty:
                tweets_with_sentiment = self.process_tweets_sentiment(recent_tweets)
                recent_daily_sentiment = self.create_daily_sentiment_features(tweets_with_sentiment)
                if not recent_daily_sentiment.empty:
                    all_sentiment_data.append(recent_daily_sentiment)
                
                # Save raw tweet data
                tweets_with_sentiment.to_csv('data/raw_tweets_with_sentiment.csv', index=False)
                print("✅ Raw tweet data saved to: data/raw_tweets_with_sentiment.csv")
        
        # Combine all sentiment data
        if all_sentiment_data:
            combined_sentiment = pd.concat(all_sentiment_data, ignore_index=True, sort=False)
            combined_sentiment = combined_sentiment.drop_duplicates(subset=['date']).sort_values('date')
            
            # Save combined sentiment features
            combined_sentiment.to_csv('data/sentiment_features.csv', index=False)
            print(f"✅ Sentiment features saved to: data/sentiment_features.csv")
            
            print(f"🎉 Sentiment pipeline complete! Created features for {len(combined_sentiment)} days")
            return combined_sentiment
        else:
            print("❌ No sentiment data created")
            return pd.DataFrame()

if __name__ == "__main__":
    # Create sentiment analyzer and run pipeline
    analyzer = SentimentAnalyzer()
    
    # Run with historical data (for backtesting) and try to scrape recent tweets
    sentiment_features = analyzer.run_sentiment_pipeline(
        historical=True, 
        scrape_recent=True
    )
    
    if not sentiment_features.empty:
        print(f"\n📊 SENTIMENT ANALYSIS SUMMARY:")
        print(f"Date range: {sentiment_features['date'].min()} to {sentiment_features['date'].max()}")
        print(f"Average sentiment: {sentiment_features['compound_mean'].mean():.3f}")
        print(f"Sentiment volatility: {sentiment_features['compound_std'].mean():.3f}")
    else:
        print("❌ No sentiment features created")
