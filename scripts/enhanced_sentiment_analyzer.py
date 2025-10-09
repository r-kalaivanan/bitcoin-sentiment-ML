import tweepy
import praw
import requests
import pandas as pd
from textblob import TextBlob
from datetime import datetime, timedelta
import re
import sqlite3
import os
from dotenv import load_dotenv
import logging
import random
import time

load_dotenv()

class CryptoSentimentAnalyzer:
    def __init__(self):
        self.setup_apis()
        self.db_path = "data/sentiment_data.db"
        self.setup_database()
        
    def setup_apis(self):
        """Setup Twitter and Reddit API connections"""
        print("🔧 Setting up API connections...")
        
        # Twitter API setup
        try:
            # Try both possible environment variable names
            bearer_token = os.getenv('X_BEARER_TOKEN') or os.getenv('TWITTER_BEARER_TOKEN')
            if bearer_token:
                self.twitter_api = tweepy.Client(bearer_token=bearer_token)
                print("✅ Twitter API configured")
            else:
                self.twitter_api = None
                print("⚠️ Twitter API not configured - using mock data")
        except Exception as e:
            print(f"⚠️ Twitter API error: {e}")
            self.twitter_api = None
            
        # Reddit API setup
        try:
            client_id = os.getenv('REDDIT_CLIENT_ID')
            client_secret = os.getenv('REDDIT_CLIENT_SECRET')
            if client_id and client_secret:
                self.reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent='bitcoin_sentiment_analyzer'
                )
                print("✅ Reddit API configured")
            else:
                self.reddit = None
                print("⚠️ Reddit API not configured - using mock data")
        except Exception as e:
            print(f"⚠️ Reddit API error: {e}")
            self.reddit = None
            
    def setup_database(self):
        """Create database for sentiment data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_scores (
                timestamp DATETIME,
                source TEXT,
                content TEXT,
                sentiment_score REAL,
                sentiment_label TEXT,
                confidence REAL,
                keywords TEXT
            )
        ''')
        conn.commit()
        conn.close()
        print("✅ Sentiment database initialized")
        
    def clean_text(self, text):
        """Clean text for sentiment analysis"""
        if not text:
            return ""
        
        # Remove URLs, mentions, hashtags, special characters
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)  # Keep hashtag content
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
        
    def extract_keywords(self, text):
        """Extract Bitcoin-related keywords"""
        bitcoin_keywords = [
            'bitcoin', 'btc', 'crypto', 'cryptocurrency', 'blockchain',
            'bull', 'bear', 'moon', 'crash', 'pump', 'dump', 'hodl',
            'buy', 'sell', 'price', 'investment', 'trading'
        ]
        
        text_lower = text.lower()
        found_keywords = [kw for kw in bitcoin_keywords if kw in text_lower]
        return ','.join(found_keywords)
        
    def analyze_text_sentiment(self, text):
        """Analyze sentiment of a single text using TextBlob"""
        cleaned_text = self.clean_text(text)
        if not cleaned_text or len(cleaned_text) < 5:
            return 0, "neutral", 0.5, ""
            
        try:
            blob = TextBlob(cleaned_text)
            polarity = blob.sentiment.polarity  # -1 to 1
            
            # Enhanced labeling
            if polarity > 0.2:
                label = "positive"
            elif polarity < -0.2:
                label = "negative"
            else:
                label = "neutral"
                
            confidence = min(abs(polarity) + 0.3, 1.0)  # Boost confidence
            keywords = self.extract_keywords(text)
            
            return polarity, label, confidence, keywords
            
        except Exception as e:
            logging.error(f"Sentiment analysis error: {e}")
            return 0, "neutral", 0.5, ""
        
    def get_twitter_sentiment(self, query="(bitcoin OR BTC OR crypto) -is:retweet", count=100):
        """Get sentiment from Twitter"""
        if not self.twitter_api:
            return self.get_mock_sentiment_data("twitter", count)
            
        try:
            print(f"🐦 Fetching {count} tweets...")
            tweets = tweepy.Paginator(
                self.twitter_api.search_recent_tweets,
                query=query,
                max_results=min(count, 100),
                tweet_fields=['created_at', 'public_metrics']
            ).flatten(limit=count)
            
            sentiments = []
            for tweet in tweets:
                score, label, confidence, keywords = self.analyze_text_sentiment(tweet.text)
                sentiments.append({
                    'timestamp': datetime.now(),
                    'source': 'twitter',
                    'content': tweet.text[:200],  # First 200 chars
                    'sentiment_score': score,
                    'sentiment_label': label,
                    'confidence': confidence,
                    'keywords': keywords
                })
                
            print(f"✅ Analyzed {len(sentiments)} tweets")
            return sentiments
            
        except Exception as e:
            print(f"❌ Twitter API error: {e}")
            return self.get_mock_sentiment_data("twitter", count)
            
    def get_reddit_sentiment(self, subreddit="bitcoin", limit=50):
        """Get sentiment from Reddit"""
        if not self.reddit:
            return self.get_mock_sentiment_data("reddit", limit)
            
        try:
            print(f"📱 Fetching from r/{subreddit}...")
            subreddit_obj = self.reddit.subreddit(subreddit)
            
            sentiments = []
            # Get hot posts
            for post in subreddit_obj.hot(limit=limit//2):
                text = f"{post.title} {post.selftext}"
                score, label, confidence, keywords = self.analyze_text_sentiment(text)
                
                sentiments.append({
                    'timestamp': datetime.now(),
                    'source': 'reddit_hot',
                    'content': post.title[:200],
                    'sentiment_score': score,
                    'sentiment_label': label,
                    'confidence': confidence,
                    'keywords': keywords
                })
                
            # Get new posts
            for post in subreddit_obj.new(limit=limit//2):
                text = f"{post.title} {post.selftext}"
                score, label, confidence, keywords = self.analyze_text_sentiment(text)
                
                sentiments.append({
                    'timestamp': datetime.now(),
                    'source': 'reddit_new',
                    'content': post.title[:200],
                    'sentiment_score': score,
                    'sentiment_label': label,
                    'confidence': confidence,
                    'keywords': keywords
                })
                
            print(f"✅ Analyzed {len(sentiments)} Reddit posts")
            return sentiments
            
        except Exception as e:
            print(f"❌ Reddit API error: {e}")
            return self.get_mock_sentiment_data("reddit", limit)
            
    def get_news_sentiment(self, count=20):
        """Get sentiment from crypto news using free sources"""
        try:
            print(f"📰 Fetching crypto news...")
            
            # CoinDesk RSS feed
            sentiments = []
            try:
                import feedparser
                feed = feedparser.parse("https://feeds.coindesk.com/bitcoin")
                
                for entry in feed.entries[:count]:
                    text = f"{entry.title} {entry.summary}"
                    score, label, confidence, keywords = self.analyze_text_sentiment(text)
                    
                    sentiments.append({
                        'timestamp': datetime.now(),
                        'source': 'news_coindesk',
                        'content': entry.title[:200],
                        'sentiment_score': score,
                        'sentiment_label': label,
                        'confidence': confidence,
                        'keywords': keywords
                    })
                    
            except ImportError:
                print("⚠️ feedparser not installed, using mock news data")
                sentiments = self.get_mock_sentiment_data("news", count)
                
            print(f"✅ Analyzed {len(sentiments)} news articles")
            return sentiments
            
        except Exception as e:
            print(f"❌ News API error: {e}")
            return self.get_mock_sentiment_data("news", count)
            
    def get_mock_sentiment_data(self, source, count=20):
        """Generate realistic mock sentiment data for testing"""
        sentiments = []
        
        # Mock content based on source
        mock_contents = {
            'twitter': [
                "Bitcoin is looking bullish today! 🚀",
                "BTC price action is concerning...",
                "Just bought more Bitcoin, hodling strong",
                "Crypto market is volatile as always",
                "Bitcoin to the moon! 🌙",
                "Bearish signals on BTC charts",
                "DCA into Bitcoin every week",
                "Bitcoin adoption is growing",
                "Worried about Bitcoin price drop",
                "Bitcoin is the future of money"
            ],
            'reddit': [
                "Bitcoin analysis: Technical indicators suggest...",
                "Should I buy Bitcoin at current price?",
                "Bitcoin mining profitability discussion",
                "Long-term Bitcoin investment strategy",
                "Bitcoin vs traditional assets comparison",
                "Bitcoin regulation news impact",
                "Bitcoin network hash rate analysis",
                "Bitcoin institutional adoption trends",
                "Bitcoin market manipulation concerns",
                "Bitcoin price prediction discussion"
            ],
            'news': [
                "Bitcoin reaches new weekly high amid institutional interest",
                "Regulatory uncertainty impacts Bitcoin market",
                "Major corporation announces Bitcoin investment",
                "Bitcoin network upgrade scheduled for next month",
                "Central bank digital currencies vs Bitcoin debate",
                "Bitcoin mining energy consumption study released",
                "Bitcoin ETF approval speculation continues",
                "Bitcoin payment adoption by major retailer",
                "Bitcoin price volatility analysis published",
                "Bitcoin blockchain security assessment report"
            ]
        }
        
        contents = mock_contents.get(source, mock_contents['twitter'])
        
        for i in range(count):
            # Generate sentiment with realistic distribution
            # 40% positive, 30% neutral, 30% negative
            rand = random.random()
            if rand < 0.4:
                score = random.uniform(0.1, 0.8)
                label = "positive"
            elif rand < 0.7:
                score = random.uniform(-0.1, 0.1)
                label = "neutral"
            else:
                score = random.uniform(-0.8, -0.1)
                label = "negative"
                
            content = random.choice(contents)
            keywords = self.extract_keywords(content)
            
            sentiments.append({
                'timestamp': datetime.now(),
                'source': f'{source}_mock',
                'content': content,
                'sentiment_score': score,
                'sentiment_label': label,
                'confidence': random.uniform(0.6, 0.9),
                'keywords': keywords
            })
            
        return sentiments
        
    def store_sentiment_data(self, sentiments):
        """Store sentiment data in database"""
        if not sentiments:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for sentiment in sentiments:
            cursor.execute('''
                INSERT INTO sentiment_scores 
                (timestamp, source, content, sentiment_score, sentiment_label, confidence, keywords)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                sentiment['timestamp'],
                sentiment['source'],
                sentiment['content'],
                sentiment['sentiment_score'],
                sentiment['sentiment_label'],
                sentiment['confidence'],
                sentiment['keywords']
            ))
            
        conn.commit()
        conn.close()
        print(f"💾 Stored {len(sentiments)} sentiment records")
        
    def get_aggregated_sentiment(self, hours=24):
        """Get aggregated sentiment score for the last N hours"""
        conn = sqlite3.connect(self.db_path)
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        df = pd.read_sql_query('''
            SELECT * FROM sentiment_scores 
            WHERE timestamp > ?
            ORDER BY timestamp DESC
        ''', conn, params=(cutoff_time,))
        conn.close()
        
        if df.empty:
            return {
                'overall_sentiment': 0,
                'sentiment_label': 'neutral',
                'confidence': 0.5,
                'sample_size': 0,
                'source_breakdown': {},
                'keyword_frequency': {},
                'time_trend': 'stable'
            }
        
        # Convert timestamp column to datetime if it's string
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # Calculate weighted average sentiment
        weights = df['confidence']
        overall_sentiment = (df['sentiment_score'] * weights).sum() / weights.sum()
        
        # Determine overall label
        if overall_sentiment > 0.15:
            sentiment_label = 'positive'
        elif overall_sentiment < -0.15:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
            
        # Source breakdown
        source_breakdown = df.groupby('source')['sentiment_score'].mean().to_dict()
        
        # Keyword frequency
        all_keywords = []
        for keywords in df['keywords'].dropna():
            if keywords:  # Check if not empty
                all_keywords.extend(keywords.split(','))
        
        keyword_freq = pd.Series([kw for kw in all_keywords if kw]).value_counts().head(10).to_dict()
        
        # Time trend (last 6 hours vs previous 18 hours)
        recent_cutoff = datetime.now() - timedelta(hours=6)
        recent_sentiment = df[df['timestamp'] > recent_cutoff]['sentiment_score'].mean()
        older_sentiment = df[df['timestamp'] <= recent_cutoff]['sentiment_score'].mean()
        
        if pd.isna(recent_sentiment) or pd.isna(older_sentiment):
            time_trend = 'stable'
        elif recent_sentiment > older_sentiment + 0.1:
            time_trend = 'improving'
        elif recent_sentiment < older_sentiment - 0.1:
            time_trend = 'declining'
        else:
            time_trend = 'stable'
        
        return {
            'overall_sentiment': float(overall_sentiment),
            'sentiment_label': sentiment_label,
            'confidence': float(df['confidence'].mean()),
            'sample_size': len(df),
            'source_breakdown': source_breakdown,
            'keyword_frequency': keyword_freq,
            'time_trend': time_trend,
            'last_updated': datetime.now().isoformat()
        }
        
    def collect_all_sentiment(self):
        """Collect sentiment from all available sources"""
        print("🔍 Starting comprehensive sentiment collection...")
        
        all_sentiments = []
        
        # Collect from Twitter
        try:
            twitter_sentiment = self.get_twitter_sentiment(count=50)
            all_sentiments.extend(twitter_sentiment)
        except Exception as e:
            print(f"⚠️ Twitter collection failed: {e}")
        
        # Collect from Reddit
        try:
            reddit_sentiment = self.get_reddit_sentiment(limit=30)
            all_sentiments.extend(reddit_sentiment)
        except Exception as e:
            print(f"⚠️ Reddit collection failed: {e}")
        
        # Collect from News
        try:
            news_sentiment = self.get_news_sentiment(count=20)
            all_sentiments.extend(news_sentiment)
        except Exception as e:
            print(f"⚠️ News collection failed: {e}")
        
        # Store all collected sentiment data
        if all_sentiments:
            self.store_sentiment_data(all_sentiments)
            
        print(f"✅ Total sentiment data points collected: {len(all_sentiments)}")
        return all_sentiments
        
    def get_sentiment_summary(self):
        """Get a comprehensive sentiment summary"""
        sentiment_data = self.get_aggregated_sentiment()
        
        print("\n📊 SENTIMENT ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Overall Sentiment: {sentiment_data['sentiment_label'].upper()} ({sentiment_data['overall_sentiment']:.3f})")
        print(f"Confidence Level: {sentiment_data['confidence']:.1%}")
        print(f"Sample Size: {sentiment_data['sample_size']} data points")
        print(f"Trend: {sentiment_data['time_trend'].upper()}")
        
        if sentiment_data['source_breakdown']:
            print("\nSource Breakdown:")
            for source, score in sentiment_data['source_breakdown'].items():
                print(f"  {source}: {score:+.3f}")
                
        if sentiment_data['keyword_frequency']:
            print("\nTop Keywords:")
            for keyword, freq in list(sentiment_data['keyword_frequency'].items())[:5]:
                if keyword:  # Skip empty keywords
                    print(f"  {keyword}: {freq} mentions")
        
        return sentiment_data

if __name__ == "__main__":
    analyzer = CryptoSentimentAnalyzer()
    
    # Collect sentiment data
    analyzer.collect_all_sentiment()
    
    # Display summary
    analyzer.get_sentiment_summary()