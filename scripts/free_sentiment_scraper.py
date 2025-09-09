#!/usr/bin/env python3
"""
FREE WEB SCRAPING CRYPTO SENTIMENT
No API keys required - scrape crypto news and social media directly
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import time
import warnings
warnings.filterwarnings('ignore')

class FreeCryptoSentimentScraper:
    """Free crypto sentiment scraper using web scraping."""
    
    def __init__(self):
        print("🌐 Initializing FREE web scraping sentiment collector")
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def scrape_crypto_news_headlines(self):
        """Scrape crypto news headlines from free sources."""
        print("📰 Scraping crypto news headlines...")
        
        news_sources = [
            {
                'name': 'CoinDesk',
                'url': 'https://www.coindesk.com/tag/bitcoin/',
                'headline_selector': 'h3.headline'
            },
            {
                'name': 'CoinTelegraph',
                'url': 'https://cointelegraph.com/tags/bitcoin',
                'headline_selector': '.post-card-inline__title'
            }
        ]
        
        all_headlines = []
        
        for source in news_sources:
            try:
                print(f"🔍 Scraping {source['name']}...")
                
                response = self.session.get(source['url'], timeout=10)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    headlines = soup.select(source['headline_selector'])
                    
                    for headline in headlines[:10]:  # First 10 headlines
                        text = headline.get_text().strip()
                        if text and len(text) > 10:
                            all_headlines.append({
                                'timestamp': datetime.now(),
                                'source': source['name'],
                                'headline': text,
                                'type': 'news'
                            })
                    
                    print(f"✅ Found {len([h for h in all_headlines if h['source'] == source['name']])} headlines")
                
                time.sleep(1)  # Be respectful to servers
                
            except Exception as e:
                print(f"⚠️ Error scraping {source['name']}: {e}")
        
        return all_headlines
    
    def scrape_reddit_public(self):
        """Scrape Reddit without API - public posts."""
        print("🔍 Scraping Reddit (public, no API)...")
        
        reddit_posts = []
        subreddits = ['bitcoin', 'cryptocurrency']
        
        for subreddit in subreddits:
            try:
                # Reddit JSON endpoint (public)
                url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=25"
                
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for post in data['data']['children']:
                        post_data = post['data']
                        
                        title = post_data.get('title', '')
                        selftext = post_data.get('selftext', '')
                        
                        if self.is_bitcoin_related(title + " " + selftext):
                            reddit_posts.append({
                                'timestamp': datetime.fromtimestamp(post_data['created_utc']),
                                'source': f"r/{subreddit}",
                                'headline': title,
                                'text': selftext,
                                'score': post_data.get('score', 0),
                                'comments': post_data.get('num_comments', 0),
                                'type': 'reddit'
                            })
                    
                    print(f"✅ Found {len([p for p in reddit_posts if p['source'] == f'r/{subreddit}'])} Bitcoin posts")
                
                time.sleep(2)  # Be respectful
                
            except Exception as e:
                print(f"⚠️ Error scraping r/{subreddit}: {e}")
        
        return reddit_posts
    
    def scrape_crypto_fear_greed_index(self):
        """Scrape Fear & Greed Index (free market sentiment)."""
        print("😨 Getting Fear & Greed Index...")
        
        try:
            # Alternative.me Fear & Greed API (free)
            url = "https://api.alternative.me/fng/"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    latest = data['data'][0]
                    
                    fear_greed_data = {
                        'timestamp': datetime.now(),
                        'source': 'Fear_Greed_Index',
                        'headline': f"Market Fear & Greed: {latest['value_classification']} ({latest['value']})",
                        'fear_greed_value': int(latest['value']),
                        'fear_greed_classification': latest['value_classification'],
                        'type': 'market_indicator'
                    }
                    
                    print(f"✅ Fear & Greed Index: {latest['value_classification']} ({latest['value']})")
                    return [fear_greed_data]
            
        except Exception as e:
            print(f"⚠️ Error getting Fear & Greed Index: {e}")
        
        return []
    
    def is_bitcoin_related(self, text):
        """Check if text is Bitcoin/crypto related."""
        bitcoin_keywords = [
            'bitcoin', 'btc', 'cryptocurrency', 'crypto', 'blockchain',
            'satoshi', 'hodl', 'mining', 'wallet', 'exchange', 'bull', 'bear'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in bitcoin_keywords)
    
    def analyze_sentiment_simple(self, text):
        """Simple rule-based sentiment analysis."""
        # Positive words
        positive_words = [
            'bull', 'bullish', 'pump', 'moon', 'surge', 'rally', 'rise', 'gain',
            'up', 'high', 'green', 'profit', 'buy', 'hodl', 'strong', 'breakthrough',
            'adoption', 'institutional', 'bullish', 'optimistic', 'positive'
        ]
        
        # Negative words  
        negative_words = [
            'bear', 'bearish', 'dump', 'crash', 'fall', 'drop', 'loss', 'down',
            'red', 'sell', 'panic', 'weak', 'correction', 'dip', 'decline',
            'pessimistic', 'negative', 'fear', 'uncertainty', 'regulation'
        ]
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 1  # Positive
        elif negative_count > positive_count:
            return -1  # Negative
        else:
            return 0  # Neutral
    
    def process_all_data(self):
        """Collect and process all free sentiment data."""
        print("🚀 FREE CRYPTO SENTIMENT COLLECTION")
        print("=" * 50)
        
        all_data = []
        
        # 1. News headlines
        news_data = self.scrape_crypto_news_headlines()
        all_data.extend(news_data)
        
        # 2. Reddit posts
        reddit_data = self.scrape_reddit_public()
        all_data.extend(reddit_data)
        
        # 3. Fear & Greed Index
        fear_greed_data = self.scrape_crypto_fear_greed_index()
        all_data.extend(fear_greed_data)
        
        if not all_data:
            print("❌ No data collected")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Add sentiment analysis
        sentiments = []
        for _, row in df.iterrows():
            text = f"{row.get('headline', '')} {row.get('text', '')}"
            sentiment = self.analyze_sentiment_simple(text)
            sentiments.append(sentiment)
        
        df['sentiment'] = sentiments
        
        # Summary
        print(f"\n📊 COLLECTION SUMMARY:")
        print(f"Total items: {len(df)}")
        print(f"Sources: {df['source'].nunique()}")
        print(f"Positive sentiment: {len(df[df['sentiment'] == 1])}")
        print(f"Negative sentiment: {len(df[df['sentiment'] == -1])}")
        print(f"Neutral sentiment: {len(df[df['sentiment'] == 0])}")
        
        # Calculate overall sentiment
        avg_sentiment = df['sentiment'].mean()
        print(f"Overall sentiment: {avg_sentiment:.3f}")
        
        return df
    
    def save_data(self, df, filename="free_crypto_sentiment.csv"):
        """Save collected data."""
        if df.empty:
            return False
        
        import os
        os.makedirs("data", exist_ok=True)
        filepath = f"data/{filename}"
        
        df['scraped_at'] = datetime.now()
        df.to_csv(filepath, index=False)
        
        print(f"✅ Saved {len(df)} sentiment items → {filepath}")
        return True

def main():
    """Main function to run free sentiment collection."""
    scraper = FreeCryptoSentimentScraper()
    
    # Collect all free sentiment data
    sentiment_df = scraper.process_all_data()
    
    if not sentiment_df.empty:
        # Save data
        scraper.save_data(sentiment_df)
        
        print(f"\n🎉 SUCCESS!")
        print(f"💡 Collected crypto sentiment data without any API limits")
        print(f"🔄 Run this script daily for continuous free sentiment monitoring")
        print(f"📈 Combine with your existing Bitcoin price data for predictions")
        
        return sentiment_df
    
    else:
        print(f"❌ No sentiment data collected")
        return None

if __name__ == "__main__":
    main()
