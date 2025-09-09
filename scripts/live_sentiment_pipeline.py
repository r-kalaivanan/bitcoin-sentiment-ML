# scripts/live_sentiment_pipeline.py - Real-time Sentiment Analysis Pipeline

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import sys
from typing import Optional

# Add the scripts directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scrape import TwitterScraper, get_live_tweets_for_prediction
from sentiment_analysis import SentimentAnalyzer

class LiveSentimentPipeline:
    """Real-time sentiment analysis pipeline for Bitcoin predictions."""
    
    def __init__(self):
        self.scraper = TwitterScraper()
        self.analyzer = SentimentAnalyzer()
        self.last_update = None
        
    def check_system_status(self) -> dict:
        """Check if all components are ready."""
        status = {
            'twitter_api': False,
            'sentiment_analyzer': True,
            'overall': False
        }
        
        # Check Twitter API
        if self.scraper.check_api_status():
            status['twitter_api'] = True
            
        # Overall status
        status['overall'] = status['twitter_api'] and status['sentiment_analyzer']
        
        return status
    
    def get_live_sentiment_features(self, hours_back: int = 6) -> pd.DataFrame:
        """
        Get live sentiment features for prediction.
        
        Args:
            hours_back: Hours to look back for tweets
            
        Returns:
            DataFrame with sentiment features or empty DataFrame
        """
        print(f"🔄 Getting live sentiment features (last {hours_back} hours)")
        
        try:
            # Get live tweets
            tweets = self.scraper.get_live_sentiment_data(hours_back=hours_back)
            
            if tweets.empty:
                print("⚠️ No live tweets available")
                return pd.DataFrame()
            
            print(f"📊 Processing {len(tweets)} tweets for sentiment analysis")
            
            # Process sentiment
            tweets_with_sentiment = self.analyzer.process_tweets_sentiment(tweets)
            
            if tweets_with_sentiment.empty:
                print("⚠️ No sentiment data generated")
                return pd.DataFrame()
            
            # Create daily features
            daily_sentiment = self.analyzer.create_daily_sentiment_features(tweets_with_sentiment)
            
            # Add metadata
            daily_sentiment['source'] = 'live_twitter'
            daily_sentiment['processed_at'] = datetime.now()
            daily_sentiment['tweet_count'] = len(tweets)
            
            print(f"✅ Generated sentiment features for {len(daily_sentiment)} days")
            
            # Save for record keeping
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_sentiment_data(daily_sentiment, f"live_sentiment_{timestamp}.csv")
            
            return daily_sentiment
            
        except Exception as e:
            print(f"❌ Error getting live sentiment features: {str(e)}")
            return pd.DataFrame()
    
    def get_sentiment_for_prediction(self, target_date: str = None) -> pd.DataFrame:
        """
        Get sentiment features specifically for prediction pipeline.
        
        Args:
            target_date: Target date for prediction (YYYY-MM-DD)
            
        Returns:
            DataFrame with sentiment features for the target date
        """
        if target_date is None:
            target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"🎯 Getting sentiment for prediction date: {target_date}")
        
        # Try to get live sentiment first
        live_sentiment = self.get_live_sentiment_features(hours_back=24)
        
        if not live_sentiment.empty:
            # Filter for target date or use latest
            target_dt = pd.to_datetime(target_date).date()
            
            # Look for exact date match
            target_sentiment = live_sentiment[live_sentiment['date'].dt.date == target_dt]
            
            if not target_sentiment.empty:
                print(f"🎯 Found exact sentiment match for {target_date}")
                return target_sentiment
            else:
                # Use latest available sentiment
                latest_sentiment = live_sentiment.iloc[-1:].copy()
                latest_sentiment['date'] = pd.to_datetime(target_date)
                print(f"📅 Using latest sentiment (adjusted to {target_date})")
                return latest_sentiment
        
        # Fallback: use the sentiment analyzer's fallback method
        print("🔄 Using sentiment analyzer fallback method")
        return self.analyzer.create_sentiment_for_prediction(target_date)
    
    def save_sentiment_data(self, sentiment_df: pd.DataFrame, filename: str):
        """Save sentiment data with metadata."""
        os.makedirs("data", exist_ok=True)
        filepath = f"data/{filename}"
        sentiment_df.to_csv(filepath, index=False)
        print(f"💾 Saved sentiment data → {filepath}")
    
    def run_continuous_monitoring(self, interval_minutes: int = 30, duration_hours: int = 24):
        """
        Run continuous sentiment monitoring.
        
        Args:
            interval_minutes: Minutes between updates
            duration_hours: Total duration to run monitoring
        """
        print(f"🔄 Starting continuous sentiment monitoring")
        print(f"⏱️ Update interval: {interval_minutes} minutes")
        print(f"⏳ Duration: {duration_hours} hours")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        iteration = 0
        
        while datetime.now() < end_time:
            iteration += 1
            print(f"\n📡 MONITORING ITERATION {iteration}")
            print(f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Get current sentiment
            sentiment_data = self.get_live_sentiment_features(hours_back=6)
            
            if not sentiment_data.empty:
                # Calculate sentiment metrics
                latest_sentiment = sentiment_data.iloc[-1]
                print(f"📊 Current sentiment: {latest_sentiment['compound_mean']:.3f}")
                print(f"📈 Sentiment volatility: {latest_sentiment['sentiment_volatility']:.3f}")
                print(f"🐦 Total tweets analyzed: {latest_sentiment['total_tweets']}")
                
                # Store for historical tracking
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.save_sentiment_data(sentiment_data, f"monitoring_{timestamp}.csv")
            else:
                print("⚠️ No sentiment data available for this iteration")
            
            # Wait for next iteration
            print(f"⏳ Waiting {interval_minutes} minutes for next update...")
            time.sleep(interval_minutes * 60)
        
        print("✅ Continuous monitoring completed")
    
    def generate_sentiment_report(self) -> dict:
        """Generate a comprehensive sentiment report."""
        print("📊 Generating sentiment report...")
        
        sentiment_data = self.get_live_sentiment_features(hours_back=24)
        
        if sentiment_data.empty:
            return {"status": "no_data", "message": "No sentiment data available"}
        
        # Calculate metrics
        latest = sentiment_data.iloc[-1]
        
        report = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data_source": "live_twitter",
            "sentiment_metrics": {
                "current_sentiment": float(latest['compound_mean']),
                "sentiment_trend": "positive" if latest['compound_mean'] > 0 else "negative",
                "volatility": float(latest['sentiment_volatility']),
                "total_tweets": int(latest['total_tweets']),
                "engagement_score": float(latest['engagement_score_sum'])
            },
            "summary": {
                "avg_sentiment_24h": float(sentiment_data['compound_mean'].mean()),
                "sentiment_range": [
                    float(sentiment_data['compound_mean'].min()),
                    float(sentiment_data['compound_mean'].max())
                ],
                "tweet_volume_24h": int(sentiment_data['total_tweets'].sum()),
                "dominant_sentiment": "positive" if sentiment_data['compound_mean'].mean() > 0 else "negative"
            }
        }
        
        return report

def main():
    """Main execution function."""
    print("🚀 LIVE SENTIMENT ANALYSIS PIPELINE")
    print("=" * 50)
    
    pipeline = LiveSentimentPipeline()
    
    # Check system status
    status = pipeline.check_system_status()
    print(f"🔧 System Status:")
    print(f"   Twitter API: {'✅' if status['twitter_api'] else '❌'}")
    print(f"   Sentiment Analyzer: {'✅' if status['sentiment_analyzer'] else '❌'}")
    print(f"   Overall: {'✅' if status['overall'] else '❌'}")
    
    if not status['overall']:
        print("\n❌ System not ready. Please check Twitter API credentials.")
        return
    
    # Menu options
    print("\n🔧 PIPELINE OPTIONS:")
    print("1. Get current sentiment for prediction")
    print("2. Generate sentiment report")
    print("3. Start continuous monitoring")
    print("4. Test live sentiment features")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        # Get sentiment for prediction
        target_date = input("Target date (YYYY-MM-DD, or press Enter for yesterday): ").strip()
        if not target_date:
            target_date = None
        
        sentiment = pipeline.get_sentiment_for_prediction(target_date)
        
        if not sentiment.empty:
            print(f"\n📊 SENTIMENT FOR PREDICTION:")
            latest = sentiment.iloc[-1]
            print(f"Date: {latest['date']}")
            print(f"Compound sentiment: {latest['compound_mean']:.3f}")
            print(f"Sentiment volatility: {latest['sentiment_volatility']:.3f}")
            print(f"Total tweets: {latest['total_tweets']}")
            print(f"Source: {latest.get('source', 'simulated')}")
        else:
            print("❌ No sentiment data available")
    
    elif choice == "2":
        # Generate report
        report = pipeline.generate_sentiment_report()
        
        if report['status'] == 'success':
            print(f"\n📊 SENTIMENT REPORT")
            print(f"Timestamp: {report['timestamp']}")
            print(f"Current sentiment: {report['sentiment_metrics']['current_sentiment']:.3f}")
            print(f"Trend: {report['sentiment_metrics']['sentiment_trend']}")
            print(f"24h average: {report['summary']['avg_sentiment_24h']:.3f}")
            print(f"Tweet volume: {report['summary']['tweet_volume_24h']}")
            print(f"Dominant sentiment: {report['summary']['dominant_sentiment']}")
        else:
            print(f"❌ {report['message']}")
    
    elif choice == "3":
        # Continuous monitoring
        interval = int(input("Update interval (minutes, default 30): ") or 30)
        duration = int(input("Duration (hours, default 24): ") or 24)
        
        pipeline.run_continuous_monitoring(interval, duration)
    
    elif choice == "4":
        # Test live features
        hours = int(input("Hours back to analyze (default 6): ") or 6)
        sentiment_data = pipeline.get_live_sentiment_features(hours_back=hours)
        
        if not sentiment_data.empty:
            print(f"\n📊 LIVE SENTIMENT TEST RESULTS:")
            print(f"Days of data: {len(sentiment_data)}")
            print(f"Average sentiment: {sentiment_data['compound_mean'].mean():.3f}")
            print(f"Total tweets: {sentiment_data['total_tweets'].sum()}")
            print(sentiment_data[['date', 'compound_mean', 'total_tweets', 'sentiment_volatility']].head())
        else:
            print("❌ No live sentiment data available")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
