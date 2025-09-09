#!/usr/bin/env python3
"""
FREE TIER DAILY AUTOMATION
Complete Bitcoin sentiment analysis workflow optimized for free tier
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import subprocess
import sys

def run_script(script_path):
    """Run a Python script and return success status."""
    try:
        print(f"🔄 Running {script_path}...")
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"✅ {script_path} completed successfully")
            return True
        else:
            print(f"❌ {script_path} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running {script_path}: {e}")
        return False

def daily_free_tier_workflow():
    """Complete daily workflow for free tier users."""
    print("🚀 FREE TIER DAILY BITCOIN SENTIMENT WORKFLOW")
    print("=" * 60)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    workflow_results = {
        'date': datetime.now(),
        'twitter_quota_status': 'exhausted_until_aug_24',
        'free_sources_used': [],
        'success': True
    }
    
    # Step 1: Collect free sentiment data
    print(f"\n🔍 STEP 1: FREE SENTIMENT DATA COLLECTION")
    print("-" * 40)
    
    free_sentiment_success = run_script("scripts/free_sentiment_scraper.py")
    
    if free_sentiment_success:
        workflow_results['free_sources_used'].append('reddit_news_fear_greed')
        print("✅ Free sentiment data collected")
    else:
        print("⚠️ Free sentiment collection had issues")
        workflow_results['success'] = False
    
    # Step 2: Update comprehensive analysis
    print(f"\n📊 STEP 2: COMPREHENSIVE ANALYSIS UPDATE")
    print("-" * 40)
    
    analysis_success = run_script("scripts/free_tier_comprehensive.py")
    
    if analysis_success:
        print("✅ Analysis updated with latest data")
    else:
        print("⚠️ Analysis update had issues")
        workflow_results['success'] = False
    
    # Step 3: Generate predictions
    print(f"\n🔮 STEP 3: PREDICTION GENERATION")
    print("-" * 40)
    
    try:
        # Check if predictor script exists, if not create a simple one
        if os.path.exists("scripts/predictor.py"):
            prediction_success = run_script("scripts/predictor.py")
        else:
            prediction_success = create_simple_predictor()
        
        if prediction_success:
            print("✅ Predictions generated")
        else:
            print("⚠️ Prediction generation had issues")
            workflow_results['success'] = False
            
    except Exception as e:
        print(f"⚠️ Prediction step error: {e}")
        workflow_results['success'] = False
    
    # Step 4: Create daily summary
    print(f"\n📋 STEP 4: DAILY SUMMARY")
    print("-" * 40)
    
    create_daily_summary(workflow_results)
    
    # Final status
    if workflow_results['success']:
        print(f"\n🎉 DAILY WORKFLOW COMPLETED SUCCESSFULLY!")
        print(f"💰 Cost: $0 (FREE TIER)")
        print(f"🔄 Next run: Tomorrow at the same time")
        
        # Show next steps
        print(f"\n💡 OPTIMIZATION TIPS:")
        print(f"   • Run this daily for continuous sentiment monitoring")
        print(f"   • Twitter quota resets: August 24, 2025")
        print(f"   • After reset: Add 3-4 strategic tweets per day")
        print(f"   • Current approach uses 100% free data sources")
        
    else:
        print(f"\n⚠️ DAILY WORKFLOW COMPLETED WITH ISSUES")
        print(f"🔧 Check individual steps above for details")
    
    return workflow_results

def create_simple_predictor():
    """Create a simple predictor if none exists."""
    print("🤖 Creating simple predictor...")
    
    try:
        # Load latest processed data
        if os.path.exists('data/processed_sentiment_price_data.csv'):
            df = pd.read_csv('data/processed_sentiment_price_data.csv')
            
            # Simple prediction based on sentiment
            if 'compound_mean' in df.columns and 'Close' in df.columns:
                latest_sentiment = df['compound_mean'].iloc[-1]
                latest_price = df['Close'].iloc[-1]
                
                # Simple rule-based prediction
                if latest_sentiment > 0.2:
                    prediction = "BULLISH - High positive sentiment"
                    direction = "UP"
                elif latest_sentiment < -0.2:
                    prediction = "BEARISH - High negative sentiment" 
                    direction = "DOWN"
                else:
                    prediction = "NEUTRAL - Mixed sentiment"
                    direction = "SIDEWAYS"
                
                # Save prediction
                prediction_data = {
                    'timestamp': datetime.now(),
                    'current_price': latest_price,
                    'current_sentiment': latest_sentiment,
                    'prediction': prediction,
                    'direction': direction,
                    'confidence': min(abs(latest_sentiment) * 100, 100)
                }
                
                pred_df = pd.DataFrame([prediction_data])
                os.makedirs('predictions', exist_ok=True)
                pred_df.to_csv('predictions/daily_prediction.csv', index=False)
                
                print(f"✅ Simple prediction: {prediction}")
                return True
        
        print("⚠️ Insufficient data for predictions")
        return False
        
    except Exception as e:
        print(f"❌ Predictor creation failed: {e}")
        return False

def create_daily_summary(workflow_results):
    """Create daily summary report."""
    try:
        summary = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'workflow_success': workflow_results['success'],
            'free_sources_active': len(workflow_results['free_sources_used']),
            'twitter_quota_remaining': 80,  # Until Aug 24
            'cost': 0.00,
            'data_sources': 'reddit, news, fear_greed_index, historical'
        }
        
        # Load recent sentiment if available
        if os.path.exists('data/free_crypto_sentiment.csv'):
            sentiment_df = pd.read_csv('data/free_crypto_sentiment.csv')
            if not sentiment_df.empty:
                summary['latest_sentiment'] = sentiment_df['sentiment'].mean()
                summary['sentiment_items_today'] = len(sentiment_df)
        
        # Save summary
        summary_df = pd.DataFrame([summary])
        os.makedirs('logs', exist_ok=True)
        
        # Append to daily log
        log_file = 'logs/daily_workflow_log.csv'
        if os.path.exists(log_file):
            summary_df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            summary_df.to_csv(log_file, index=False)
        
        print(f"✅ Daily summary saved to logs/")
        
        # Print summary
        print(f"\n📊 TODAY'S SUMMARY:")
        if 'latest_sentiment' in summary:
            print(f"   Sentiment Score: {summary['latest_sentiment']:.3f}")
        if 'sentiment_items_today' in summary:
            print(f"   Data Points: {summary['sentiment_items_today']}")
        print(f"   Cost: ${summary['cost']:.2f}")
        print(f"   Twitter Quota: {summary['twitter_quota_remaining']} remaining")
        
    except Exception as e:
        print(f"⚠️ Summary creation error: {e}")

if __name__ == "__main__":
    # Run the complete daily workflow
    results = daily_free_tier_workflow()
    
    print(f"\n🔄 AUTOMATION SETUP:")
    print(f"   • Windows: Create a scheduled task to run this daily")
    print(f"   • Command: python scripts/daily_free_tier_workflow.py") 
    print(f"   • Frequency: Once per day (morning recommended)")
    print(f"   • Duration: ~2-3 minutes per run")
    
    exit(0 if results['success'] else 1)
