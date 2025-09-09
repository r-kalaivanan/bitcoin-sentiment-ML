#!/usr/bin/env python3
"""
FREE TIER BITCOIN SENTIMENT ANALYSIS
Maximize existing data while Twitter quota is exhausted
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def analyze_existing_sentiment_data():
    """Comprehensive analysis of existing sentiment data."""
    print("🔍 FREE TIER: Analyzing existing sentiment data...")
    print("=" * 60)
    
    # Load existing data
    try:
        sentiment_df = pd.read_csv('data/sentiment_features.csv')
        btc_df = pd.read_csv('data/btc_data.csv')
        
        print(f"✅ Loaded sentiment data: {len(sentiment_df)} days")
        print(f"✅ Loaded BTC price data: {len(btc_df)} days")
        
        # Convert dates
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        btc_df['Date'] = pd.to_datetime(btc_df['Date'])
        
        # Data summary
        print(f"\n📊 SENTIMENT DATA SUMMARY")
        print(f"Date range: {sentiment_df['date'].min()} to {sentiment_df['date'].max()}")
        print(f"Total tweets analyzed: {sentiment_df['total_tweets'].sum():,}")
        print(f"Avg daily sentiment: {sentiment_df['compound_mean'].mean():.3f}")
        print(f"Sentiment volatility: {sentiment_df['sentiment_volatility'].mean():.3f}")
        
        # Merge datasets
        merged_df = pd.merge(
            sentiment_df, 
            btc_df, 
            left_on='date', 
            right_on='Date', 
            how='inner'
        )
        print(f"\n🔄 Merged dataset: {len(merged_df)} days")
        
        return merged_df
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return None

def create_sentiment_insights(df):
    """Generate insights from sentiment data."""
    if df is None:
        return
    
    print(f"\n🧠 SENTIMENT INSIGHTS")
    print("-" * 40)
    
    # Correlation analysis
    sentiment_price_corr = df['compound_mean'].corr(df['Close'])
    print(f"📈 Sentiment-Price Correlation: {sentiment_price_corr:.3f}")
    
    # Sentiment vs Returns
    df['btc_return'] = df['Close'].pct_change()
    sentiment_return_corr = df['compound_mean'].corr(df['btc_return'])
    print(f"💹 Sentiment-Return Correlation: {sentiment_return_corr:.3f}")
    
    # High/Low sentiment periods
    high_sentiment_days = df[df['compound_mean'] > df['compound_mean'].quantile(0.8)]
    low_sentiment_days = df[df['compound_mean'] < df['compound_mean'].quantile(0.2)]
    
    print(f"\n📊 SENTIMENT EXTREMES")
    print(f"High sentiment days: {len(high_sentiment_days)}")
    print(f"Avg return on high sentiment: {high_sentiment_days['btc_return'].mean():.3%}")
    print(f"Low sentiment days: {len(low_sentiment_days)}")
    print(f"Avg return on low sentiment: {low_sentiment_days['btc_return'].mean():.3%}")
    
    # Volume correlation
    if 'Volume' in df.columns:
        volume_sentiment_corr = df['compound_mean'].corr(df['Volume'])
        print(f"📊 Sentiment-Volume Correlation: {volume_sentiment_corr:.3f}")
    
    return {
        'sentiment_price_corr': sentiment_price_corr,
        'sentiment_return_corr': sentiment_return_corr,
        'high_sentiment_return': high_sentiment_days['btc_return'].mean(),
        'low_sentiment_return': low_sentiment_days['btc_return'].mean()
    }

def create_prediction_features(df):
    """Create features for prediction model."""
    if df is None:
        return None
    
    print(f"\n🛠️ FEATURE ENGINEERING")
    print("-" * 40)
    
    # Technical indicators
    df['price_ma_7'] = df['Close'].rolling(7).mean()
    df['price_ma_21'] = df['Close'].rolling(21).mean()
    df['price_volatility'] = df['Close'].rolling(7).std()
    
    # Sentiment features
    df['sentiment_ma_3'] = df['compound_mean'].rolling(3).mean()
    df['sentiment_trend'] = df['compound_mean'].diff()
    
    # Target variable (next day return)
    df['target'] = df['Close'].shift(-1) / df['Close'] - 1
    
    # Feature list
    features = [
        'compound_mean', 'sentiment_volatility', 'sentiment_ma_3', 'sentiment_trend',
        'positive_mean', 'negative_mean', 'total_tweets', 'engagement_score_sum',
        'price_ma_7', 'price_ma_21', 'price_volatility', 'Volume'
    ]
    
    # Clean features
    feature_df = df[features + ['target', 'date']].dropna()
    
    print(f"✅ Created {len(features)} features")
    print(f"✅ Clean dataset: {len(feature_df)} samples")
    
    return feature_df, features

def build_simple_model(feature_df, features):
    """Build simple prediction model."""
    if feature_df is None:
        return None
    
    print(f"\n🤖 BUILDING PREDICTION MODEL")
    print("-" * 40)
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Prepare data
    X = feature_df[features]
    y = feature_df['target']
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"✅ Model trained successfully")
    print(f"📊 R² Score: {r2:.3f}")
    print(f"📊 MSE: {mse:.6f}")
    print(f"📊 RMSE: {np.sqrt(mse):.6f}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🏆 TOP 5 FEATURES:")
    for idx, row in importance_df.head().iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    return model, importance_df, {
        'r2': r2, 
        'mse': mse, 
        'test_size': len(X_test)
    }

def save_analysis_results(insights, importance_df, model_metrics):
    """Save analysis results."""
    print(f"\n💾 SAVING RESULTS")
    print("-" * 40)
    
    # Create results summary
    results = {
        'analysis_date': datetime.now().isoformat(),
        'data_type': 'existing_sentiment_data',
        'twitter_quota_status': 'exhausted_free_tier',
        **insights,
        **model_metrics
    }
    
    # Save to files
    results_df = pd.DataFrame([results])
    results_df.to_csv('data/free_tier_analysis_results.csv', index=False)
    
    importance_df.to_csv('data/feature_importance_free_tier.csv', index=False)
    
    print(f"✅ Saved analysis results")
    print(f"✅ Saved feature importance")

def create_visualization(df):
    """Create visualizations."""
    if df is None:
        return
    
    print(f"\n📊 CREATING VISUALIZATIONS")
    print("-" * 40)
    
    plt.figure(figsize=(15, 10))
    
    # Sentiment vs Price
    plt.subplot(2, 2, 1)
    plt.plot(df['date'], df['compound_mean'], alpha=0.7, label='Sentiment')
    plt.title('Bitcoin Sentiment Over Time')
    plt.ylabel('Compound Sentiment')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Price chart
    plt.subplot(2, 2, 2)
    plt.plot(df['date'], df['Close'], color='orange', alpha=0.7)
    plt.title('Bitcoin Price Over Time')
    plt.ylabel('Price ($)')
    plt.xticks(rotation=45)
    
    # Correlation scatter
    plt.subplot(2, 2, 3)
    plt.scatter(df['compound_mean'], df['Close'], alpha=0.5)
    plt.xlabel('Sentiment')
    plt.ylabel('BTC Price')
    plt.title('Sentiment vs Price Correlation')
    
    # Returns distribution
    plt.subplot(2, 2, 4)
    df['btc_return'].hist(bins=50, alpha=0.7)
    plt.xlabel('Daily Returns')
    plt.ylabel('Frequency')
    plt.title('Bitcoin Returns Distribution')
    
    plt.tight_layout()
    plt.savefig('plots/free_tier_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization: plots/free_tier_analysis.png")

def main():
    """Main analysis function."""
    print("🚀 FREE TIER BITCOIN SENTIMENT ANALYSIS")
    print("🎯 Making the most of existing data while quota resets")
    print("=" * 60)
    
    # Step 1: Load and analyze existing data
    df = analyze_existing_sentiment_data()
    if df is None:
        print("❌ Cannot proceed without data")
        return
    
    # Step 2: Generate insights
    insights = create_sentiment_insights(df)
    
    # Step 3: Feature engineering
    feature_df, features = create_prediction_features(df)
    
    # Step 4: Build model
    model, importance_df, metrics = build_simple_model(feature_df, features)
    
    # Step 5: Save results
    save_analysis_results(insights, importance_df, metrics)
    
    # Step 6: Visualize
    create_visualization(df)
    
    print(f"\n🎉 FREE TIER ANALYSIS COMPLETE!")
    print(f"💡 Next steps:")
    print(f"   1. Build Reddit scraper for additional sentiment")
    print(f"   2. Add news sentiment analysis") 
    print(f"   3. Wait for Twitter quota reset (Aug 24)")
    print(f"   4. Implement strategic tweet collection")

if __name__ == "__main__":
    main()
