#!/usr/bin/env python3
"""
FREE TIER BITCOIN SENTIMENT ANALYSIS - FIXED VERSION
Analyze existing data without requiring new Twitter API calls
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data():
    """Load and clean existing data."""
    print("🔧 Loading and cleaning data...")
    
    # Load sentiment data
    sentiment_df = pd.read_csv('data/sentiment_features.csv')
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    
    # Load BTC data - handle the header row issue
    btc_raw = pd.read_csv('data/btc_data.csv')
    
    # Check if first row has actual data (not header)
    if btc_raw.iloc[1, 0] == '':  # Empty Date field means it's a header row
        btc_df = btc_raw.iloc[2:].copy()  # Skip first two rows
        btc_df.columns = btc_raw.iloc[0]  # Use first row as column names
    else:
        btc_df = btc_raw.copy()
    
    # Clean column names and reset index
    btc_df = btc_df.reset_index(drop=True)
    btc_df.columns.name = None
    
    # Convert Date column
    btc_df['Date'] = pd.to_datetime(btc_df['Date'])
    
    # Convert price columns to numeric, handling any text values
    price_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
    for col in price_cols:
        if col in btc_df.columns:
            btc_df[col] = pd.to_numeric(btc_df[col], errors='coerce')
    
    # Remove any rows with NaN values in key columns
    btc_df = btc_df.dropna(subset=['Date', 'Close'])
    
    print(f"✅ Sentiment data: {len(sentiment_df)} rows")
    print(f"✅ BTC price data: {len(btc_df)} rows")
    
    return sentiment_df, btc_df

def comprehensive_analysis():
    """Run comprehensive analysis without API calls."""
    print("🚀 FREE TIER BITCOIN SENTIMENT ANALYSIS")
    print("🎯 Analyzing existing data (No Twitter API needed)")
    print("=" * 60)
    
    # Load data
    sentiment_df, btc_df = load_and_clean_data()
    
    # Merge datasets
    merged_df = pd.merge(
        sentiment_df, 
        btc_df, 
        left_on='date', 
        right_on='Date', 
        how='inner'
    )
    
    print(f"🔄 Merged dataset: {len(merged_df)} days")
    print(f"📊 Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    
    # Basic statistics
    print(f"\n📈 BASIC STATISTICS")
    print("-" * 30)
    print(f"Total tweets analyzed: {sentiment_df['total_tweets'].sum():,}")
    print(f"Avg daily sentiment: {merged_df['compound_mean'].mean():.3f}")
    print(f"Avg BTC price: ${merged_df['Close'].mean():,.2f}")
    print(f"BTC price range: ${merged_df['Close'].min():,.2f} - ${merged_df['Close'].max():,.2f}")
    
    # Calculate returns
    merged_df['btc_return'] = merged_df['Close'].pct_change()
    
    # Correlations
    sentiment_price_corr = merged_df['compound_mean'].corr(merged_df['Close'])
    sentiment_return_corr = merged_df['compound_mean'].corr(merged_df['btc_return'])
    
    print(f"\n🧠 CORRELATION ANALYSIS")
    print("-" * 30)
    print(f"Sentiment ↔ Price: {sentiment_price_corr:.3f}")
    print(f"Sentiment ↔ Returns: {sentiment_return_corr:.3f}")
    
    # Sentiment extremes analysis
    high_sentiment = merged_df[merged_df['compound_mean'] > merged_df['compound_mean'].quantile(0.8)]
    low_sentiment = merged_df[merged_df['compound_mean'] < merged_df['compound_mean'].quantile(0.2)]
    
    print(f"\n📊 SENTIMENT EXTREMES IMPACT")
    print("-" * 30)
    print(f"High sentiment periods: {len(high_sentiment)} days")
    print(f"Avg return on high sentiment: {high_sentiment['btc_return'].mean():.2%}")
    print(f"Low sentiment periods: {len(low_sentiment)} days")
    print(f"Avg return on low sentiment: {low_sentiment['btc_return'].mean():.2%}")
    
    # Feature engineering for simple model
    print(f"\n🛠️ FEATURE ENGINEERING")
    print("-" * 30)
    
    # Technical indicators
    merged_df['price_ma_7'] = merged_df['Close'].rolling(7).mean()
    merged_df['price_ma_21'] = merged_df['Close'].rolling(21).mean()
    merged_df['volatility'] = merged_df['Close'].rolling(7).std()
    
    # Sentiment features
    merged_df['sentiment_ma_7'] = merged_df['compound_mean'].rolling(7).mean()
    merged_df['sentiment_change'] = merged_df['compound_mean'].diff()
    
    # Target: Next day return
    merged_df['next_day_return'] = merged_df['Close'].shift(-1) / merged_df['Close'] - 1
    
    # Select features for modeling
    features = [
        'compound_mean', 'sentiment_volatility', 'sentiment_ma_7', 'sentiment_change',
        'positive_mean', 'negative_mean', 'total_tweets',
        'price_ma_7', 'price_ma_21', 'volatility'
    ]
    
    # Clean data for modeling
    model_df = merged_df[features + ['next_day_return', 'date']].dropna()
    
    print(f"✅ Model features: {len(features)}")
    print(f"✅ Clean dataset: {len(model_df)} samples")
    
    # Simple prediction model
    print(f"\n🤖 SIMPLE PREDICTION MODEL")
    print("-" * 30)
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        
        # Prepare data
        X = model_df[features]
        y = model_df['next_day_return']
        
        # Split data (last 20% for testing)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"✅ Model R² Score: {r2:.3f}")
        print(f"✅ Model RMSE: {np.sqrt(mse):.4f}")
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🏆 TOP 5 MOST IMPORTANT FEATURES:")
        for idx, row in importance_df.head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
    except ImportError:
        print("⚠️ Scikit-learn not available, skipping ML model")
    
    # Save results
    print(f"\n💾 SAVING RESULTS")
    print("-" * 30)
    
    # Summary results
    results = {
        'analysis_date': datetime.now().isoformat(),
        'total_days': len(merged_df),
        'total_tweets': int(sentiment_df['total_tweets'].sum()),
        'sentiment_price_correlation': sentiment_price_corr,
        'sentiment_return_correlation': sentiment_return_corr,
        'avg_sentiment': merged_df['compound_mean'].mean(),
        'avg_price': merged_df['Close'].mean(),
        'data_quality': 'good',
        'api_usage': 'none_required'
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv('data/free_tier_analysis_summary.csv', index=False)
    
    # Save processed data
    merged_df.to_csv('data/processed_sentiment_price_data.csv', index=False)
    
    print(f"✅ Saved analysis summary")
    print(f"✅ Saved processed dataset")
    
    # Create simple visualization
    create_visualizations(merged_df)
    
    return merged_df

def create_visualizations(df):
    """Create visualizations without requiring API calls."""
    print(f"\n📊 CREATING VISUALIZATIONS")
    print("-" * 30)
    
    # Set style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Sentiment over time
    axes[0, 0].plot(df['date'], df['compound_mean'], alpha=0.7, color='blue')
    axes[0, 0].set_title('Bitcoin Sentiment Over Time')
    axes[0, 0].set_ylabel('Compound Sentiment')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Price over time
    axes[0, 1].plot(df['date'], df['Close'], alpha=0.7, color='orange')
    axes[0, 1].set_title('Bitcoin Price Over Time')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Sentiment vs Price scatter
    axes[1, 0].scatter(df['compound_mean'], df['Close'], alpha=0.5, s=1)
    axes[1, 0].set_xlabel('Sentiment')
    axes[1, 0].set_ylabel('BTC Price ($)')
    axes[1, 0].set_title('Sentiment vs Price Relationship')
    
    # 4. Returns distribution
    if 'btc_return' in df.columns:
        axes[1, 1].hist(df['btc_return'].dropna(), bins=50, alpha=0.7, color='green')
        axes[1, 1].set_xlabel('Daily Returns')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Bitcoin Daily Returns Distribution')
    
    plt.tight_layout()
    plt.savefig('plots/free_tier_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization: plots/free_tier_comprehensive_analysis.png")
    plt.close()

def main():
    """Main function - no API calls needed."""
    try:
        df = comprehensive_analysis()
        
        print(f"\n🎉 FREE TIER ANALYSIS COMPLETE!")
        print(f"✅ No Twitter API calls required")
        print(f"✅ Used {len(df)} days of existing data")
        print(f"✅ Generated insights and predictions")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. ✅ Analyzed existing sentiment data (DONE)")
        print(f"   2. 🔄 Build Reddit scraper (FREE alternative)")
        print(f"   3. 🔄 Add news sentiment (FREE)")
        print(f"   4. ⏳ Wait for Twitter quota reset (Aug 24)")
        print(f"   5. 🎯 Strategic Twitter collection (3-4 tweets/day)")
        
        return df
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        return None

if __name__ == "__main__":
    main()
