#!/usr/bin/env python3
"""
Advanced Sentiment Feature Engineering Pipeline
Creates sophisticated sentiment features for ML models
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentFeatureEngineer:
    """Advanced sentiment feature engineering for Bitcoin prediction."""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.sentiment_db = self.data_dir / "sentiment_data.db"
        self.features_file = self.data_dir / "btc_sentiment_features_enhanced.csv"
        
    def load_sentiment_data(self):
        """Load sentiment data from database and CSV files."""
        try:
            # Load existing processed sentiment data
            if os.path.exists(self.data_dir / "processed_sentiment_price_data.csv"):
                logger.info("📊 Loading existing sentiment+price data...")
                df = pd.read_csv(self.data_dir / "processed_sentiment_price_data.csv")
                df['date'] = pd.to_datetime(df['date'])
                logger.info(f"✅ Loaded {len(df)} records from processed data")
                return df
            
            # Fallback to sentiment features
            elif os.path.exists(self.data_dir / "sentiment_features.csv"):
                logger.info("📊 Loading sentiment features...")
                df = pd.read_csv(self.data_dir / "sentiment_features.csv")
                df['date'] = pd.to_datetime(df['date'])
                logger.info(f"✅ Loaded {len(df)} sentiment records")
                return df
                
            else:
                logger.warning("⚠️ No sentiment data found, creating mock data")
                return self.create_mock_sentiment_data()
                
        except Exception as e:
            logger.error(f"❌ Error loading sentiment data: {e}")
            return self.create_mock_sentiment_data()
    
    def create_mock_sentiment_data(self, days=2000):
        """Create mock sentiment data for testing."""
        logger.info("🎭 Creating mock sentiment data...")
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate realistic sentiment patterns
        np.random.seed(42)
        base_sentiment = 0.1  # Slight positive bias for Bitcoin
        
        data = []
        for date in dates:
            # Add market sentiment cycles
            day_of_year = date.dayofyear
            cycle_sentiment = 0.3 * np.sin(2 * np.pi * day_of_year / 365)
            
            # Add random noise
            noise = np.random.normal(0, 0.2)
            compound_mean = base_sentiment + cycle_sentiment + noise
            
            # Generate correlated features
            total_tweets = max(50, int(np.random.normal(300, 100)))
            sentiment_volatility = abs(np.random.normal(0.2, 0.1))
            
            positive_ratio = max(0.1, min(0.9, 0.4 + compound_mean * 0.5))
            negative_ratio = max(0.1, min(0.8, 0.3 - compound_mean * 0.3))
            neutral_ratio = 1 - positive_ratio - negative_ratio
            
            data.append({
                'date': date,
                'compound_mean': compound_mean,
                'compound_std': sentiment_volatility,
                'positive_mean': positive_ratio * 0.8,
                'negative_mean': negative_ratio * 0.6,
                'neutral_mean': neutral_ratio * 0.7,
                'total_tweets': total_tweets,
                'engagement_score_sum': int(total_tweets * np.random.uniform(15, 25)),
                'sentiment_positive_ratio': positive_ratio,
                'sentiment_negative_ratio': negative_ratio,
                'sentiment_neutral_ratio': neutral_ratio,
                'sentiment_volatility': sentiment_volatility,
                'weighted_sentiment': compound_mean * (total_tweets / 300)
            })
            
        df = pd.DataFrame(data)
        logger.info(f"✅ Created {len(df)} mock sentiment records")
        return df
    
    def create_advanced_sentiment_features(self, df):
        """Create advanced sentiment features for ML models."""
        logger.info("🔧 Creating advanced sentiment features...")
        
        # Ensure data is sorted by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # 1. Sentiment Momentum Features
        df['sentiment_momentum_3d'] = df['compound_mean'].rolling(3).apply(lambda x: x.iloc[-1] - x.iloc[0])
        df['sentiment_momentum_7d'] = df['compound_mean'].rolling(7).apply(lambda x: x.iloc[-1] - x.iloc[0])
        df['sentiment_momentum_14d'] = df['compound_mean'].rolling(14).apply(lambda x: x.iloc[-1] - x.iloc[0])
        
        # 2. Sentiment Volatility Features
        df['sentiment_volatility_lag_1d'] = df['sentiment_volatility'].shift(1)
        df['sentiment_volatility_lag_3d'] = df['sentiment_volatility'].shift(3)
        df['sentiment_volatility_lag_7d'] = df['sentiment_volatility'].shift(7)
        
        # 3. Sentiment Moving Averages and Ratios
        for window in [3, 7, 14, 21]:
            df[f'sentiment_ma_{window}d'] = df['compound_mean'].rolling(window).mean()
            df[f'sentiment_ratio_{window}d'] = df['compound_mean'] / df[f'sentiment_ma_{window}d']
            df[f'sentiment_std_{window}d'] = df['compound_mean'].rolling(window).std()
        
        # 4. Sentiment Extremes and Divergence
        df['sentiment_z_score_7d'] = (df['compound_mean'] - df['sentiment_ma_7d']) / df['sentiment_std_7d']
        df['sentiment_percentile_14d'] = df['compound_mean'].rolling(14).rank(pct=True)
        
        # 5. Volume-Weighted Sentiment Features
        df['volume_weighted_sentiment'] = df['compound_mean'] * np.log1p(df['total_tweets'])
        df['engagement_per_tweet'] = df['engagement_score_sum'] / df['total_tweets']
        df['engagement_sentiment_ratio'] = df['engagement_per_tweet'] * df['compound_mean']
        
        # 6. Sentiment Polarity Features
        df['sentiment_polarity_strength'] = df['sentiment_positive_ratio'] - df['sentiment_negative_ratio']
        df['sentiment_uncertainty'] = df['sentiment_neutral_ratio'] * df['sentiment_volatility']
        df['sentiment_conviction'] = (df['sentiment_positive_ratio'] + df['sentiment_negative_ratio']) / df['sentiment_neutral_ratio']
        
        # 7. Lagged Sentiment Features
        for lag in [1, 2, 3, 7]:
            df[f'compound_mean_lag_{lag}d'] = df['compound_mean'].shift(lag)
            df[f'sentiment_positive_ratio_lag_{lag}d'] = df['sentiment_positive_ratio'].shift(lag)
            df[f'weighted_sentiment_lag_{lag}d'] = df['weighted_sentiment'].shift(lag)
        
        # 8. Sentiment Change and Acceleration
        df['sentiment_change_1d'] = df['compound_mean'].diff()
        df['sentiment_change_3d'] = df['compound_mean'].diff(3)
        df['sentiment_acceleration'] = df['sentiment_change_1d'].diff()
        
        # 9. Sentiment Regime Detection
        df['sentiment_regime_bull'] = (df['sentiment_ma_7d'] > df['sentiment_ma_21d']).astype(int)
        df['sentiment_regime_bear'] = (df['sentiment_ma_7d'] < df['sentiment_ma_21d']).astype(int)
        
        # 10. Cross-correlations and Interactions
        if 'Close' in df.columns:
            df['sentiment_return_interaction'] = df['compound_mean'] * df.get('btc_return', 0)
            df['sentiment_volatility_interaction'] = df['sentiment_volatility'] * df.get('volatility', 0)
            
            # Price-sentiment divergence
            price_change = df['Close'].pct_change()
            df['price_sentiment_divergence'] = (price_change > 0).astype(int) - (df['compound_mean'] > 0).astype(int)
            
        # 11. Technical Analysis on Sentiment
        # RSI-like indicator for sentiment
        sentiment_delta = df['compound_mean'].diff()
        sentiment_gain = sentiment_delta.where(sentiment_delta > 0, 0)
        sentiment_loss = -sentiment_delta.where(sentiment_delta < 0, 0)
        sentiment_rs = sentiment_gain.rolling(14).mean() / sentiment_loss.rolling(14).mean()
        df['sentiment_rsi'] = 100 - (100 / (1 + sentiment_rs))
        
        # Bollinger Bands for sentiment
        sentiment_ma_20 = df['compound_mean'].rolling(20).mean()
        sentiment_std_20 = df['compound_mean'].rolling(20).std()
        df['sentiment_bb_upper'] = sentiment_ma_20 + (2 * sentiment_std_20)
        df['sentiment_bb_lower'] = sentiment_ma_20 - (2 * sentiment_std_20)
        df['sentiment_bb_position'] = (df['compound_mean'] - df['sentiment_bb_lower']) / (df['sentiment_bb_upper'] - df['sentiment_bb_lower'])
        
        # 12. Sentiment Trend Strength
        df['sentiment_trend_strength'] = abs(df['sentiment_momentum_7d']) * df['total_tweets'] / 300
        
        logger.info(f"✅ Created {len([col for col in df.columns if 'sentiment' in col])} sentiment features")
        return df
    
    def merge_with_price_data(self, sentiment_df):
        """Merge sentiment features with Bitcoin price data."""
        try:
            # Load Bitcoin price data
            btc_file = self.data_dir / "btc_data.csv"
            if os.path.exists(btc_file):
                logger.info("📈 Loading Bitcoin price data...")
                price_df = pd.read_csv(btc_file)
                price_df['Date'] = pd.to_datetime(price_df['Date'])
                
                # Merge on date
                merged_df = pd.merge(sentiment_df, price_df, left_on='date', right_on='Date', how='inner')
                logger.info(f"✅ Merged sentiment and price data: {len(merged_df)} records")
                
                # Add price-based features
                merged_df = self.add_price_features(merged_df)
                
                return merged_df
            else:
                logger.warning("⚠️ No Bitcoin price data found")
                return sentiment_df
                
        except Exception as e:
            logger.error(f"❌ Error merging with price data: {e}")
            return sentiment_df
    
    def add_price_features(self, df):
        """Add price-based technical features."""
        logger.info("📊 Adding price-based features...")
        
        # Basic price features
        df['price_return_1d'] = df['Close'].pct_change()
        df['price_return_3d'] = df['Close'].pct_change(3)
        df['price_return_7d'] = df['Close'].pct_change(7)
        
        # Price volatility
        df['price_volatility_7d'] = df['price_return_1d'].rolling(7).std()
        df['price_volatility_14d'] = df['price_return_1d'].rolling(14).std()
        
        # Moving averages
        for window in [7, 14, 21, 50]:
            df[f'price_ma_{window}'] = df['Close'].rolling(window).mean()
            df[f'price_ratio_{window}'] = df['Close'] / df[f'price_ma_{window}']
        
        # RSI
        price_delta = df['Close'].diff()
        price_gain = price_delta.where(price_delta > 0, 0).rolling(14).mean()
        price_loss = -price_delta.where(price_delta < 0, 0).rolling(14).mean()
        price_rs = price_gain / price_loss
        df['price_rsi'] = 100 - (100 / (1 + price_rs))
        
        # Bollinger Bands
        price_ma_20 = df['Close'].rolling(20).mean()
        price_std_20 = df['Close'].rolling(20).std()
        df['price_bb_upper'] = price_ma_20 + (2 * price_std_20)
        df['price_bb_lower'] = price_ma_20 - (2 * price_std_20)
        df['price_bb_position'] = (df['Close'] - df['price_bb_lower']) / (df['price_bb_upper'] - df['price_bb_lower'])
        
        # Sentiment-Price interaction features
        df['sentiment_rsi_interaction'] = df['sentiment_rsi'] * df['price_rsi']
        df['sentiment_momentum_price_return'] = df['sentiment_momentum_7d'] * df['price_return_7d']
        
        logger.info("✅ Added price-based features")
        return df
    
    def create_target_variable(self, df):
        """Create target variable for prediction."""
        # Predict next day's price direction
        df['next_day_return'] = df['Close'].pct_change().shift(-1)
        df['target'] = (df['next_day_return'] > 0).astype(int)
        
        # Remove last row (no target)
        df = df[:-1].copy()
        
        logger.info("✅ Created target variable")
        return df
    
    def save_features(self, df):
        """Save enhanced features to CSV."""
        try:
            # Select feature columns (exclude intermediate calculations)
            feature_cols = [col for col in df.columns if not col.startswith('price_ma_') or col.endswith('_ratio')]
            
            # Save full dataset
            output_file = self.data_dir / "btc_sentiment_features_enhanced.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"✅ Saved enhanced features to {output_file}")
            
            # Save feature list for model training
            feature_importance_file = self.data_dir / "sentiment_feature_list.csv"
            feature_df = pd.DataFrame({
                'feature': [col for col in df.columns if col not in ['date', 'Date', 'target', 'next_day_return']],
                'type': ['sentiment' if 'sentiment' in col else 'price' if any(x in col for x in ['Close', 'price', 'rsi', 'bb']) else 'other' 
                        for col in df.columns if col not in ['date', 'Date', 'target', 'next_day_return']]
            })
            feature_df.to_csv(feature_importance_file, index=False)
            logger.info(f"✅ Saved feature list to {feature_importance_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"❌ Error saving features: {e}")
            return None
    
    def run_feature_engineering_pipeline(self):
        """Run the complete feature engineering pipeline."""
        logger.info("🚀 Starting sentiment feature engineering pipeline...")
        
        try:
            # 1. Load sentiment data
            sentiment_df = self.load_sentiment_data()
            
            # 2. Create advanced sentiment features
            enhanced_df = self.create_advanced_sentiment_features(sentiment_df)
            
            # 3. Merge with price data
            merged_df = self.merge_with_price_data(enhanced_df)
            
            # 4. Create target variable
            final_df = self.create_target_variable(merged_df)
            
            # 5. Clean data (remove NaN values)
            initial_rows = len(final_df)
            final_df = final_df.dropna()
            final_rows = len(final_df)
            
            logger.info(f"📊 Data cleaning: {initial_rows} → {final_rows} rows ({initial_rows - final_rows} removed)")
            
            # 6. Save features
            output_file = self.save_features(final_df)
            
            # 7. Summary statistics
            sentiment_features = [col for col in final_df.columns if 'sentiment' in col]
            price_features = [col for col in final_df.columns if any(x in col for x in ['Close', 'price', 'rsi', 'bb']) and 'sentiment' not in col]
            
            logger.info("📈 Feature Engineering Summary:")
            logger.info(f"   📊 Total Records: {len(final_df)}")
            logger.info(f"   📅 Date Range: {final_df['date'].min().strftime('%Y-%m-%d')} to {final_df['date'].max().strftime('%Y-%m-%d')}")
            logger.info(f"   🎭 Sentiment Features: {len(sentiment_features)}")
            logger.info(f"   💰 Price Features: {len(price_features)}")
            logger.info(f"   📊 Total Features: {len(final_df.columns) - 4}")  # Exclude date, Date, target, next_day_return
            logger.info(f"   🎯 Target Distribution: {final_df['target'].value_counts().to_dict()}")
            
            logger.info("🎉 Feature engineering pipeline completed successfully!")
            return output_file, final_df
            
        except Exception as e:
            logger.error(f"❌ Feature engineering pipeline failed: {e}")
            return None, None

def main():
    """Run the sentiment feature engineering pipeline."""
    engineer = SentimentFeatureEngineer()
    output_file, df = engineer.run_feature_engineering_pipeline()
    
    if output_file:
        print(f"\n✅ Enhanced features saved to: {output_file}")
        print(f"📊 Dataset shape: {df.shape}")
        print("\n🎯 Ready for model training!")
    else:
        print("\n❌ Feature engineering failed!")
        return False
    
    return True

if __name__ == "__main__":
    main()