# scripts/data_merger.py

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataMerger:
    """
    Merges Bitcoin price features with sentiment features to create a unified dataset
    for machine learning model training.
    """
    
    def __init__(self):
        self.price_data = None
        self.sentiment_data = None
        self.merged_data = None
    
    def load_price_features(self, filepath='data/btc_features_enhanced.csv'):
        """Load Bitcoin price features."""
        try:
            print(f"📈 Loading price features from {filepath}...")
            self.price_data = pd.read_csv(filepath)
            self.price_data['Date'] = pd.to_datetime(self.price_data['Date']).dt.date
            print(f"✅ Loaded {len(self.price_data)} days of price features")
            return True
        except Exception as e:
            print(f"❌ Error loading price features: {str(e)}")
            return False
    
    def load_sentiment_features(self, filepath='data/sentiment_features.csv'):
        """Load sentiment features."""
        try:
            print(f"🐦 Loading sentiment features from {filepath}...")
            self.sentiment_data = pd.read_csv(filepath)
            self.sentiment_data['date'] = pd.to_datetime(self.sentiment_data['date']).dt.date
            print(f"✅ Loaded {len(self.sentiment_data)} days of sentiment features")
            return True
        except Exception as e:
            print(f"❌ Error loading sentiment features: {str(e)}")
            return False
    
    def merge_datasets(self, merge_type='left'):
        """
        Merge price and sentiment datasets.
        
        Args:
            merge_type: Type of merge ('left', 'inner', 'outer')
                       'left' keeps all price data
                       'inner' keeps only dates with both price and sentiment
                       'outer' keeps all dates from both datasets
        """
        if self.price_data is None or self.sentiment_data is None:
            print("❌ Please load both price and sentiment data first")
            return False
        
        print(f"🔗 Merging datasets using {merge_type} join...")
        
        # Rename date column in sentiment data to match price data
        sentiment_for_merge = self.sentiment_data.rename(columns={'date': 'Date'})
        
        # Merge datasets
        self.merged_data = pd.merge(
            self.price_data, 
            sentiment_for_merge, 
            on='Date', 
            how=merge_type,
            suffixes=('', '_sentiment')
        )
        
        print(f"✅ Merged dataset created with {len(self.merged_data)} rows")
        return True
    
    def handle_missing_sentiment(self, method='forward_fill'):
        """
        Handle missing sentiment data.
        
        Args:
            method: 'forward_fill', 'backward_fill', 'interpolate', 'drop', or 'neutral'
        """
        if self.merged_data is None:
            print("❌ Please merge datasets first")
            return False
        
        sentiment_cols = [col for col in self.merged_data.columns if 'sentiment' in col.lower() or 'compound' in col]
        
        missing_before = self.merged_data[sentiment_cols].isnull().sum().sum()
        if missing_before == 0:
            print("✅ No missing sentiment data found")
            return True
        
        print(f"🔧 Handling {missing_before} missing sentiment values using {method}...")
        
        if method == 'forward_fill':
            self.merged_data[sentiment_cols] = self.merged_data[sentiment_cols].fillna(method='ffill')
        elif method == 'backward_fill':
            self.merged_data[sentiment_cols] = self.merged_data[sentiment_cols].fillna(method='bfill')
        elif method == 'interpolate':
            self.merged_data[sentiment_cols] = self.merged_data[sentiment_cols].interpolate()
        elif method == 'neutral':
            # Fill with neutral sentiment values
            neutral_values = {
                'compound_mean': 0.0,
                'positive_mean': 0.25,
                'negative_mean': 0.25,
                'neutral_mean': 0.5,
                'sentiment_positive_ratio': 0.33,
                'sentiment_negative_ratio': 0.33,
                'sentiment_neutral_ratio': 0.34,
                'sentiment_volatility': 0.2,
                'total_tweets': 100,
                'weighted_sentiment': 0.0
            }
            for col in sentiment_cols:
                if col in neutral_values:
                    self.merged_data[col] = self.merged_data[col].fillna(neutral_values[col])
                else:
                    self.merged_data[col] = self.merged_data[col].fillna(0.0)
        elif method == 'drop':
            self.merged_data = self.merged_data.dropna(subset=sentiment_cols)
        
        missing_after = self.merged_data[sentiment_cols].isnull().sum().sum()
        print(f"✅ Missing sentiment values reduced from {missing_before} to {missing_after}")
        return True
    
    def create_lagged_sentiment_features(self, lags=[1, 3, 7]):
        """
        Create lagged sentiment features to capture sentiment momentum.
        
        Args:
            lags: List of lag periods to create
        """
        if self.merged_data is None:
            print("❌ Please merge datasets first")
            return False
        
        print(f"📊 Creating lagged sentiment features for lags: {lags}...")
        
        # Key sentiment columns to lag
        key_sentiment_cols = [
            'compound_mean', 'sentiment_positive_ratio', 'sentiment_negative_ratio',
            'sentiment_volatility', 'weighted_sentiment'
        ]
        
        # Create lagged features
        for lag in lags:
            for col in key_sentiment_cols:
                if col in self.merged_data.columns:
                    new_col_name = f"{col}_lag_{lag}d"
                    self.merged_data[new_col_name] = self.merged_data[col].shift(lag)
        
        # Create rolling sentiment features
        for window in [3, 7, 14]:
            if 'compound_mean' in self.merged_data.columns:
                self.merged_data[f'sentiment_ma_{window}d'] = self.merged_data['compound_mean'].rolling(window).mean()
                self.merged_data[f'sentiment_std_{window}d'] = self.merged_data['compound_mean'].rolling(window).std()
        
        # Sentiment momentum features
        if 'compound_mean' in self.merged_data.columns:
            self.merged_data['sentiment_momentum_3d'] = (
                self.merged_data['compound_mean'] - self.merged_data['sentiment_ma_3d']
            )
            self.merged_data['sentiment_momentum_7d'] = (
                self.merged_data['compound_mean'] - self.merged_data['sentiment_ma_7d']
            )
        
        print("✅ Lagged sentiment features created")
        return True
    
    def create_sentiment_price_interactions(self):
        """Create interaction features between sentiment and price data."""
        if self.merged_data is None:
            print("❌ Please merge datasets first")
            return False
        
        print("🔗 Creating sentiment-price interaction features...")
        
        # Sentiment * Volatility interactions
        if 'compound_mean' in self.merged_data.columns and 'VOLATILITY_14D' in self.merged_data.columns:
            self.merged_data['sentiment_volatility_interaction'] = (
                self.merged_data['compound_mean'] * self.merged_data['VOLATILITY_14D']
            )
        
        # Sentiment * RSI interactions
        if 'compound_mean' in self.merged_data.columns and 'RSI_14' in self.merged_data.columns:
            self.merged_data['sentiment_rsi_interaction'] = (
                self.merged_data['compound_mean'] * (self.merged_data['RSI_14'] - 50) / 50
            )
        
        # Sentiment * Return interactions
        if 'compound_mean' in self.merged_data.columns and 'RETURN_1D' in self.merged_data.columns:
            self.merged_data['sentiment_return_interaction'] = (
                self.merged_data['compound_mean'] * self.merged_data['RETURN_1D']
            )
        
        # Volume-weighted sentiment
        if 'compound_mean' in self.merged_data.columns and 'VOLUME_RATIO' in self.merged_data.columns:
            self.merged_data['volume_weighted_sentiment'] = (
                self.merged_data['compound_mean'] * self.merged_data['VOLUME_RATIO']
            )
        
        print("✅ Sentiment-price interaction features created")
        return True
    
    def validate_merged_data(self):
        """Validate the merged dataset."""
        if self.merged_data is None:
            print("❌ No merged data to validate")
            return False
        
        print("🔍 Validating merged dataset...")
        
        # Check for missing target variable
        if 'TARGET' not in self.merged_data.columns:
            print("❌ Target variable 'TARGET' not found")
            return False
        
        # Check date coverage
        date_range = (self.merged_data['Date'].max() - self.merged_data['Date'].min()).days
        print(f"📅 Date coverage: {date_range} days")
        
        # Check missing values
        missing_values = self.merged_data.isnull().sum()
        critical_missing = missing_values[missing_values > len(self.merged_data) * 0.1]  # >10% missing
        
        if len(critical_missing) > 0:
            print(f"⚠️ Columns with >10% missing values:")
            for col, missing in critical_missing.items():
                print(f"   {col}: {missing} ({missing/len(self.merged_data)*100:.1f}%)")
        
        # Check data types
        numeric_cols = self.merged_data.select_dtypes(include=[np.number]).columns
        print(f"📊 Numeric columns: {len(numeric_cols)}")
        
        # Check target distribution
        target_dist = self.merged_data['TARGET'].value_counts()
        print(f"🎯 Target distribution:")
        for idx, count in target_dist.items():
            label = "Down" if idx == 0 else "Up"
            pct = count / len(self.merged_data) * 100
            print(f"   {label}: {count} ({pct:.1f}%)")
        
        print("✅ Dataset validation complete")
        return True
    
    def save_merged_data(self, filepath='data/merged_features.csv'):
        """Save the merged dataset."""
        if self.merged_data is None:
            print("❌ No merged data to save")
            return False
        
        try:
            self.merged_data.to_csv(filepath, index=False)
            print(f"✅ Merged dataset saved to: {filepath}")
            
            # Save feature list
            feature_columns = [col for col in self.merged_data.columns 
                             if col not in ['Date', 'TARGET', 'TARGET_2D', 'TARGET_STRONG']]
            
            feature_list = pd.DataFrame({
                'Feature': feature_columns,
                'Type': ['numeric'] * len(feature_columns)
            })
            feature_list.to_csv('data/final_feature_list.csv', index=False)
            print(f"✅ Feature list saved to: data/final_feature_list.csv")
            
            return True
        except Exception as e:
            print(f"❌ Error saving merged data: {str(e)}")
            return False
    
    def run_merger_pipeline(self, 
                           price_file='data/btc_features_enhanced.csv',
                           sentiment_file='data/sentiment_features.csv',
                           output_file='data/merged_features.csv'):
        """
        Run the complete data merger pipeline.
        
        Args:
            price_file: Path to price features file
            sentiment_file: Path to sentiment features file
            output_file: Path for output merged file
        """
        print("🚀 Starting data merger pipeline...")
        
        # Load datasets
        if not self.load_price_features(price_file):
            return False
        
        if not self.load_sentiment_features(sentiment_file):
            return False
        
        # Merge datasets
        if not self.merge_datasets(merge_type='left'):
            return False
        
        # Handle missing sentiment data
        if not self.handle_missing_sentiment(method='neutral'):
            return False
        
        # Create additional features
        if not self.create_lagged_sentiment_features():
            return False
        
        if not self.create_sentiment_price_interactions():
            return False
        
        # Validate and save
        if not self.validate_merged_data():
            return False
        
        if not self.save_merged_data(output_file):
            return False
        
        print(f"🎉 Data merger pipeline complete!")
        print(f"Final dataset shape: {self.merged_data.shape}")
        return self.merged_data

if __name__ == "__main__":
    # Create data merger and run pipeline
    merger = DataMerger()
    merged_data = merger.run_merger_pipeline()
    
    if merged_data is not None:
        print(f"\n📊 MERGED DATASET SUMMARY:")
        print(f"Shape: {merged_data.shape}")
        print(f"Date range: {merged_data['Date'].min()} to {merged_data['Date'].max()}")
        print(f"Features: {len([col for col in merged_data.columns if col not in ['Date', 'TARGET']])}")
        print(f"Target balance: {merged_data['TARGET'].mean():.1%} up days")
    else:
        print("❌ Data merger pipeline failed")
