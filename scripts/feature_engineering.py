# scripts/feature_engineering.py

import pandas as pd
import numpy as np
import os
import yfinance as yf
import ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Comprehensive feature engineering for Bitcoin price prediction.
    Creates technical indicators, price features, time features, and market regime indicators.
    """
    
    def __init__(self, start_date='2020-01-01', end_date=None):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
    def download_bitcoin_data(self):
        """Download Bitcoin price data from Yahoo Finance with retry logic"""
        print("📥 Downloading Bitcoin data...")
        
        # Try different approaches if SSL fails
        methods = [
            lambda: yf.download("BTC-USD", start=self.start_date, end=self.end_date, progress=False),
            lambda: yf.Ticker("BTC-USD").history(start=self.start_date, end=self.end_date),
        ]
        
        for i, method in enumerate(methods):
            try:
                df = method()
                
                if df.empty:
                    continue
                
                # Reset index to get Date as column
                df.reset_index(inplace=True)
                
                # Fix MultiIndex columns (flatten them)
                if isinstance(df.columns, pd.MultiIndex):
                    # Flatten the MultiIndex columns
                    df.columns = [col[0] if col[1] == '' else f"{col[0]}" for col in df.columns.values]
                
                # Ensure we have minimum required data (60 days for technical indicators)
                if len(df) < 60:
                    print(f"⚠️ Only {len(df)} days of data available. Need at least 60 days for reliable technical indicators.")
                    # Extend start date to get more data
                    extended_start = (datetime.strptime(self.start_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
                    print(f"📥 Trying to download more data from {extended_start}...")
                    
                    if i == 0:
                        df = yf.download("BTC-USD", start=extended_start, end=self.end_date, progress=False)
                    else:
                        df = yf.Ticker("BTC-USD").history(start=extended_start, end=self.end_date)
                    
                    if not df.empty:
                        df.reset_index(inplace=True)
                        
                        # Fix MultiIndex columns (flatten them)
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = [col[0] if col[1] == '' else f"{col[0]}" for col in df.columns.values]
                
                if len(df) >= 60:
                    print(f"✅ Downloaded {len(df)} days of Bitcoin data")
                    return df
                    
            except Exception as e:
                print(f"⚠️ Method {i+1} failed: {e}")
                continue
        
        # If all methods fail, create dummy data for testing
        print("❌ Could not download real data. Creating sample data for testing...")
        return self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample Bitcoin data for testing when download fails"""
        print("📊 Creating sample Bitcoin data...")
        
        # Generate 500 days of sample data
        start_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
        dates = pd.date_range(start=start_dt, periods=500, freq='D')
        
        # Generate realistic Bitcoin price data
        np.random.seed(42)  # For reproducibility
        base_price = 30000
        returns = np.random.normal(0.001, 0.03, len(dates))  # Mean daily return 0.1%, volatility 3%
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create realistic OHLC data
        df = pd.DataFrame({
            'Date': dates,
            'Open': [p * np.random.uniform(0.99, 1.01) for p in prices],
            'High': [p * np.random.uniform(1.00, 1.05) for p in prices],
            'Low': [p * np.random.uniform(0.95, 1.00) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        })
        
        # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
        df['High'] = np.maximum(df['High'], np.maximum(df['Open'], df['Close']))
        df['Low'] = np.minimum(df['Low'], np.minimum(df['Open'], df['Close']))
        
        print(f"✅ Created {len(df)} days of sample Bitcoin data")
        return df
    
    def create_technical_indicators(self, df):
        """Create comprehensive technical indicators with safety checks."""
        print("🔧 Creating technical indicators...")
        
        # Check data length
        if len(df) < 60:
            print(f"⚠️ Warning: Only {len(df)} days of data. Some indicators may not be reliable.")
            
        # Prepare data
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        volume = df['Volume'].squeeze()
        
        # Only create indicators if we have enough data
        min_periods = min(len(df) - 1, 14)  # Use smaller window if insufficient data
        
        try:
            # Momentum Indicators
            df['RSI_14'] = ta.momentum.RSIIndicator(close=close, window=min(14, min_periods)).rsi()
            df['RSI_7'] = ta.momentum.RSIIndicator(close=close, window=min(7, min_periods)).rsi()
            df['RSI_21'] = ta.momentum.RSIIndicator(close=close, window=min(21, len(df)//2)).rsi()
            
            if len(df) >= 14:
                stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=min_periods)
                df['STOCH_K'] = stoch.stoch()
                df['STOCH_D'] = stoch.stoch_signal()
                
                df['WILLIAMS_R'] = ta.momentum.WilliamsRIndicator(high=high, low=low, close=close, lbp=min_periods).williams_r()
                
            df['ROC_10'] = ta.momentum.ROCIndicator(close=close, window=min(10, min_periods)).roc()
            df['ROC_20'] = ta.momentum.ROCIndicator(close=close, window=min(20, len(df)//2)).roc()
            df['CCI'] = ta.trend.CCIIndicator(high=high, low=low, close=close, window=min(20, len(df)//2)).cci()
            
            # Trend Indicators
            df['SMA_5'] = ta.trend.SMAIndicator(close=close, window=min(5, min_periods)).sma_indicator()
            df['SMA_10'] = ta.trend.SMAIndicator(close=close, window=min(10, min_periods)).sma_indicator()
            df['SMA_20'] = ta.trend.SMAIndicator(close=close, window=min(20, len(df)//2)).sma_indicator()
            df['SMA_50'] = ta.trend.SMAIndicator(close=close, window=min(50, len(df)//2)).sma_indicator()
            df['SMA_100'] = ta.trend.SMAIndicator(close=close, window=min(100, len(df)//2)).sma_indicator()
            df['SMA_200'] = ta.trend.SMAIndicator(close=close, window=min(200, len(df)//2)).sma_indicator()
            
            df['EMA_12'] = ta.trend.EMAIndicator(close=close, window=min(12, min_periods)).ema_indicator()
            df['EMA_26'] = ta.trend.EMAIndicator(close=close, window=min(26, len(df)//2)).ema_indicator()
            df['EMA_50'] = ta.trend.EMAIndicator(close=close, window=min(50, len(df)//2)).ema_indicator()
            
            # MACD
            if len(df) >= 26:
                macd = ta.trend.MACD(close=close, window_slow=min(26, len(df)//2), window_fast=min(12, min_periods), window_sign=min(9, min_periods))
                df['MACD'] = macd.macd()
                df['MACD_SIGNAL'] = macd.macd_signal()
                df['MACD_HISTOGRAM'] = macd.macd_diff()
            
            # ADX - only if we have enough data
            if len(df) >= 14:
                adx = ta.trend.ADXIndicator(high=high, low=low, close=close, window=min_periods)
                df['ADX'] = adx.adx()
                df['ADX_POS'] = adx.adx_pos()
                df['ADX_NEG'] = adx.adx_neg()
            
            # Volatility Indicators
            if len(df) >= 20:
                bb = ta.volatility.BollingerBands(close=close, window=min(20, len(df)//2), window_dev=2)
                df['BB_HIGH'] = bb.bollinger_hband()
                df['BB_LOW'] = bb.bollinger_lband()
                df['BB_MID'] = bb.bollinger_mavg()
                df['BB_WIDTH'] = bb.bollinger_wband()
                df['BB_PERCENT'] = bb.bollinger_pband()
            
            if len(df) >= 14:
                df['ATR'] = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=min_periods).average_true_range()
            
        except Exception as e:
            print(f"⚠️ Warning: Some technical indicators failed: {e}")
            # Fill any missing indicators with 0
            indicator_cols = ['RSI_14', 'RSI_7', 'RSI_21', 'STOCH_K', 'STOCH_D', 'WILLIAMS_R', 
                            'ROC_10', 'ROC_20', 'CCI', 'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 
                            'SMA_100', 'SMA_200', 'EMA_12', 'EMA_26', 'EMA_50', 'MACD', 
                            'MACD_SIGNAL', 'MACD_HISTOGRAM', 'ADX', 'ADX_POS', 'ADX_NEG',
                            'BB_HIGH', 'BB_LOW', 'BB_MID', 'BB_WIDTH', 'BB_PERCENT', 'ATR']
            
            for col in indicator_cols:
                if col not in df.columns:
                    df[col] = 0
        
        print("✅ Technical indicators created")
        return df
    
    def create_price_features(self, df):
        """Create price-based features."""
        print("💰 Creating price-based features...")
        
        # Returns
        df['RETURN_1D'] = df['Close'].pct_change()
        df['RETURN_3D'] = df['Close'].pct_change(periods=3)
        df['RETURN_7D'] = df['Close'].pct_change(periods=7)
        df['RETURN_14D'] = df['Close'].pct_change(periods=14)
        df['RETURN_30D'] = df['Close'].pct_change(periods=30)
        
        # Log returns
        df['LOG_RETURN_1D'] = np.log(df['Close'] / df['Close'].shift(1))
        df['LOG_RETURN_7D'] = np.log(df['Close'] / df['Close'].shift(7))
        
        # Volatility
        df['VOLATILITY_7D'] = df['RETURN_1D'].rolling(window=7).std()
        df['VOLATILITY_14D'] = df['RETURN_1D'].rolling(window=14).std()
        df['VOLATILITY_30D'] = df['RETURN_1D'].rolling(window=30).std()
        
        # Price patterns (with safety checks)
        df['PRICE_POSITION_14D'] = (df['Close'] - df['Low'].rolling(14).min()) / (df['High'].rolling(14).max() - df['Low'].rolling(14).min())
        
        # Debug: Check column types
        print(f"DataFrame columns: {df.columns.tolist()}")
        
        # Distance from moving averages (with safety checks)
        if 'SMA_20' in df.columns:
            try:
                sma_20 = df['SMA_20']
                if hasattr(sma_20, 'iloc'):  # Check if it's a series
                    close_vals = df['Close']
                    dist_vals = (close_vals - sma_20) / sma_20
                    df['DIST_SMA_20'] = dist_vals.fillna(0)
                else:
                    df['DIST_SMA_20'] = 0
            except Exception as e:
                print(f"Error with SMA_20: {e}")
                df['DIST_SMA_20'] = 0
        else:
            df['DIST_SMA_20'] = 0
            
        if 'SMA_50' in df.columns:
            try:
                sma_50 = df['SMA_50']
                if hasattr(sma_50, 'iloc'):  # Check if it's a series
                    close_vals = df['Close']
                    dist_vals = (close_vals - sma_50) / sma_50
                    df['DIST_SMA_50'] = dist_vals.fillna(0)
                else:
                    df['DIST_SMA_50'] = 0
            except Exception as e:
                print(f"Error with SMA_50: {e}")
                df['DIST_SMA_50'] = 0
        else:
            df['DIST_SMA_50'] = 0
            
        if 'SMA_200' in df.columns:
            try:
                sma_200 = df['SMA_200']
                if hasattr(sma_200, 'iloc'):  # Check if it's a series
                    close_vals = df['Close']
                    dist_vals = (close_vals - sma_200) / sma_200
                    df['DIST_SMA_200'] = dist_vals.fillna(0)
                else:
                    df['DIST_SMA_200'] = 0
            except Exception as e:
                print(f"Error with SMA_200: {e}")
                df['DIST_SMA_200'] = 0
        else:
            df['DIST_SMA_200'] = 0
        
        # Moving average crossovers (with safety checks)
        try:
            if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
                df['SMA_CROSS_20_50'] = (df['SMA_20'] > df['SMA_50']).astype(int)
            else:
                df['SMA_CROSS_20_50'] = 0
                
            if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
                df['SMA_CROSS_50_200'] = (df['SMA_50'] > df['SMA_200']).astype(int)
            else:
                df['SMA_CROSS_50_200'] = 0
                
            if 'EMA_12' in df.columns and 'EMA_26' in df.columns:
                df['EMA_CROSS_12_26'] = (df['EMA_12'] > df['EMA_26']).astype(int)
            else:
                df['EMA_CROSS_12_26'] = 0
        except Exception as e:
            print(f"Error with crossovers: {e}")
            df['SMA_CROSS_20_50'] = 0
            df['SMA_CROSS_50_200'] = 0
            df['EMA_CROSS_12_26'] = 0
        
        # Volume features
        df['VOLUME_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['VOLUME_RATIO'] = df['Volume'] / df['VOLUME_SMA_20']
        
        # Candlestick patterns
        df['BODY_SIZE'] = abs(df['Close'] - df['Open']) / df['Close']
        df['UPPER_WICK'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['Close']
        df['LOWER_WICK'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Close']
        df['BULLISH_CANDLE'] = (df['Close'] > df['Open']).astype(int)
        
        print("✅ Price-based features created")
        return df
    
    def create_time_features(self, df):
        """Create time-based features."""
        print("📅 Creating time-based features...")
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Basic time features
        df['YEAR'] = df['Date'].dt.year
        df['MONTH'] = df['Date'].dt.month
        df['DAY'] = df['Date'].dt.day
        df['DAY_OF_WEEK'] = df['Date'].dt.dayofweek
        df['QUARTER'] = df['Date'].dt.quarter
        
        # Weekend indicators
        df['IS_WEEKEND'] = (df['DAY_OF_WEEK'] >= 5).astype(int)
        df['IS_MONDAY'] = (df['DAY_OF_WEEK'] == 0).astype(int)
        df['IS_FRIDAY'] = (df['DAY_OF_WEEK'] == 4).astype(int)
        
        # Cyclical encoding
        df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH'] / 12)
        df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH'] / 12)
        df['DAY_OF_WEEK_SIN'] = np.sin(2 * np.pi * df['DAY_OF_WEEK'] / 7)
        df['DAY_OF_WEEK_COS'] = np.cos(2 * np.pi * df['DAY_OF_WEEK'] / 7)
        
        print("✅ Time-based features created")
        return df
    
    def create_market_regime_features(self, df):
        """Create market regime indicators."""
        print("📈 Creating market regime indicators...")
        
        # Trend strength
        df['BULL_MARKET_20'] = (df['Close'] > df['SMA_20']).astype(int)
        df['BULL_MARKET_50'] = (df['Close'] > df['SMA_50']).astype(int)
        df['BULL_MARKET_200'] = (df['Close'] > df['SMA_200']).astype(int)
        
        # Strong trends
        df['STRONG_UPTREND'] = ((df['SMA_20'] > df['SMA_50']) & 
                               (df['SMA_50'] > df['SMA_200']) & 
                               (df['Close'] > df['SMA_20'])).astype(int)
        
        df['STRONG_DOWNTREND'] = ((df['SMA_20'] < df['SMA_50']) & 
                                 (df['SMA_50'] < df['SMA_200']) & 
                                 (df['Close'] < df['SMA_20'])).astype(int)
        
        # Volatility regime
        df['HIGH_VOLATILITY'] = (df['VOLATILITY_14D'] > df['VOLATILITY_14D'].rolling(50).quantile(0.75)).astype(int)
        df['LOW_VOLATILITY'] = (df['VOLATILITY_14D'] < df['VOLATILITY_14D'].rolling(50).quantile(0.25)).astype(int)
        
        # Momentum regime
        df['MOMENTUM_REGIME'] = np.where(df['RSI_14'] > 70, 2,  # Overbought
                                        np.where(df['RSI_14'] < 30, 0, 1))  # Oversold, Normal
        
        print("✅ Market regime indicators created")
        return df
    
    def create_target_variable(self, df):
        """Create target variable for prediction."""
        print("🎯 Creating target variable...")
        
        # Primary target: 1 if next day's price is higher, 0 otherwise
        df['TARGET'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Additional target variations
        df['TARGET_2D'] = (df['Close'].shift(-2) > df['Close']).astype(int)
        df['TARGET_STRONG'] = (df['Close'].shift(-1) / df['Close'] > 1.02).astype(int)  # >2% gain
        
        print("✅ Target variable created")
        return df
    
    def clean_data(self, df):
        """Clean and validate the dataset."""
        print("🧹 Cleaning data...")
        
        # Replace infinite values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Remove rows with NaN target
        df = df.dropna(subset=['TARGET'])
        
        # Forward fill some indicators
        fill_columns = ['RSI_14', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26']
        for col in fill_columns:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')
        
        # Drop remaining NaN values
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        print(f"✅ Data cleaned. Final shape: {df.shape}")
        return df
    
    def generate_features(self, save_to_file=True):
        """Main method to generate all features."""
        print("🚀 Starting comprehensive feature engineering...")
        
        # Download data
        df = self.download_bitcoin_data()
        
        # Create all features
        df = self.create_technical_indicators(df)
        df = self.create_price_features(df)
        df = self.create_time_features(df)
        df = self.create_market_regime_features(df)
        df = self.create_target_variable(df)
        
        # Clean data
        df = self.clean_data(df)
        
        if save_to_file:
            # Save enhanced dataset
            output_file = 'data/btc_features_enhanced.csv'
            df.to_csv(output_file, index=False)
            print(f"✅ Enhanced dataset saved to: {output_file}")
            
            # Save feature metadata
            feature_columns = [col for col in df.columns if col not in ['Date', 'TARGET', 'TARGET_2D', 'TARGET_STRONG']]
            feature_info = pd.DataFrame({
                'Feature': feature_columns,
                'Type': ['numeric'] * len(feature_columns)
            })
            feature_info.to_csv('data/feature_info.csv', index=False)
            print(f"✅ Feature info saved to: data/feature_info.csv")
        
        print(f"🎉 Feature engineering complete! Dataset shape: {df.shape}")
        return df

if __name__ == "__main__":
    # Create feature engineer and generate features
    engineer = FeatureEngineer()
    enhanced_data = engineer.generate_features()
    
    # Print summary
    print(f"\n📊 SUMMARY:")
    print(f"Total features: {len([col for col in enhanced_data.columns if col not in ['Date']])}")
    print(f"Target distribution: {enhanced_data['TARGET'].mean():.1%} up days")
    print(f"Date range: {enhanced_data['Date'].min()} to {enhanced_data['Date'].max()}")
