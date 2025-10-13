#!/usr/bin/env python3
"""
Bitcoin Data Update Script
Updates the local Bitcoin dataset with the latest price data from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_bitcoin_data():
    """Update Bitcoin data with latest prices from Yahoo Finance."""
    try:
        # Data file path
        data_file = 'data/btc_data.csv'
        
        # Check if we have existing data
        if os.path.exists(data_file):
            logger.info("📊 Loading existing Bitcoin data...")
            existing_data = pd.read_csv(data_file)
            existing_data['Date'] = pd.to_datetime(existing_data['Date'])
            
            # Get the last date in our data
            last_date = existing_data['Date'].max()
            start_date = last_date + timedelta(days=1)
            
            logger.info(f"📅 Last date in data: {last_date.strftime('%Y-%m-%d')}")
            logger.info(f"🔄 Fetching data from: {start_date.strftime('%Y-%m-%d')}")
        else:
            logger.info("📊 No existing data found, downloading full dataset...")
            start_date = datetime(2020, 1, 1)
            existing_data = pd.DataFrame()
        
        # Current date
        end_date = datetime.now()
        
        # Only fetch if we need new data
        if start_date.date() >= end_date.date():
            logger.info("✅ Data is already up to date!")
            if not existing_data.empty:
                latest_price = existing_data['Close'].iloc[-1]
                latest_date = existing_data['Date'].iloc[-1].strftime('%Y-%m-%d')
                logger.info(f"💰 Current Bitcoin Price: ${latest_price:,.2f} ({latest_date})")
            return True
        
        # Fetch new data from Yahoo Finance
        logger.info("🌐 Fetching data from Yahoo Finance...")
        btc_ticker = yf.Ticker("BTC-USD")
        
        # Get historical data
        new_data = btc_ticker.history(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1d'
        )
        
        if new_data.empty:
            logger.info("ℹ️ No new data available")
            return True
        
        # Clean the new data
        new_data = new_data.reset_index()
        new_data['Date'] = new_data['Date'].dt.tz_localize(None)  # Remove timezone
        
        # Select relevant columns
        columns_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        new_data = new_data[columns_to_keep]
        
        # Combine with existing data
        if not existing_data.empty:
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            combined_data = new_data
        
        # Remove duplicates and sort
        combined_data = combined_data.drop_duplicates(subset=['Date']).sort_values('Date')
        combined_data = combined_data.reset_index(drop=True)
        
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Save updated data
        combined_data.to_csv(data_file, index=False)
        
        # Log success
        logger.info(f"✅ Successfully updated Bitcoin data!")
        logger.info(f"📊 Total records: {len(combined_data)}")
        logger.info(f"📅 Date range: {combined_data['Date'].min().strftime('%Y-%m-%d')} to {combined_data['Date'].max().strftime('%Y-%m-%d')}")
        
        # Show current price
        latest_price = combined_data['Close'].iloc[-1]
        latest_date = combined_data['Date'].iloc[-1].strftime('%Y-%m-%d')
        logger.info(f"💰 Current Bitcoin Price: ${latest_price:,.2f} ({latest_date})")
        
        # Show recent changes
        if len(combined_data) >= 2:
            prev_price = combined_data['Close'].iloc[-2]
            price_change = ((latest_price - prev_price) / prev_price) * 100
            direction = "📈" if price_change > 0 else "📉"
            logger.info(f"{direction} Price change: {price_change:+.2f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error updating Bitcoin data: {str(e)}")
        return False

def validate_data_quality():
    """Validate the quality of the Bitcoin data."""
    try:
        data_file = 'data/btc_data.csv'
        
        if not os.path.exists(data_file):
            logger.error("❌ Bitcoin data file not found!")
            return False
        
        # Load and validate data
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'])
        
        logger.info("🔍 Validating data quality...")
        
        # Check for required columns
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"❌ Missing columns: {missing_columns}")
            return False
        
        # Check for missing values
        missing_values = df[required_columns].isnull().sum()
        if missing_values.any():
            logger.warning(f"⚠️ Missing values found: {missing_values.to_dict()}")
        
        # Check for duplicate dates
        duplicate_dates = df['Date'].duplicated().sum()
        if duplicate_dates > 0:
            logger.warning(f"⚠️ Found {duplicate_dates} duplicate dates")
        
        # Check data consistency (High >= Low, etc.)
        consistency_issues = 0
        
        if (df['High'] < df['Low']).any():
            consistency_issues += 1
            logger.warning("⚠️ Found records where High < Low")
        
        if (df['High'] < df['Close']).any():
            inconsistent_high = (df['High'] < df['Close']).sum()
            logger.warning(f"⚠️ Found {inconsistent_high} records where High < Close")
        
        if (df['Low'] > df['Close']).any():
            inconsistent_low = (df['Low'] > df['Close']).sum()
            logger.warning(f"⚠️ Found {inconsistent_low} records where Low > Close")
        
        # Check for reasonable price ranges
        min_price = df['Close'].min()
        max_price = df['Close'].max()
        
        if min_price < 100 or max_price > 1000000:
            logger.warning(f"⚠️ Unusual price range: ${min_price:,.2f} - ${max_price:,.2f}")
        
        # Summary
        logger.info(f"✅ Data validation completed")
        logger.info(f"📊 Records: {len(df)}")
        logger.info(f"📅 Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        logger.info(f"💰 Price range: ${min_price:,.2f} - ${max_price:,.2f}")
        
        if consistency_issues == 0 and missing_values.sum() == 0:
            logger.info("✅ Data quality: EXCELLENT")
        else:
            logger.info("⚠️ Data quality: GOOD (minor issues found)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validating data: {str(e)}")
        return False

def main():
    """Main function to update and validate Bitcoin data."""
    logger.info("🚀 Starting Bitcoin data update process...")
    
    # Update data
    if update_bitcoin_data():
        logger.info("✅ Data update completed successfully")
        
        # Validate data quality
        if validate_data_quality():
            logger.info("✅ Data validation completed successfully")
        else:
            logger.error("❌ Data validation failed")
            return False
    else:
        logger.error("❌ Data update failed")
        return False
    
    logger.info("🎉 Bitcoin data update process completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
