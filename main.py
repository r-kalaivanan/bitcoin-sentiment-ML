# main.py - Complete Bitcoin Sentiment ML Pipeline

"""
Bitcoin Sentiment ML - Complete Pipeline

This is the main entry point for the Bitcoin price prediction system.
It orchestrates the entire pipeline from data collection to prediction.

Usage:
    python main.py --mode [pipeline|train|predict]
    
    pipeline: Run complete pipeline (data collection → training → prediction)
    train: Train models only
    predict: Make prediction only
"""

import argparse
import os
import sys
from datetime import datetime

# Add scripts directory to path
sys.path.append('scripts')

from feature_engineering import FeatureEngineer
from sentiment_analysis import SentimentAnalyzer
from data_merger import DataMerger
from model_trainer import ModelTrainer
from predictor import BitcoinPredictor

def setup_directories():
    """Create necessary directories for the project."""
    directories = ['data', 'models', 'plots', 'predictions']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Project directories ready")

def run_data_pipeline():
    """Run the complete data collection and preprocessing pipeline."""
    print("🚀 Starting data pipeline...")
    
    # Step 1: Feature Engineering
    print("\n" + "="*50)
    print("STEP 1: FEATURE ENGINEERING")
    print("="*50)
    engineer = FeatureEngineer()
    price_data = engineer.generate_features()
    
    if price_data is None or len(price_data) == 0:
        print("❌ Feature engineering failed")
        return False
    
    # Step 2: Sentiment Analysis
    print("\n" + "="*50)
    print("STEP 2: SENTIMENT ANALYSIS")
    print("="*50)
    analyzer = SentimentAnalyzer()
    sentiment_data = analyzer.run_sentiment_pipeline(historical=True, scrape_recent=True)
    
    # Step 3: Data Merging
    print("\n" + "="*50)
    print("STEP 3: DATA MERGING")
    print("="*50)
    merger = DataMerger()
    merged_data = merger.run_merger_pipeline()
    
    if merged_data is None or len(merged_data) == 0:
        print("❌ Data pipeline failed")
        return False
    
    print("✅ Data pipeline completed successfully")
    return True

def run_training_pipeline():
    """Run the model training pipeline."""
    print("\n" + "="*50)
    print("STEP 4: MODEL TRAINING")
    print("="*50)
    
    trainer = ModelTrainer()
    results = trainer.run_training_pipeline()
    
    if results is None:
        print("❌ Model training failed")
        return False
    
    print("✅ Model training completed successfully")
    return True

def run_prediction():
    """Make a prediction using the trained model with real Twitter integration."""
    print("\n" + "="*50)
    print("STEP 5: PREDICTION")
    print("="*50)
    
    # Check if Twitter API is available for real-time sentiment
    try:
        from scripts.scrape import TwitterScraper
        scraper = TwitterScraper()
        
        if scraper.check_api_status():
            print("🐦 Twitter API available - will use REAL sentiment data")
        else:
            print("⚠️ Twitter API not available - will use simulated sentiment data")
            print("💡 To use real data, set X_BEARER_TOKEN in .env file")
    except Exception as e:
        print(f"⚠️ Twitter API check failed: {e}")
        print("🔄 Will use simulated sentiment data")
    
    predictor = BitcoinPredictor()
    result = predictor.run_prediction_pipeline()
    
    if result is None:
        print("❌ Prediction failed")
        return False
    
    print("✅ Prediction completed successfully")
    return True

def main():
    """Main function to run the Bitcoin sentiment ML pipeline."""
    parser = argparse.ArgumentParser(description='Bitcoin Sentiment ML Pipeline')
    parser.add_argument('--mode', choices=['pipeline', 'train', 'predict'], 
                       default='pipeline', help='Mode to run')
    parser.add_argument('--skip-data', action='store_true', 
                       help='Skip data collection (use existing data)')
    
    args = parser.parse_args()
    
    print("🪙 BITCOIN SENTIMENT ML PIPELINE")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Setup project structure
    setup_directories()
    
    success = True
    
    if args.mode == 'pipeline':
        # Run complete pipeline
        if not args.skip_data:
            success = success and run_data_pipeline()
        
        if success:
            success = success and run_training_pipeline()
        
        if success:
            success = success and run_prediction()
    
    elif args.mode == 'train':
        # Run training only
        success = run_training_pipeline()
    
    elif args.mode == 'predict':
        # Run prediction only
        success = run_prediction()
    
    # Final status
    print("\n" + "="*50)
    if success:
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("📊 Check the following for results:")
        print("   • data/ - Processed datasets")
        print("   • models/ - Trained models and results")
        print("   • plots/ - Performance visualizations")
        print("   • predictions/ - Latest predictions")
    else:
        print("❌ PIPELINE FAILED!")
        print("Please check the error messages above")
    print("="*50)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
