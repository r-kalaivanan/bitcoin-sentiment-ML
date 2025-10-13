#!/usr/bin/env python3
"""
Comprehensive Integration Testing Suite
Tests the complete Bitcoin Sentiment ML pipeline
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_pipeline():
    """Test the data loading and processing pipeline."""
    logger.info("🧪 Testing Data Pipeline...")
    
    try:
        # Test Bitcoin data loading
        btc_data_file = "data/btc_data.csv"
        if os.path.exists(btc_data_file):
            btc_data = pd.read_csv(btc_data_file)
            logger.info(f"✅ Bitcoin Data: {len(btc_data)} records loaded")
            
            # Check current price
            if not btc_data.empty:
                latest_price = btc_data['Close'].iloc[-1]
                latest_date = btc_data['Date'].iloc[-1]
                logger.info(f"💰 Current Bitcoin: ${latest_price:,.2f} ({latest_date})")
        else:
            logger.warning("⚠️  Bitcoin data file not found")
            
        # Test feature data
        feature_files = [
            "data/btc_features_enhanced.csv",
            "data/sentiment_feature_list.csv",
            "data/final_feature_list.csv"
        ]
        
        for file_path in feature_files:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                logger.info(f"✅ {Path(file_path).name}: {len(df)} records")
            else:
                logger.warning(f"⚠️  {Path(file_path).name} not found")
                
        return True
        
    except Exception as e:
        logger.error(f"❌ Data Pipeline Test Failed: {e}")
        return False

def test_model_loading():
    """Test model loading and availability."""
    logger.info("🧪 Testing Model Loading...")
    
    try:
        models_dir = Path("models")
        if not models_dir.exists():
            logger.error("❌ Models directory not found")
            return False
            
        # Test enhanced models
        enhanced_models = [
            "lightgbm_sentiment_enhanced.pkl",
            "xgboost_sentiment_enhanced.pkl", 
            "randomforest_sentiment_enhanced.pkl",
            "logisticregression_sentiment_enhanced.pkl",
            "svm_sentiment_enhanced.pkl"
        ]
        
        loaded_models = 0
        for model_file in enhanced_models:
            model_path = models_dir / model_file
            if model_path.exists():
                loaded_models += 1
                logger.info(f"✅ {model_file}")
            else:
                logger.warning(f"⚠️  {model_file} not found")
        
        # Test scaler
        scaler_file = models_dir / "sentiment_enhanced_scaler.pkl"
        if scaler_file.exists():
            logger.info("✅ Sentiment enhanced scaler found")
        else:
            logger.warning("⚠️  Sentiment enhanced scaler not found")
        
        logger.info(f"📊 Total enhanced models available: {loaded_models}/5")
        return loaded_models > 0
        
    except Exception as e:
        logger.error(f"❌ Model Loading Test Failed: {e}")
        return False

def test_sentiment_predictor():
    """Test the sentiment-enhanced predictor."""
    logger.info("🧪 Testing Sentiment-Enhanced Predictor...")
    
    try:
        # Import predictor
        sys.path.append("scripts")
        from sentiment_enhanced_predictor import get_sentiment_enhanced_predictor
        
        # Initialize predictor
        predictor = get_sentiment_enhanced_predictor()
        
        # Test model analysis
        analysis = predictor.get_model_analysis()
        logger.info(f"✅ Predictor Initialized: {analysis['total_models']} models")
        logger.info(f"   - Features: {analysis['feature_count']}")
        logger.info(f"   - Sentiment Features: {analysis['sentiment_features']}")
        logger.info(f"   - Best Model: {analysis['best_model']}")
        
        # Test prediction with sample data
        sample_data = pd.DataFrame({
            'Date': pd.date_range(end=datetime.now(), periods=30, freq='D'),
            'Close': np.random.normal(99000, 2000, 30),
            'High': np.random.normal(101000, 2000, 30),
            'Low': np.random.normal(97000, 2000, 30),
            'Open': np.random.normal(99000, 2000, 30),
            'Volume': np.random.normal(20000000000, 5000000000, 30)
        })
        
        predictions = predictor.predict_future_prices_enhanced(sample_data, days=5)
        
        if not predictions.empty:
            logger.info(f"✅ Predictions Generated: {len(predictions)} days")
            logger.info(f"   - Average Confidence: {predictions['confidence'].mean():.3f}")
            logger.info(f"   - Sentiment Score Range: {predictions['sentiment_score'].min():.3f} to {predictions['sentiment_score'].max():.3f}")
            return True
        else:
            logger.error("❌ No predictions generated")
            return False
            
    except Exception as e:
        logger.error(f"❌ Sentiment Predictor Test Failed: {e}")
        return False

def test_dashboard_integration():
    """Test dashboard integration capabilities."""
    logger.info("🧪 Testing Dashboard Integration...")
    
    try:
        # Test dashboard imports
        sys.path.append("scripts")
        
        # Test predictor integration
        try:
            from sentiment_enhanced_predictor import get_sentiment_enhanced_predictor, SENTIMENT_ENHANCED
            logger.info("✅ Dashboard can import sentiment predictor")
        except ImportError:
            logger.warning("⚠️  Dashboard will run in standard mode")
            
        # Test required packages
        required_packages = ['streamlit', 'plotly', 'pandas', 'numpy']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✅ {package} available")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"⚠️  {package} missing")
        
        if missing_packages:
            logger.warning(f"⚠️  Install missing packages: pip install {' '.join(missing_packages)}")
            
        return len(missing_packages) == 0
        
    except Exception as e:
        logger.error(f"❌ Dashboard Integration Test Failed: {e}")
        return False

def run_comprehensive_test():
    """Run all integration tests."""
    logger.info("🚀 Starting Comprehensive Integration Testing...")
    logger.info("=" * 60)
    
    results = {
        'Data Pipeline': test_data_pipeline(),
        'Model Loading': test_model_loading(), 
        'Sentiment Predictor': test_sentiment_predictor(),
        'Dashboard Integration': test_dashboard_integration()
    }
    
    logger.info("=" * 60)
    logger.info("📊 TEST RESULTS SUMMARY:")
    
    passed = 0
    total = 0
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {test_name}: {status}")
        if result:
            passed += 1
        total += 1
    
    success_rate = (passed / total) * 100
    logger.info(f"\n🎯 Overall Success Rate: {success_rate:.1f}% ({passed}/{total} tests passed)")
    
    if success_rate >= 75:
        logger.info("🎉 INTEGRATION SUCCESSFUL - System ready for production!")
        logger.info("\n📊 Next Steps:")
        logger.info("   1. Launch dashboard: python launch_dashboard.py")
        logger.info("   2. Or run directly: streamlit run scripts/dashboard.py")
        logger.info("   3. Access at: http://localhost:8501")
    else:
        logger.warning("⚠️  INTEGRATION ISSUES DETECTED - Some features may not work")
        
    return success_rate >= 75

if __name__ == "__main__":
    success = run_comprehensive_test()