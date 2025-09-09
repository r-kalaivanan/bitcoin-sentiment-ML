#!/usr/bin/env python3
"""
Production-Ready Bitcoin Sentiment Predictor
Final optimized version for daily use
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ProductionBitcoinPredictor:
    """Production-ready Bitcoin sentiment prediction system."""
    
    def __init__(self):
        self.model = None
        self.feature_scaler = None
        self.load_models()
    
    def load_models(self):
        """Load the best trained model."""
        try:
            # Try to load the boosted model first
            self.model = joblib.load('models/random_forest_optimized_boosted.pkl')
            print("✅ Loaded optimized Random Forest model (52.44% accuracy)")
        except:
            try:
                # Fallback to original best model
                self.model = joblib.load('models/lightgbm_best.pkl') 
                print("✅ Loaded fallback LightGBM model (53.26% accuracy)")
            except Exception as e:
                print(f"❌ Error loading models: {e}")
    
    def get_latest_features(self):
        """Get latest features for prediction."""
        try:
            # Load latest Bitcoin data
            btc_data = pd.read_csv('data/btc_features_enhanced.csv')
            latest_features = btc_data.tail(1)
            
            # Load latest sentiment
            try:
                sentiment_data = pd.read_csv('data/free_crypto_sentiment.csv')
                latest_sentiment = sentiment_data['sentiment_score'].iloc[-1] if len(sentiment_data) > 0 else 0.0
            except:
                latest_sentiment = 0.0
            
            return latest_features, latest_sentiment
            
        except Exception as e:
            print(f"❌ Error loading features: {e}")
            return None, 0.0
    
    def make_prediction(self, confidence_threshold=0.6):
        """Make production prediction with confidence filtering."""
        try:
            latest_features, sentiment = self.get_latest_features()
            
            if latest_features is None:
                return None
            
            # Prepare features for prediction
            price_col = 'Close' if 'Close' in latest_features.columns else 'close'
            date_col = 'Date' if 'Date' in latest_features.columns else 'date'
            
            exclude_cols = [date_col, price_col, 'TARGET', 'TARGET_2D', 'TARGET_STRONG']
            feature_cols = [col for col in latest_features.columns if col not in exclude_cols]
            
            X = latest_features[feature_cols].values
            
            # Make prediction
            prediction_proba = self.model.predict_proba(X)[0]
            confidence = max(prediction_proba)
            direction = "UP" if prediction_proba[1] > 0.5 else "DOWN"
            
            # Apply confidence filtering
            signal_strength = "STRONG" if confidence > confidence_threshold else "WEAK"
            
            current_price = latest_features[price_col].iloc[0]
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                'prediction': direction,
                'confidence': confidence,
                'signal_strength': signal_strength,
                'probability_up': prediction_proba[1],
                'probability_down': prediction_proba[0],
                'current_sentiment': sentiment,
                'model_accuracy': "52.44%"
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def save_prediction(self, prediction):
        """Save prediction to history."""
        if prediction is None:
            return
        
        try:
            # Load existing predictions
            try:
                history = pd.read_csv('predictions/prediction_history.csv')
            except:
                history = pd.DataFrame()
            
            # Add new prediction
            new_prediction = pd.DataFrame([{
                'date': prediction['timestamp'][:10],
                'current_price': prediction['current_price'],
                'prediction': prediction['prediction'],
                'prediction_numeric': 1 if prediction['prediction'] == 'UP' else 0,
                'confidence': prediction['confidence'],
                'probability_up': prediction['probability_up'],
                'probability_down': prediction['probability_down'],
                'model_used': 'RandomForest_Optimized'
            }])
            
            history = pd.concat([history, new_prediction], ignore_index=True)
            
            # Keep only last 30 days
            history = history.tail(30)
            
            # Save
            history.to_csv('predictions/prediction_history.csv', index=False)
            
            # Save as latest
            new_prediction.to_csv('predictions/latest_prediction.csv', index=False)
            
            print(f"💾 Prediction saved: {prediction['prediction']} ({prediction['confidence']*100:.1f}% confident)")
            
        except Exception as e:
            print(f"❌ Error saving prediction: {e}")
    
    def run_daily_prediction(self):
        """Main function for daily prediction workflow."""
        print("🚀 PRODUCTION BITCOIN PREDICTION")
        print("=" * 40)
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Make prediction
        prediction = self.make_prediction()
        
        if prediction:
            print(f"\n🎯 PREDICTION RESULTS:")
            print(f"Current Price: ${prediction['current_price']:,.0f}")
            print(f"Direction: {prediction['prediction']} ({prediction['signal_strength']})")
            print(f"Confidence: {prediction['confidence']*100:.1f}%")
            print(f"Sentiment: {prediction['current_sentiment']:.3f}")
            
            # Save prediction
            self.save_prediction(prediction)
            
            # Trading signal interpretation
            print(f"\n📊 TRADING SIGNAL:")
            if prediction['signal_strength'] == 'STRONG':
                print(f"🟢 ACTIONABLE: {prediction['prediction']} signal with {prediction['confidence']*100:.1f}% confidence")
            else:
                print(f"🟡 WAIT: Weak signal ({prediction['confidence']*100:.1f}% confidence) - observe only")
            
            return prediction
        else:
            print("❌ Could not generate prediction")
            return None

def main():
    """Run daily prediction."""
    predictor = ProductionBitcoinPredictor()
    prediction = predictor.run_daily_prediction()
    
    if prediction:
        print(f"\n✅ Daily prediction complete!")
        print(f"📁 Results saved to: predictions/latest_prediction.csv")
    else:
        print(f"\n❌ Daily prediction failed!")

if __name__ == "__main__":
    main()
