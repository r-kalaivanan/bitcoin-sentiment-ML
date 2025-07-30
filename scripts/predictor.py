# scripts/predictor.py

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BitcoinPredictor:
    """
    Bitcoin price prediction system using trained ML models.
    Makes real-time predictions and provides confidence scores.
    """
    
    def __init__(self, model_path='models/', model_name='best'):
        self.model_path = model_path
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.selected_features = None
        self.feature_engineer = None
        self.sentiment_analyzer = None
        
    def load_model_artifacts(self):
        """Load trained model and preprocessing artifacts."""
        try:
            print("🤖 Loading trained model artifacts...")
            
            # Find the best model file
            import os
            model_files = [f for f in os.listdir(self.model_path) if f.endswith('_best.pkl')]
            if not model_files:
                raise FileNotFoundError("No best model file found")
            
            model_file = model_files[0]
            self.model = joblib.load(f"{self.model_path}{model_file}")
            
            # Load scaler
            self.scaler = joblib.load(f"{self.model_path}feature_scaler.pkl")
            
            # Load selected features
            features_df = pd.read_csv(f"{self.model_path}selected_features.csv")
            self.selected_features = features_df['feature'].tolist()
            
            print(f"✅ Model artifacts loaded successfully")
            print(f"Model: {model_file}")
            print(f"Features: {len(self.selected_features)}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model artifacts: {str(e)}")
            return False
    
    def get_latest_data(self, days_back=5):
        """
        Get the latest Bitcoin data for prediction.
        
        Args:
            days_back: Number of days of data to retrieve
        """
        try:
            print(f"📥 Fetching latest Bitcoin data ({days_back} days)...")
            
            # Import feature engineering and sentiment analysis
            from feature_engineering import FeatureEngineer
            from sentiment_analysis import SentimentAnalyzer
            from data_merger import DataMerger
            
            # Get latest price data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days_back+50)).strftime('%Y-%m-%d')  # Extra buffer for indicators
            
            engineer = FeatureEngineer(start_date=start_date, end_date=end_date)
            price_data = engineer.generate_features(save_to_file=False)
            
            # Get latest sentiment data
            analyzer = SentimentAnalyzer()
            sentiment_data = analyzer.run_sentiment_pipeline(historical=False, scrape_recent=True)
            
            # Merge data
            merger = DataMerger()
            merger.price_data = price_data
            merger.sentiment_data = sentiment_data if not sentiment_data.empty else None
            
            if merger.sentiment_data is not None:
                merger.merge_datasets(merge_type='left')
                merger.handle_missing_sentiment(method='neutral')
                merger.create_lagged_sentiment_features()
                merger.create_sentiment_price_interactions()
                latest_data = merger.merged_data
            else:
                # Use only price data if sentiment is not available
                latest_data = price_data
                print("⚠️ Sentiment data not available, using price features only")
            
            # Get the most recent complete row (not the last row which might have NaN target)
            latest_data = latest_data.dropna(subset=[col for col in latest_data.columns 
                                                   if col in self.selected_features])
            
            if len(latest_data) == 0:
                raise ValueError("No valid data available for prediction")
            
            # Take the last complete row
            self.latest_data = latest_data.iloc[-1:].copy()
            
            print(f"✅ Retrieved latest data for {self.latest_data['Date'].iloc[0]}")
            return True
            
        except Exception as e:
            print(f"❌ Error getting latest data: {str(e)}")
            return False
    
    def prepare_prediction_features(self):
        """Prepare features for prediction."""
        try:
            print("🔧 Preparing prediction features...")
            
            # Select only the features used in training
            available_features = [f for f in self.selected_features if f in self.latest_data.columns]
            missing_features = [f for f in self.selected_features if f not in self.latest_data.columns]
            
            if missing_features:
                print(f"⚠️ Missing features: {len(missing_features)}")
                # Fill missing features with neutral values
                for feature in missing_features:
                    if 'sentiment' in feature.lower():
                        self.latest_data[feature] = 0.0  # Neutral sentiment
                    else:
                        self.latest_data[feature] = 0.0  # Default value
            
            # Extract features in the correct order
            self.X_latest = self.latest_data[self.selected_features].copy()
            
            # Handle any remaining missing values
            self.X_latest = self.X_latest.fillna(0)
            
            # Scale features
            self.X_latest_scaled = pd.DataFrame(
                self.scaler.transform(self.X_latest),
                columns=self.X_latest.columns,
                index=self.X_latest.index
            )
            
            print(f"✅ Features prepared for prediction")
            return True
            
        except Exception as e:
            print(f"❌ Error preparing features: {str(e)}")
            return False
    
    def make_prediction(self):
        """Make price direction prediction."""
        try:
            print("🔮 Making prediction...")
            
            # Get prediction
            prediction = self.model.predict(self.X_latest_scaled)[0]
            
            # Get prediction probability if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(self.X_latest_scaled)[0]
                confidence = max(probabilities)
                prob_up = probabilities[1]
                prob_down = probabilities[0]
            else:
                confidence = 0.5
                prob_up = 0.5 if prediction == 1 else 0.5
                prob_down = 0.5 if prediction == 0 else 0.5
            
            # Create prediction result
            self.prediction_result = {
                'date': self.latest_data['Date'].iloc[0],
                'current_price': self.latest_data['Close'].iloc[0],
                'prediction': 'UP' if prediction == 1 else 'DOWN',
                'prediction_numeric': int(prediction),
                'confidence': confidence,
                'probability_up': prob_up,
                'probability_down': prob_down,
                'model_used': type(self.model).__name__
            }
            
            print(f"✅ Prediction complete")
            return True
            
        except Exception as e:
            print(f"❌ Error making prediction: {str(e)}")
            return False
    
    def get_feature_importance(self, top_n=10):
        """Get the most important features for the prediction."""
        try:
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': self.selected_features,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                return feature_importance.head(top_n)
            else:
                return None
        except:
            return None
    
    def display_prediction(self):
        """Display the prediction results in a formatted way."""
        if not hasattr(self, 'prediction_result'):
            print("❌ No prediction available")
            return
        
        result = self.prediction_result
        
        print("\n" + "="*50)
        print("🔮 BITCOIN PRICE PREDICTION")
        print("="*50)
        print(f"📅 Date: {result['date']}")
        print(f"💰 Current Price: ${result['current_price']:.2f}")
        print(f"🎯 Prediction: {result['prediction']}")
        print(f"📊 Confidence: {result['confidence']:.1%}")
        print(f"📈 Probability UP: {result['probability_up']:.1%}")
        print(f"📉 Probability DOWN: {result['probability_down']:.1%}")
        print(f"🤖 Model: {result['model_used']}")
        
        # Display feature importance
        feature_importance = self.get_feature_importance(top_n=5)
        if feature_importance is not None:
            print(f"\n🎯 TOP 5 INFLUENTIAL FEATURES:")
            for _, row in feature_importance.iterrows():
                print(f"   {row['feature']}: {row['importance']:.3f}")
        
        print("="*50)
    
    def save_prediction(self, filepath='predictions/latest_prediction.csv'):
        """Save prediction to file."""
        if not hasattr(self, 'prediction_result'):
            print("❌ No prediction to save")
            return False
        
        try:
            import os
            os.makedirs('predictions', exist_ok=True)
            
            # Create prediction dataframe
            pred_df = pd.DataFrame([self.prediction_result])
            
            # Save to CSV
            pred_df.to_csv(filepath, index=False)
            
            # Also append to history file
            history_file = 'predictions/prediction_history.csv'
            if os.path.exists(history_file):
                history_df = pd.read_csv(history_file)
                updated_history = pd.concat([history_df, pred_df], ignore_index=True)
            else:
                updated_history = pred_df
            
            updated_history.to_csv(history_file, index=False)
            
            print(f"✅ Prediction saved to {filepath}")
            print(f"✅ Added to prediction history: {history_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving prediction: {str(e)}")
            return False
    
    def run_prediction_pipeline(self):
        """Run the complete prediction pipeline."""
        print("🚀 Starting Bitcoin price prediction pipeline...")
        
        # Load model artifacts
        if not self.load_model_artifacts():
            return None
        
        # Get latest data
        if not self.get_latest_data():
            return None
        
        # Prepare features
        if not self.prepare_prediction_features():
            return None
        
        # Make prediction
        if not self.make_prediction():
            return None
        
        # Display and save results
        self.display_prediction()
        self.save_prediction()
        
        print(f"🎉 Prediction pipeline complete!")
        return self.prediction_result

def predict_bitcoin_price():
    """Convenience function to make a quick prediction."""
    predictor = BitcoinPredictor()
    return predictor.run_prediction_pipeline()

if __name__ == "__main__":
    # Run prediction
    result = predict_bitcoin_price()
    
    if result:
        print(f"\n🎯 QUICK SUMMARY:")
        print(f"Bitcoin is predicted to go {result['prediction']} with {result['confidence']:.1%} confidence")
    else:
        print("❌ Prediction failed")
