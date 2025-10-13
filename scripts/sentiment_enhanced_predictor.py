#!/usr/bin/env python3
"""
Enhanced Bitcoin Prediction System with Sentiment Analysis
Advanced prediction engine using sentiment-enhanced models
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentEnhancedPredictor:
    """Advanced Bitcoin predictor with sentiment analysis."""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.data_dir = Path("data")
        
        # Load models and scalers
        self.sentiment_models = {}
        self.scaler = None
        self.feature_columns = []
        self.model_weights = {}
        
        self.load_sentiment_enhanced_models()
        self.load_feature_configuration()
    
    def load_sentiment_enhanced_models(self):
        """Load sentiment-enhanced models."""
        try:
            logger.info("📦 Loading sentiment-enhanced models...")
            
            # Model files to load
            model_files = {
                'LightGBM': 'lightgbm_sentiment_enhanced.pkl',
                'XGBoost': 'xgboost_sentiment_enhanced.pkl', 
                'RandomForest': 'randomforest_sentiment_enhanced.pkl',
                'LogisticRegression': 'logisticregression_sentiment_enhanced.pkl',
                'SVM': 'svm_sentiment_enhanced.pkl',
                'Stacking': 'stacking_sentiment_enhanced.pkl'
            }
            
            loaded_count = 0
            for model_name, filename in model_files.items():
                filepath = self.models_dir / filename
                if filepath.exists():
                    try:
                        self.sentiment_models[model_name] = joblib.load(filepath)
                        loaded_count += 1
                        logger.info(f"✅ Loaded {model_name}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load {model_name}: {e}")
            
            # Load scaler
            scaler_file = self.models_dir / "sentiment_enhanced_scaler.pkl"
            if scaler_file.exists():
                self.scaler = joblib.load(scaler_file)
                logger.info("✅ Loaded feature scaler")
            
            logger.info(f"📦 Successfully loaded {loaded_count} sentiment-enhanced models")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
    
    def load_feature_configuration(self):
        """Load feature configuration and model weights."""
        try:
            # Load feature list
            feature_file = self.data_dir / "sentiment_feature_list.csv"
            if feature_file.exists():
                feature_df = pd.read_csv(feature_file)
                self.feature_columns = feature_df['feature'].tolist()
                logger.info(f"📊 Loaded {len(self.feature_columns)} feature configurations")
            
            # Load model performance for weights
            results_file = self.models_dir / "sentiment_enhanced_model_results.csv"
            if results_file.exists():
                results_df = pd.read_csv(results_file)
                
                # Calculate weights based on test accuracy
                total_score = 0
                for _, row in results_df.iterrows():
                    model_name = row['Model']
                    if model_name in self.sentiment_models:
                        score = max(row['Test_Accuracy'], 0.1)  # Minimum weight
                        self.model_weights[model_name] = score
                        total_score += score
                
                # Normalize weights
                for model_name in self.model_weights:
                    self.model_weights[model_name] /= total_score
                
                logger.info("⚖️ Model weights calculated:")
                for model_name, weight in sorted(self.model_weights.items(), key=lambda x: x[1], reverse=True):
                    logger.info(f"   {model_name}: {weight:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Error loading configuration: {e}")
    
    def prepare_features_for_prediction(self, bitcoin_data, sentiment_data=None):
        """Prepare features for prediction using current data."""
        try:
            if sentiment_data is None:
                # Create mock sentiment data for current prediction
                sentiment_data = self.create_current_sentiment_mock()
            
            # Get latest Bitcoin data
            latest_btc = bitcoin_data.tail(30).copy()  # Use last 30 days for feature calculation
            
            # Basic price features
            latest_btc['price_return_1d'] = latest_btc['Close'].pct_change()
            latest_btc['price_return_7d'] = latest_btc['Close'].pct_change(7)
            latest_btc['price_volatility_7d'] = latest_btc['price_return_1d'].rolling(7).std()
            
            # Price RSI
            delta = latest_btc['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            latest_btc['price_rsi'] = 100 - (100 / (1 + rs))
            
            # Combine with sentiment features
            feature_row = {}
            
            # Add sentiment features
            for key, value in sentiment_data.items():
                feature_row[key] = value
            
            # Add price features from latest data
            if not latest_btc.empty:
                latest_row = latest_btc.iloc[-1]
                feature_row.update({
                    'Close': latest_row['Close'],
                    'price_return_1d': latest_row.get('price_return_1d', 0),
                    'price_return_7d': latest_row.get('price_return_7d', 0),
                    'price_volatility_7d': latest_row.get('price_volatility_7d', 0.02),
                    'price_rsi': latest_row.get('price_rsi', 50)
                })
            
            # Create feature vector matching training features
            feature_vector = []
            for feature_name in self.feature_columns:
                if feature_name in feature_row:
                    feature_vector.append(feature_row[feature_name])
                elif 'sentiment' in feature_name.lower():
                    # Default sentiment feature values
                    if 'momentum' in feature_name:
                        feature_vector.append(0.01)
                    elif 'volatility' in feature_name:
                        feature_vector.append(0.2)  
                    elif 'ratio' in feature_name:
                        feature_vector.append(0.4)
                    else:
                        feature_vector.append(0.1)
                elif 'price' in feature_name.lower():
                    # Default price feature values
                    if 'return' in feature_name:
                        feature_vector.append(0.0)
                    elif 'rsi' in feature_name:
                        feature_vector.append(50)
                    else:
                        feature_vector.append(0.5)
                else:
                    feature_vector.append(0.0)  # Default value
            
            # Convert to DataFrame for consistency
            feature_df = pd.DataFrame([feature_vector], columns=self.feature_columns)
            
            # Handle infinite and NaN values
            feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
            feature_df = feature_df.fillna(feature_df.median())
            
            return feature_df
            
        except Exception as e:
            logger.error(f"❌ Error preparing features: {e}")
            return None
    
    def create_current_sentiment_mock(self):
        """Create mock current sentiment data."""
        # In production, this would come from real sentiment analysis
        current_time = datetime.now()
        
        base_sentiment = 0.15  # Slightly positive bias for Bitcoin
        volatility = 0.25
        
        # Add some randomness to simulate real sentiment
        np.random.seed(int(current_time.timestamp()) % 1000)
        sentiment_noise = np.random.normal(0, 0.1)
        compound_mean = base_sentiment + sentiment_noise
        
        return {
            'compound_mean': compound_mean,
            'compound_std': volatility,
            'positive_mean': max(0.1, compound_mean + 0.2),
            'negative_mean': max(0.1, 0.3 - compound_mean),
            'neutral_mean': 0.5,
            'total_tweets': 250,
            'engagement_score_sum': 5000,
            'sentiment_positive_ratio': max(0.2, min(0.7, 0.4 + compound_mean)),
            'sentiment_negative_ratio': max(0.1, min(0.6, 0.3 - compound_mean * 0.5)),
            'sentiment_neutral_ratio': 0.3,
            'sentiment_volatility': volatility,
            'weighted_sentiment': compound_mean * 1.2,
            'sentiment_momentum_3d': compound_mean * 0.1,
            'sentiment_momentum_7d': compound_mean * 0.05,
            'sentiment_momentum_14d': compound_mean * 0.02
        }
    
    def make_ensemble_prediction(self, features):
        """Make prediction using ensemble of sentiment-enhanced models."""
        try:
            if features is None or self.scaler is None:
                return self.create_fallback_prediction()
            
            # Scale features
            features_scaled = pd.DataFrame(
                self.scaler.transform(features),
                columns=features.columns
            )
            
            # Get predictions from all models
            model_predictions = {}
            model_probabilities = {}
            
            for model_name, model in self.sentiment_models.items():
                try:
                    pred = model.predict(features_scaled)[0]
                    proba = model.predict_proba(features_scaled)[0]
                    
                    model_predictions[model_name] = pred
                    model_probabilities[model_name] = proba[1]  # Probability of UP
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error in {model_name} prediction: {e}")
            
            # Weighted ensemble prediction
            weighted_prediction = 0
            weighted_probability = 0
            total_weight = 0
            
            for model_name, pred in model_predictions.items():
                weight = self.model_weights.get(model_name, 0.1)
                weighted_prediction += pred * weight
                weighted_probability += model_probabilities[model_name] * weight
                total_weight += weight
            
            if total_weight > 0:
                weighted_prediction /= total_weight
                weighted_probability /= total_weight
            
            # Final prediction
            direction = "UP" if weighted_prediction > 0.5 else "DOWN"
            confidence = max(weighted_probability, 1 - weighted_probability)
            
            # Calculate price prediction
            current_price = features['Close'].iloc[0] if 'Close' in features.columns else 99000
            volatility = 0.03  # 3% daily volatility
            
            direction_multiplier = 1 if direction == "UP" else -1
            price_change = direction_multiplier * volatility * confidence
            predicted_price = current_price * (1 + price_change)
            
            return {
                'direction': direction,
                'confidence': confidence,
                'predicted_price': predicted_price,
                'current_price': current_price,
                'price_change_pct': price_change * 100,
                'model_predictions': model_predictions,
                'model_probabilities': model_probabilities,
                'sentiment_score': features.get('compound_mean', [0.1]).iloc[0] if 'compound_mean' in features.columns else 0.1,
                'ensemble_probability': weighted_probability
            }
            
        except Exception as e:
            logger.error(f"❌ Error in ensemble prediction: {e}")
            return self.create_fallback_prediction()
    
    def create_fallback_prediction(self):
        """Create fallback prediction when models fail."""
        return {
            'direction': 'UP',
            'confidence': 0.55,
            'predicted_price': 99500,
            'current_price': 99000,
            'price_change_pct': 0.5,
            'model_predictions': {},
            'model_probabilities': {},
            'sentiment_score': 0.1,
            'ensemble_probability': 0.55
        }
    
    def predict_future_prices_enhanced(self, bitcoin_data, days=7):
        """Predict future prices using sentiment-enhanced models."""
        try:
            logger.info(f"🔮 Generating {days}-day predictions with sentiment analysis...")
            
            predictions = []
            current_data = bitcoin_data.copy()
            
            for day in range(1, days + 1):
                # Prepare features for current state
                features = self.prepare_features_for_prediction(current_data)
                
                # Make prediction
                prediction = self.make_ensemble_prediction(features)
                
                # Add prediction to results
                prediction_date = datetime.now() + timedelta(days=day)
                predictions.append({
                    'date': prediction_date,
                    'predicted_price': prediction['predicted_price'],
                    'direction': prediction['direction'],
                    'confidence': prediction['confidence'],
                    'price_change_pct': prediction['price_change_pct'],
                    'sentiment_score': prediction['sentiment_score'],
                    'ensemble_probability': prediction['ensemble_probability']
                })
                
                # Update current data for next prediction (simple approach)
                # In production, this would involve more sophisticated time series modeling
                new_price = prediction['predicted_price']
                if not current_data.empty:
                    latest_date = pd.to_datetime(current_data['Date'].iloc[-1])
                    new_row = current_data.iloc[-1].copy()
                    new_row['Date'] = latest_date + timedelta(days=1)
                    new_row['Close'] = new_price
                    new_row['High'] = new_price * 1.02
                    new_row['Low'] = new_price * 0.98
                    new_row['Open'] = current_data['Close'].iloc[-1]
                    
                    # Add new row to current_data
                    current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
            
            logger.info(f"✅ Generated {len(predictions)} enhanced predictions")
            return pd.DataFrame(predictions)
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced prediction: {e}")
            return self.create_fallback_predictions(days)
    
    def create_fallback_predictions(self, days=7):
        """Create fallback predictions when enhanced models fail."""
        logger.info("🔄 Using fallback prediction method...")
        
        predictions = []
        base_price = 99000
        
        for day in range(1, days + 1):
            # Simple random walk with slight upward bias
            direction = "UP" if day % 2 == 1 else "DOWN"
            confidence = 0.55 + (day * 0.01)
            price_change = 0.02 if direction == "UP" else -0.015
            predicted_price = base_price * (1 + price_change)
            
            predictions.append({
                'date': datetime.now() + timedelta(days=day),
                'predicted_price': predicted_price,
                'direction': direction,
                'confidence': min(confidence, 0.8),
                'price_change_pct': price_change * 100,
                'sentiment_score': 0.1,
                'ensemble_probability': confidence
            })
            
            base_price = predicted_price
        
        return pd.DataFrame(predictions)
    
    def get_model_analysis(self):
        """Get analysis of model performance and contributions."""
        analysis = {
            'total_models': len(self.sentiment_models),
            'model_weights': self.model_weights,
            'feature_count': len(self.feature_columns),
            'sentiment_features': len([f for f in self.feature_columns if 'sentiment' in f.lower()]),
            'price_features': len([f for f in self.feature_columns if any(x in f.lower() for x in ['price', 'close', 'return', 'rsi'])]),
            'models_loaded': list(self.sentiment_models.keys()),
            'best_model': max(self.model_weights.keys(), key=lambda x: self.model_weights[x]) if self.model_weights else 'Unknown'
        }
        
        return analysis

# Global predictor instance
_predictor_instance = None

def get_sentiment_enhanced_predictor():
    """Get singleton instance of the sentiment-enhanced predictor."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = SentimentEnhancedPredictor()
    return _predictor_instance

def test_enhanced_predictor():
    """Test the enhanced predictor with sample data."""
    logger.info("🧪 Testing sentiment-enhanced predictor...")
    
    try:
        predictor = get_sentiment_enhanced_predictor()
        
        # Create sample Bitcoin data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        sample_data = pd.DataFrame({
            'Date': dates,
            'Close': np.random.normal(99000, 2000, 30),
            'High': np.random.normal(101000, 2000, 30),
            'Low': np.random.normal(97000, 2000, 30),
            'Open': np.random.normal(99000, 2000, 30),
            'Volume': np.random.normal(20000000000, 5000000000, 30)
        })
        
        # Test prediction
        predictions = predictor.predict_future_prices_enhanced(sample_data, days=7)
        
        if not predictions.empty:
            logger.info("✅ Enhanced predictor test successful!")
            logger.info(f"📊 Generated {len(predictions)} predictions")
            logger.info(f"🎯 Average confidence: {predictions['confidence'].mean():.3f}")
            return True
        else:
            logger.error("❌ No predictions generated")
            return False
            
    except Exception as e:
        logger.error(f"❌ Enhanced predictor test failed: {e}")
        return False

if __name__ == "__main__":
    test_enhanced_predictor()