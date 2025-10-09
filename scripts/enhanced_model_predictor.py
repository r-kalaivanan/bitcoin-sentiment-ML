import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score, mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVR
import joblib
import sqlite3
from datetime import datetime, timedelta
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

class EnhancedBitcoinPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.ensemble_model = None
        self.feature_importance = None
        self.model_weights = {}
        self.performance_history = []
        
    def load_data_with_sentiment(self):
        """Load Bitcoin data enhanced with sentiment scores"""
        print("📊 Loading and preparing enhanced dataset...")
        
        # Load main Bitcoin features
        try:
            btc_data = pd.read_csv('data/btc_features_enhanced.csv')
            btc_data['Date'] = pd.to_datetime(btc_data['Date'])
            print(f"✅ Loaded {len(btc_data)} Bitcoin records")
        except FileNotFoundError:
            print("❌ Enhanced Bitcoin features not found. Run feature_engineering.py first.")
            return None
        
        # Load sentiment data if available
        try:
            conn = sqlite3.connect('data/sentiment_data.db')
            sentiment_df = pd.read_sql_query('''
                SELECT 
                    DATE(timestamp) as date,
                    AVG(sentiment_score) as avg_sentiment,
                    AVG(confidence) as avg_confidence,
                    COUNT(*) as sentiment_volume,
                    SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as positive_ratio,
                    SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as negative_ratio
                FROM sentiment_scores
                GROUP BY DATE(timestamp)
                ORDER BY date
            ''', conn)
            conn.close()
            
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
            print(f"✅ Loaded sentiment data for {len(sentiment_df)} days")
            
            # Merge with Bitcoin data
            btc_data = btc_data.merge(
                sentiment_df, 
                left_on='Date', 
                right_on='date', 
                how='left'
            )
            
            # Fill missing sentiment values
            sentiment_cols = ['avg_sentiment', 'avg_confidence', 'sentiment_volume', 'positive_ratio', 'negative_ratio']
            for col in sentiment_cols:
                if col in btc_data.columns:
                    btc_data[col] = btc_data[col].fillna(0)
            
            print(f"✅ Merged data shape: {btc_data.shape}")
            
        except Exception as e:
            print(f"⚠️ Sentiment data not available: {e}")
            # Add dummy sentiment columns
            btc_data['avg_sentiment'] = 0
            btc_data['avg_confidence'] = 0.5
            btc_data['sentiment_volume'] = 0
            btc_data['positive_ratio'] = 0.33
            btc_data['negative_ratio'] = 0.33
        
        return btc_data
        
    def prepare_features(self, df):
        """Prepare features for model training"""
        print("🔧 Preparing features for training...")
        
        # Select features (exclude date, target, and helper columns)
        exclude_cols = ['Date', 'date', 'TARGET', 'TARGET_2D', 'TARGET_STRONG', 'TARGET_UP', 'Close']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        
        # Use the correct target column name
        if 'TARGET' in df.columns:
            y = df['TARGET'].copy()
        elif 'TARGET_UP' in df.columns:
            y = df['TARGET_UP'].copy()
        else:
            y = None
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        
        print(f"✅ Feature matrix shape: {X.shape}")
        print(f"✅ Features selected: {len(feature_cols)}")
        
        return X, y, feature_cols
        
    def create_models(self):
        """Create individual models for ensemble"""
        print("🤖 Creating model ensemble...")
        
        models = {
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                eval_metric='logloss'
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        }
        
        return models
        
    def train_individual_models(self, X_train, y_train, X_val, y_val):
        """Train individual models and calculate performance"""
        print("🎯 Training individual models...")
        
        models = self.create_models()
        model_scores = {}
        
        for name, model in models.items():
            try:
                print(f"  Training {name}...")
                
                # Scale features if needed
                if name in ['logistic_regression']:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    self.scalers[name] = scaler
                else:
                    X_train_scaled = X_train
                    X_val_scaled = X_val
                
                # Train model
                if name in ['random_forest', 'gradient_boosting']:
                    # For regression models, predict probability of up movement
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_val_scaled)
                    y_pred_binary = (y_pred > 0.5).astype(int)
                else:
                    model.fit(X_train_scaled, y_train)
                    y_pred_binary = model.predict(X_val_scaled)
                
                # Calculate accuracy
                accuracy = accuracy_score(y_val, y_pred_binary)
                model_scores[name] = accuracy
                
                # Store model
                self.models[name] = model
                
                print(f"    {name} accuracy: {accuracy:.4f}")
                
            except Exception as e:
                print(f"    ❌ {name} failed: {e}")
                model_scores[name] = 0.5  # Random performance
        
        return model_scores
        
    def create_ensemble_model(self, X_train, y_train, model_scores):
        """Create weighted ensemble model"""
        print("🔗 Creating ensemble model...")
        
        # Calculate weights based on performance
        total_score = sum(model_scores.values())
        if total_score > 0:
            self.model_weights = {name: score/total_score for name, score in model_scores.items()}
        else:
            self.model_weights = {name: 1.0/len(model_scores) for name in model_scores.keys()}
        
        print("Model weights:")
        for name, weight in self.model_weights.items():
            print(f"  {name}: {weight:.3f}")
        
        return self.model_weights
        
    def predict_ensemble(self, X):
        """Make ensemble predictions"""
        if not self.models:
            print("❌ No trained models available")
            return None
            
        predictions = {}
        
        for name, model in self.models.items():
            try:
                # Scale features if needed
                if name in self.scalers:
                    X_scaled = self.scalers[name].transform(X)
                else:
                    X_scaled = X
                
                # Get prediction
                if hasattr(model, 'predict_proba') and name not in ['random_forest', 'gradient_boosting']:
                    pred = model.predict_proba(X_scaled)[:, 1]  # Probability of class 1
                else:
                    pred = model.predict(X_scaled)
                    if name in ['random_forest', 'gradient_boosting']:
                        pred = np.clip(pred, 0, 1)  # Clip to [0,1] range
                
                predictions[name] = pred
                
            except Exception as e:
                print(f"⚠️ Prediction failed for {name}: {e}")
                predictions[name] = np.full(len(X), 0.5)  # Default to neutral
        
        # Calculate weighted ensemble prediction
        if predictions:
            ensemble_pred = np.zeros(len(X))
            for name, pred in predictions.items():
                weight = self.model_weights.get(name, 1.0/len(predictions))
                ensemble_pred += weight * pred
                
            return ensemble_pred, predictions
        
        return None, None
        
    def train_models(self):
        """Main training pipeline"""
        print("🚀 Starting enhanced model training pipeline...")
        
        # Load data
        df = self.load_data_with_sentiment()
        if df is None:
            return False
        
        # Prepare features
        X, y, feature_cols = self.prepare_features(df)
        if y is None:
            print("❌ Target variable not found")
            return False
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=3)
        train_idx, val_idx = list(tscv.split(X))[-1]  # Use last split for final training
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        
        # Train individual models
        model_scores = self.train_individual_models(X_train, y_train, X_val, y_val)
        
        # Create ensemble
        ensemble_weights = self.create_ensemble_model(X_train, y_train, model_scores)
        
        # Test ensemble performance
        ensemble_pred, individual_preds = self.predict_ensemble(X_val)
        if ensemble_pred is not None:
            ensemble_binary = (ensemble_pred > 0.5).astype(int)
            ensemble_accuracy = accuracy_score(y_val, ensemble_binary)
            print(f"🎯 Ensemble accuracy: {ensemble_accuracy:.4f}")
        
        # Calculate feature importance
        self.calculate_feature_importance(feature_cols)
        
        # Save models
        self.save_models()
        
        print("✅ Enhanced model training completed!")
        return True
        
    def calculate_feature_importance(self, feature_cols):
        """Calculate and store feature importance"""
        print("📊 Calculating feature importance...")
        
        importance_dict = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance_dict[name] = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        
        if importance_dict:
            # Average importance across models
            avg_importance = np.mean(list(importance_dict.values()), axis=0)
            
            self.feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': avg_importance
            }).sort_values('importance', ascending=False)
            
            print("Top 10 most important features:")
            for idx, row in self.feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
    def save_models(self):
        """Save all models and scalers"""
        print("💾 Saving models...")
        
        try:
            # Save individual models
            for name, model in self.models.items():
                joblib.dump(model, f'models/{name}_enhanced.pkl')
            
            # Save scalers
            for name, scaler in self.scalers.items():
                joblib.dump(scaler, f'models/{name}_scaler_enhanced.pkl')
            
            # Save ensemble weights
            joblib.dump(self.model_weights, 'models/ensemble_weights.pkl')
            
            # Save feature importance
            if self.feature_importance is not None:
                self.feature_importance.to_csv('models/feature_importance_enhanced.csv', index=False)
            
            print("✅ Models saved successfully")
            
        except Exception as e:
            print(f"❌ Error saving models: {e}")
    
    def load_models(self):
        """Load saved models"""
        print("📥 Loading saved models...")
        
        try:
            # Load individual models
            model_names = ['lightgbm', 'xgboost', 'random_forest', 'gradient_boosting', 'logistic_regression']
            
            for name in model_names:
                try:
                    self.models[name] = joblib.load(f'models/{name}_enhanced.pkl')
                except FileNotFoundError:
                    print(f"⚠️ Model {name} not found")
            
            # Load scalers
            for name in model_names:
                try:
                    self.scalers[name] = joblib.load(f'models/{name}_scaler_enhanced.pkl')
                except FileNotFoundError:
                    pass  # Scalers are optional
            
            # Load ensemble weights
            try:
                self.model_weights = joblib.load('models/ensemble_weights.pkl')
            except FileNotFoundError:
                print("⚠️ Ensemble weights not found")
                
            # Load feature importance
            try:
                self.feature_importance = pd.read_csv('models/feature_importance_enhanced.csv')
            except FileNotFoundError:
                print("⚠️ Feature importance not found")
            
            print(f"✅ Loaded {len(self.models)} models")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def predict_next_movement(self, include_sentiment=True):
        """Predict next Bitcoin price movement"""
        print("🔮 Generating enhanced prediction...")
        
        # Load latest data
        df = self.load_data_with_sentiment()
        if df is None:
            return None
        
        # Get latest features
        latest_data = df.iloc[-1:].copy()
        X, _, feature_cols = self.prepare_features(df)
        X_latest = X.iloc[-1:].copy()
        
        # Make ensemble prediction
        ensemble_pred, individual_preds = self.predict_ensemble(X_latest)
        
        if ensemble_pred is None:
            return None
        
        # Get current price info
        current_price = latest_data['Close'].iloc[0] if 'Close' in latest_data.columns else None
        
        # Get sentiment info
        sentiment_info = {}
        if include_sentiment:
            try:
                from enhanced_sentiment_analyzer import CryptoSentimentAnalyzer
                analyzer = CryptoSentimentAnalyzer()
                sentiment_info = analyzer.get_aggregated_sentiment(hours=24)
            except Exception as e:
                print(f"⚠️ Could not get sentiment: {e}")
        
        prediction_result = {
            'timestamp': datetime.now().isoformat(),
            'current_price': current_price,
            'ensemble_prediction': float(ensemble_pred[0]),
            'prediction_label': 'UP' if ensemble_pred[0] > 0.5 else 'DOWN',
            'confidence': abs(ensemble_pred[0] - 0.5) * 2,  # Scale to 0-1
            'individual_predictions': {name: float(pred[0]) for name, pred in individual_preds.items()},
            'model_weights': self.model_weights,
            'sentiment_info': sentiment_info
        }
        
        return prediction_result
        
    def generate_prediction_report(self):
        """Generate comprehensive prediction report"""
        prediction = self.predict_next_movement()
        
        if prediction is None:
            print("❌ Could not generate prediction")
            return None
        
        print("\n🎯 ENHANCED BITCOIN PREDICTION REPORT")
        print("=" * 60)
        print(f"Timestamp: {prediction['timestamp']}")
        
        if prediction['current_price']:
            print(f"Current BTC Price: ${prediction['current_price']:,.2f}")
        
        print(f"\n🔮 ENSEMBLE PREDICTION: {prediction['prediction_label']}")
        print(f"Confidence Level: {prediction['confidence']:.1%}")
        print(f"Raw Score: {prediction['ensemble_prediction']:.3f}")
        
        print(f"\n🤖 INDIVIDUAL MODEL PREDICTIONS:")
        for model, pred in prediction['individual_predictions'].items():
            direction = "UP" if pred > 0.5 else "DOWN"
            confidence = abs(pred - 0.5) * 2
            print(f"  {model:20}: {direction:4} ({confidence:.1%} confidence)")
        
        print(f"\n⚖️ MODEL WEIGHTS:")
        for model, weight in prediction['model_weights'].items():
            print(f"  {model:20}: {weight:.1%}")
        
        if prediction['sentiment_info'] and prediction['sentiment_info']['sample_size'] > 0:
            sentiment = prediction['sentiment_info']
            print(f"\n💭 SENTIMENT ANALYSIS:")
            print(f"  Overall Sentiment: {sentiment['sentiment_label'].upper()} ({sentiment['overall_sentiment']:+.3f})")
            print(f"  Confidence: {sentiment['confidence']:.1%}")
            print(f"  Sample Size: {sentiment['sample_size']} data points")
            print(f"  Trend: {sentiment['time_trend'].upper()}")
        
        # Save prediction
        self.save_prediction(prediction)
        
        return prediction
        
    def save_prediction(self, prediction):
        """Save prediction to file"""
        try:
            predictions_df = pd.DataFrame([prediction])
            
            # Try to append to existing file
            try:
                existing_df = pd.read_csv('predictions/enhanced_predictions.csv')
                predictions_df = pd.concat([existing_df, predictions_df], ignore_index=True)
            except FileNotFoundError:
                pass
            
            predictions_df.to_csv('predictions/enhanced_predictions.csv', index=False)
            print("💾 Prediction saved to predictions/enhanced_predictions.csv")
            
        except Exception as e:
            print(f"⚠️ Could not save prediction: {e}")

if __name__ == "__main__":
    predictor = EnhancedBitcoinPredictor()
    
    # Try to load existing models first
    if not predictor.load_models() or len(predictor.models) == 0:
        print("🎯 No existing models found. Training new models...")
        success = predictor.train_models()
        if not success:
            print("❌ Model training failed")
            exit(1)
    
    # Generate prediction report
    predictor.generate_prediction_report()