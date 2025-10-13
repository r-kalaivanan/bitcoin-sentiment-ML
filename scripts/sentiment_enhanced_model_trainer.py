#!/usr/bin/env python3
"""
Enhanced Model Training Pipeline with Sentiment Analysis
Trains ML models with combined price and sentiment features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xgb
import joblib
import os
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentEnhancedModelTrainer:
    """Train ML models with sentiment and price features."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.feature_importance = {}
        self.data_dir = "data"
        self.models_dir = "models"
        
        # Ensure directories exist
        os.makedirs(self.models_dir, exist_ok=True)
        
    def load_enhanced_features(self):
        """Load enhanced sentiment+price features."""
        try:
            features_file = os.path.join(self.data_dir, "btc_sentiment_features_enhanced.csv")
            
            if os.path.exists(features_file):
                logger.info("📊 Loading enhanced sentiment+price features...")
                df = pd.read_csv(features_file)
                df['date'] = pd.to_datetime(df['date'])
                logger.info(f"✅ Loaded {len(df)} records with {len(df.columns)} features")
                return df
            else:
                logger.error("❌ Enhanced features file not found!")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error loading enhanced features: {e}")
            return None
    
    def prepare_features_and_target(self, df):
        """Prepare feature matrix and target variable."""
        try:
            # Exclude non-feature columns
            exclude_cols = ['date', 'Date', 'target', 'next_day_return']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            # Prepare features and target
            X = df[feature_cols].copy()
            y = df['target'].copy()
            
            # Handle infinite values
            X = X.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values with median (more robust than mean)
            X = X.fillna(X.median())
            
            logger.info(f"📊 Prepared features: {X.shape}")
            logger.info(f"🎯 Target distribution: {y.value_counts().to_dict()}")
            logger.info(f"📈 Feature categories:")
            
            # Categorize features
            sentiment_features = [col for col in feature_cols if 'sentiment' in col.lower()]
            price_features = [col for col in feature_cols if any(x in col.lower() for x in ['close', 'price', 'return', 'rsi', 'bb', 'ma']) and 'sentiment' not in col.lower()]
            other_features = [col for col in feature_cols if col not in sentiment_features + price_features]
            
            logger.info(f"   🎭 Sentiment features: {len(sentiment_features)}")
            logger.info(f"   💰 Price features: {len(price_features)}")
            logger.info(f"   📊 Other features: {len(other_features)}")
            
            return X, y, feature_cols
            
        except Exception as e:
            logger.error(f"❌ Error preparing features: {e}")
            return None, None, None
    
    def create_time_series_split(self, X, y, test_size=0.2, validation_size=0.1):
        """Create time-aware train/validation/test splits."""
        try:
            n_samples = len(X)
            
            # Calculate split indices
            test_start = int(n_samples * (1 - test_size))
            val_start = int(test_start * (1 - validation_size))
            
            # Create splits
            X_train = X.iloc[:val_start]
            y_train = y.iloc[:val_start]
            
            X_val = X.iloc[val_start:test_start]
            y_val = y.iloc[val_start:test_start]
            
            X_test = X.iloc[test_start:]
            y_test = y.iloc[test_start:]
            
            logger.info(f"📊 Data splits:")
            logger.info(f"   🏋️ Training: {len(X_train)} samples")
            logger.info(f"   🔍 Validation: {len(X_val)} samples")
            logger.info(f"   🧪 Test: {len(X_test)} samples")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            logger.error(f"❌ Error creating time series split: {e}")
            return None, None, None, None, None, None
    
    def scale_features(self, X_train, X_val, X_test):
        """Scale features using robust scaling."""
        try:
            logger.info("⚖️ Scaling features...")
            
            # Use RobustScaler for better handling of outliers
            scaler = RobustScaler()
            
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            
            X_val_scaled = pd.DataFrame(
                scaler.transform(X_val),
                columns=X_val.columns,
                index=X_val.index
            )
            
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            
            # Save scaler
            scaler_file = os.path.join(self.models_dir, "sentiment_enhanced_scaler.pkl")
            joblib.dump(scaler, scaler_file)
            logger.info(f"✅ Scaler saved to {scaler_file}")
            
            return X_train_scaled, X_val_scaled, X_test_scaled, scaler
            
        except Exception as e:
            logger.error(f"❌ Error scaling features: {e}")
            return X_train, X_val, X_test, None
    
    def initialize_models(self):
        """Initialize ML models with optimized hyperparameters."""
        
        self.models = {
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=-1
            ),
            
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            ),
            
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42
            ),
            
            'LogisticRegression': LogisticRegression(
                C=1.0,
                penalty='l2',
                random_state=42,
                max_iter=1000
            ),
            
            'SVM': SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=42
            )
        }
        
        logger.info(f"🤖 Initialized {len(self.models)} models")
    
    def train_individual_models(self, X_train, X_val, y_train, y_val):
        """Train individual models and evaluate performance."""
        
        logger.info("🏋️ Training individual models...")
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"🔄 Training {model_name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_val_pred = model.predict(X_val)
                y_val_proba = model.predict_proba(X_val)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_val, y_val_pred)
                precision = precision_score(y_val, y_val_pred)
                recall = recall_score(y_val, y_val_pred)
                f1 = f1_score(y_val, y_val_pred)
                auc = roc_auc_score(y_val, y_val_proba)
                
                self.results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc
                }
                
                # Save feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = dict(zip(X_train.columns, model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    self.feature_importance[model_name] = dict(zip(X_train.columns, abs(model.coef_[0])))
                
                # Save model
                model_file = os.path.join(self.models_dir, f"{model_name.lower()}_sentiment_enhanced.pkl")
                joblib.dump(model, model_file)
                
                logger.info(f"✅ {model_name}: Accuracy={accuracy:.4f}, AUC={auc:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Error training {model_name}: {e}")
                self.results[model_name] = {
                    'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0
                }
    
    def create_ensemble_models(self, X_train, X_val, y_train, y_val):
        """Create ensemble models using voting and stacking."""
        
        logger.info("🔗 Creating ensemble models...")
        
        try:
            # Prepare base models for ensemble
            base_models = []
            for name in ['LightGBM', 'XGBoost', 'RandomForest']:
                if name in self.models and self.results[name]['accuracy'] > 0.5:
                    base_models.append((name.lower(), self.models[name]))
            
            if len(base_models) < 2:
                logger.warning("⚠️ Not enough good models for ensemble")
                return
            
            # 1. Voting Classifier (Hard and Soft voting)
            voting_hard = VotingClassifier(estimators=base_models, voting='hard')
            voting_soft = VotingClassifier(estimators=base_models, voting='soft')
            
            ensemble_models = {
                'VotingHard': voting_hard,
                'VotingSoft': voting_soft
            }
            
            # 2. Stacking Classifier
            if len(base_models) >= 3:
                stacking = StackingClassifier(
                    estimators=base_models,
                    final_estimator=LogisticRegression(random_state=42),
                    cv=3
                )
                ensemble_models['Stacking'] = stacking
            
            # Train and evaluate ensemble models
            for ensemble_name, ensemble_model in ensemble_models.items():
                try:
                    logger.info(f"🔄 Training {ensemble_name}...")
                    
                    ensemble_model.fit(X_train, y_train)
                    
                    y_val_pred = ensemble_model.predict(X_val)
                    y_val_proba = ensemble_model.predict_proba(X_val)[:, 1]
                    
                    accuracy = accuracy_score(y_val, y_val_pred)
                    precision = precision_score(y_val, y_val_pred)
                    recall = recall_score(y_val, y_val_pred)
                    f1 = f1_score(y_val, y_val_pred)
                    auc = roc_auc_score(y_val, y_val_proba)
                    
                    self.results[ensemble_name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'auc': auc
                    }
                    
                    self.models[ensemble_name] = ensemble_model
                    
                    # Save ensemble model
                    model_file = os.path.join(self.models_dir, f"{ensemble_name.lower()}_sentiment_enhanced.pkl")
                    joblib.dump(ensemble_model, model_file)
                    
                    logger.info(f"✅ {ensemble_name}: Accuracy={accuracy:.4f}, AUC={auc:.4f}")
                    
                except Exception as e:
                    logger.error(f"❌ Error training {ensemble_name}: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Error creating ensemble models: {e}")
    
    def evaluate_on_test_set(self, X_test, y_test):
        """Final evaluation on test set."""
        
        logger.info("🧪 Final evaluation on test set...")
        
        test_results = {}
        
        for model_name, model in self.models.items():
            try:
                y_test_pred = model.predict(X_test)
                y_test_proba = model.predict_proba(X_test)[:, 1]
                
                accuracy = accuracy_score(y_test, y_test_pred)
                precision = precision_score(y_test, y_test_pred)
                recall = recall_score(y_test, y_test_pred)
                f1 = f1_score(y_test, y_test_pred)
                auc = roc_auc_score(y_test, y_test_proba)
                
                test_results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc
                }
                
                logger.info(f"🎯 {model_name} Test Results: Accuracy={accuracy:.4f}, AUC={auc:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Error evaluating {model_name}: {e}")
        
        return test_results
    
    def save_results(self, test_results):
        """Save training and test results."""
        
        try:
            # Combine validation and test results
            results_df = pd.DataFrame({
                'Model': list(self.results.keys()),
                'Val_Accuracy': [self.results[model]['accuracy'] for model in self.results.keys()],
                'Val_Precision': [self.results[model]['precision'] for model in self.results.keys()],
                'Val_Recall': [self.results[model]['recall'] for model in self.results.keys()],
                'Val_F1': [self.results[model]['f1'] for model in self.results.keys()],
                'Val_AUC': [self.results[model]['auc'] for model in self.results.keys()],
                'Test_Accuracy': [test_results.get(model, {}).get('accuracy', 0) for model in self.results.keys()],
                'Test_Precision': [test_results.get(model, {}).get('precision', 0) for model in self.results.keys()],
                'Test_Recall': [test_results.get(model, {}).get('recall', 0) for model in self.results.keys()],
                'Test_F1': [test_results.get(model, {}).get('f1', 0) for model in self.results.keys()],
                'Test_AUC': [test_results.get(model, {}).get('auc', 0) for model in self.results.keys()]
            })
            
            results_file = os.path.join(self.models_dir, "sentiment_enhanced_model_results.csv")
            results_df.to_csv(results_file, index=False)
            logger.info(f"✅ Results saved to {results_file}")
            
            # Save feature importance
            if self.feature_importance:
                importance_data = []
                for model_name, features in self.feature_importance.items():
                    for feature, importance in features.items():
                        importance_data.append({
                            'model': model_name,
                            'feature': feature,
                            'importance': importance
                        })
                
                importance_df = pd.DataFrame(importance_data)
                importance_file = os.path.join(self.models_dir, "sentiment_enhanced_feature_importance.csv")
                importance_df.to_csv(importance_file, index=False)
                logger.info(f"✅ Feature importance saved to {importance_file}")
            
        except Exception as e:
            logger.error(f"❌ Error saving results: {e}")
    
    def run_training_pipeline(self):
        """Run the complete model training pipeline."""
        
        logger.info("🚀 Starting sentiment-enhanced model training pipeline...")
        
        try:
            # 1. Load enhanced features
            df = self.load_enhanced_features()
            if df is None:
                return False
            
            # 2. Prepare features and target
            X, y, feature_cols = self.prepare_features_and_target(df)
            if X is None:
                return False
            
            # 3. Create time series splits
            X_train, X_val, X_test, y_train, y_val, y_test = self.create_time_series_split(X, y)
            if X_train is None:
                return False
            
            # 4. Scale features
            X_train_scaled, X_val_scaled, X_test_scaled, scaler = self.scale_features(X_train, X_val, X_test)
            
            # 5. Initialize models
            self.initialize_models()
            
            # 6. Train individual models
            self.train_individual_models(X_train_scaled, X_val_scaled, y_train, y_val)
            
            # 7. Create ensemble models
            self.create_ensemble_models(X_train_scaled, X_val_scaled, y_train, y_val)
            
            # 8. Final evaluation on test set
            test_results = self.evaluate_on_test_set(X_test_scaled, y_test)
            
            # 9. Save results
            self.save_results(test_results)
            
            # 10. Summary
            logger.info("📈 Training Summary:")
            logger.info(f"   📊 Models Trained: {len(self.models)}")
            logger.info(f"   🎯 Best Validation Accuracy: {max([r['accuracy'] for r in self.results.values()]):.4f}")
            logger.info(f"   🏆 Best Test Accuracy: {max([r.get('accuracy', 0) for r in test_results.values()]):.4f}")
            
            best_model = max(test_results.keys(), key=lambda x: test_results[x]['accuracy'])
            logger.info(f"   🥇 Best Model: {best_model}")
            
            logger.info("🎉 Model training pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model training pipeline failed: {e}")
            return False

def main():
    """Run the sentiment-enhanced model training."""
    trainer = SentimentEnhancedModelTrainer()
    success = trainer.run_training_pipeline()
    
    if success:
        print("\n✅ Sentiment-enhanced models trained successfully!")
        print("🎯 Models ready for deployment!")
    else:
        print("\n❌ Model training failed!")
        return False
    
    return True

if __name__ == "__main__":
    main()