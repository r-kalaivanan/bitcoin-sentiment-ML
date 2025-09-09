#!/usr/bin/env python3
"""
Model Accuracy Booster - Ensemble & Advanced Techniques
Target: Boost accuracy from 53% to 65-70%
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
warnings.filterwarnings('ignore')

class AccuracyBooster:
    """Advanced techniques to boost model accuracy."""
    
    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.feature_importance = {}
        
    def load_enhanced_features(self):
        """Load and create advanced features."""
        print("🔍 Loading enhanced feature set...")
        
        # Load existing features
        features_df = pd.read_csv('data/btc_features_enhanced.csv')
        
        # Create advanced technical indicators
        features_df = self.create_advanced_features(features_df)
        
        return features_df
    
    def create_advanced_features(self, df):
        """Create advanced technical and sentiment features."""
        print("⚙️ Creating advanced features...")
        
        # Map column names (handle both cases)
        price_col = 'Close' if 'Close' in df.columns else 'close'
        volume_col = 'Volume' if 'Volume' in df.columns else 'volume'
        
        # Advanced moving averages (only if not already present)
        if 'sma_50_new' not in df.columns:
            df['sma_50_new'] = df[price_col].rolling(50).mean()
        if 'sma_200_new' not in df.columns:
            df['sma_200_new'] = df[price_col].rolling(200).mean()
        if 'ema_12_new' not in df.columns:
            df['ema_12_new'] = df[price_col].ewm(span=12).mean()
        if 'ema_26_new' not in df.columns:
            df['ema_26_new'] = df[price_col].ewm(span=26).mean()
        
        # MACD (enhanced)
        if 'macd_new' not in df.columns:
            df['macd_new'] = df['ema_12_new'] - df['ema_26_new']
            df['macd_signal_new'] = df['macd_new'].ewm(span=9).mean()
            df['macd_histogram_new'] = df['macd_new'] - df['macd_signal_new']
        
        # Enhanced RSI variations
        if 'rsi_new' not in df.columns:
            delta = df[price_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_new'] = 100 - (100 / (1 + rs))
            
            # RSI momentum
            df['rsi_momentum'] = df['rsi_new'].diff()
            df['rsi_acceleration'] = df['rsi_momentum'].diff()
        
        # Enhanced Bollinger Bands
        if 'bb_position_new' not in df.columns:
            df['bb_middle_new'] = df[price_col].rolling(20).mean()
            bb_std = df[price_col].rolling(20).std()
            df['bb_upper_new'] = df['bb_middle_new'] + (bb_std * 2)
            df['bb_lower_new'] = df['bb_middle_new'] - (bb_std * 2)
            df['bb_position_new'] = (df[price_col] - df['bb_lower_new']) / (df['bb_upper_new'] - df['bb_lower_new'])
            df['bb_squeeze'] = bb_std / df['bb_middle_new']  # Volatility measure
        
        # Volume indicators (enhanced)
        if volume_col in df.columns:
            if 'volume_momentum' not in df.columns:
                df['volume_sma_new'] = df[volume_col].rolling(20).mean()
                df['volume_ratio_new'] = df[volume_col] / df['volume_sma_new']
                df['volume_momentum'] = df[volume_col].pct_change()
                df['price_volume_trend'] = (df[price_col].pct_change() * df[volume_col]).rolling(10).mean()
        
        # Price momentum features
        if 'price_momentum_3' not in df.columns:
            df['price_momentum_3'] = df[price_col].pct_change(3)
            df['price_momentum_7'] = df[price_col].pct_change(7)
            df['price_momentum_14'] = df[price_col].pct_change(14)
            df['price_acceleration'] = df[price_col].pct_change().diff()
        
        # Market regime features
        if 'volatility_regime_new' not in df.columns:
            df['volatility_regime_new'] = df[price_col].rolling(30).std() / df[price_col].rolling(30).mean()
            df['trend_strength_new'] = abs(df[price_col].rolling(20).corr(pd.Series(range(len(df)))))
            
        # Price patterns
        if 'higher_highs' not in df.columns:
            df['higher_highs'] = (df[price_col].rolling(5).max() > df[price_col].rolling(10).max().shift(5)).astype(int)
            df['lower_lows'] = (df[price_col].rolling(5).min() < df[price_col].rolling(10).min().shift(5)).astype(int)
        
        # Cross-over signals
        if 'sma_cross_new' not in df.columns:
            df['sma_cross_new'] = ((df['sma_50_new'] > df['sma_200_new']) != 
                                   (df['sma_50_new'].shift(1) > df['sma_200_new'].shift(1))).astype(int)
            df['ema_cross_new'] = ((df['ema_12_new'] > df['ema_26_new']) != 
                                   (df['ema_12_new'].shift(1) > df['ema_26_new'].shift(1))).astype(int)
        
        # Drop NaN values
        df = df.dropna()
        
        print(f"✅ Created {len(df.columns)} total features")
        return df
    
    def optimize_hyperparameters(self, X_train, y_train):
        """Optimize hyperparameters for each model."""
        print("🔧 Optimizing hyperparameters...")
        
        # LightGBM optimization
        lgb_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.15],
            'num_leaves': [31, 50, 70]
        }
        
        lgb_model = lgb.LGBMClassifier(random_state=42)
        lgb_grid = GridSearchCV(lgb_model, lgb_params, cv=3, scoring='accuracy', n_jobs=-1)
        lgb_grid.fit(X_train, y_train)
        self.models['lightgbm_optimized'] = lgb_grid.best_estimator_
        print(f"✅ LightGBM best params: {lgb_grid.best_params_}")
        
        # XGBoost optimization
        xgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.15]
        }
        
        xgb_model = xgb.XGBClassifier(random_state=42)
        xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=3, scoring='accuracy', n_jobs=-1)
        xgb_grid.fit(X_train, y_train)
        self.models['xgboost_optimized'] = xgb_grid.best_estimator_
        print(f"✅ XGBoost best params: {xgb_grid.best_params_}")
        
        # Random Forest optimization
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }
        
        rf_model = RandomForestClassifier(random_state=42)
        rf_grid = GridSearchCV(rf_model, rf_params, cv=3, scoring='accuracy', n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        self.models['random_forest_optimized'] = rf_grid.best_estimator_
        print(f"✅ Random Forest best params: {rf_grid.best_params_}")
    
    def create_ensemble_model(self, X_train, y_train):
        """Create advanced ensemble model."""
        print("🤖 Creating ensemble model...")
        
        # Voting ensemble
        voting_ensemble = VotingClassifier(
            estimators=[
                ('lgb', self.models['lightgbm_optimized']),
                ('xgb', self.models['xgboost_optimized']),
                ('rf', self.models['random_forest_optimized'])
            ],
            voting='soft'  # Use probability predictions
        )
        
        # Stacking ensemble
        stacking_ensemble = StackingClassifier(
            estimators=[
                ('lgb', self.models['lightgbm_optimized']),
                ('xgb', self.models['xgboost_optimized']),
                ('rf', self.models['random_forest_optimized'])
            ],
            final_estimator=LogisticRegression(random_state=42),
            cv=5
        )
        
        # Train both ensembles
        voting_ensemble.fit(X_train, y_train)
        stacking_ensemble.fit(X_train, y_train)
        
        self.models['voting_ensemble'] = voting_ensemble
        self.models['stacking_ensemble'] = stacking_ensemble
        
        print("✅ Ensemble models created")
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all models and find the best one."""
        print("📊 Evaluating all models...")
        
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
            print(f"{name}: {accuracy:.4f}")
        
        # Find best model
        best_model_name = max(results, key=results.get)
        best_accuracy = results[best_model_name]
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"🎯 ACCURACY: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        
        # Detailed evaluation of best model
        best_model = self.models[best_model_name]
        y_pred = best_model.predict(X_test)
        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return best_model_name, best_model, best_accuracy
    
    def save_best_model(self, model_name, model):
        """Save the best model."""
        model_path = f'models/{model_name}_boosted.pkl'
        joblib.dump(model, model_path)
        print(f"💾 Best model saved: {model_path}")
        
        # Save model info
        model_info = {
            'model_name': model_name,
            'accuracy': self.best_accuracy,
            'features_used': len(self.feature_columns),
            'training_date': pd.Timestamp.now().isoformat()
        }
        
        pd.DataFrame([model_info]).to_csv('models/boosted_model_info.csv', index=False)
    
    def run_accuracy_boost(self):
        """Main method to boost model accuracy."""
        print("🚀 ACCURACY BOOSTING PIPELINE STARTED")
        print("=" * 50)
        
        # Load enhanced features
        df = self.load_enhanced_features()
        
        # Prepare features and target
        price_col = 'Close' if 'Close' in df.columns else 'close'
        date_col = 'Date' if 'Date' in df.columns else 'date'
        
        # Exclude non-feature columns
        exclude_columns = [date_col, price_col, 'next_day_direction', 'TARGET', 'TARGET_2D', 'TARGET_STRONG']
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        self.feature_columns = feature_columns
        
        X = df[feature_columns]
        
        # Create target if not exists
        if 'TARGET' in df.columns:
            y = df['TARGET']
        elif 'next_day_direction' in df.columns:
            y = df['next_day_direction']
        else:
            # Create target: 1 if next day price is higher, 0 otherwise
            y = (df[price_col].shift(-1) > df[price_col]).astype(int)
            X = X[:-1]  # Remove last row since we can't predict it
            y = y[:-1]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training set: {len(X_train)} samples")
        print(f"📊 Test set: {len(X_test)} samples")
        print(f"📊 Features: {len(feature_columns)}")
        
        # Optimize hyperparameters
        self.optimize_hyperparameters(X_train, y_train)
        
        # Create ensemble models
        self.create_ensemble_model(X_train, y_train)
        
        # Evaluate all models
        best_model_name, best_model, best_accuracy = self.evaluate_all_models(X_test, y_test)
        self.best_accuracy = best_accuracy
        
        # Save best model
        self.save_best_model(best_model_name, best_model)
        
        print("\n✅ ACCURACY BOOSTING COMPLETE!")
        print(f"🎯 Previous accuracy: 53.26%")
        print(f"🚀 New accuracy: {best_accuracy*100:.2f}%")
        print(f"📈 Improvement: +{(best_accuracy*100 - 53.26):.2f}%")
        
        return best_model, best_accuracy

if __name__ == "__main__":
    booster = AccuracyBooster()
    best_model, accuracy = booster.run_accuracy_boost()
