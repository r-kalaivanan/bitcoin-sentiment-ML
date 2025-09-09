#!/usr/bin/env python3
"""
Ensemble Model Implementation - Immediate Performance Boost
Combines top 3 models (LightGBM, XGBoost, Gradient Boosting) for better predictions
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
warnings.filterwarnings('ignore')

class EnsembleModelBuilder:
    """Build ensemble models for improved Bitcoin prediction accuracy."""
    
    def __init__(self):
        self.data_dir = "data/"
        self.models_dir = "models/"
        
    def load_data(self):
        """Load prepared feature data."""
        try:
            # Load feature data
            features_df = pd.read_csv(f"{self.data_dir}btc_features_enhanced.csv")
            
            # Load selected features
            selected_features = pd.read_csv(f"{self.models_dir}selected_features.csv")
            feature_names = selected_features['feature'].tolist()
            
            # Prepare features and target
            X = features_df[feature_names].fillna(0)
            y = features_df['target'].fillna(0)
            
            # Remove any remaining NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            print(f"✅ Loaded data: {len(X)} samples, {len(feature_names)} features")
            
            # Time series split (important for financial data)
            split_point = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
            
            return X_train, X_test, y_train, y_test, feature_names
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None, None, None, None, None
    
    def create_base_models(self):
        """Create optimized base models for ensemble."""
        
        # LightGBM - Best performing
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1
        )
        
        # XGBoost - Second best
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        # Gradient Boosting - Third best  
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        
        return lgb_model, xgb_model, gb_model
    
    def create_voting_ensemble(self, lgb_model, xgb_model, gb_model):
        """Create voting ensemble (simple averaging)."""
        
        voting_ensemble = VotingClassifier(
            estimators=[
                ('lightgbm', lgb_model),
                ('xgboost', xgb_model), 
                ('gradient_boosting', gb_model)
            ],
            voting='soft'  # Use probability averaging
        )
        
        return voting_ensemble
    
    def create_stacking_ensemble(self, lgb_model, xgb_model, gb_model):
        """Create stacking ensemble (meta-learner)."""
        
        stacking_ensemble = StackingClassifier(
            estimators=[
                ('lightgbm', lgb_model),
                ('xgboost', xgb_model),
                ('gradient_boosting', gb_model)
            ],
            final_estimator=LogisticRegression(random_state=42),
            cv=5
        )
        
        return stacking_ensemble
    
    def evaluate_ensemble(self, model, X_train, X_test, y_train, y_test, model_name):
        """Evaluate ensemble model performance."""
        
        print(f"\n🧪 EVALUATING {model_name.upper()}")
        print("-" * 40)
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Metrics
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred, average='weighted')
        
        print(f"Train Accuracy: {train_acc:.3f}")
        print(f"Test Accuracy:  {test_acc:.3f}")
        print(f"Test F1 Score:  {test_f1:.3f}")
        
        # Cross-validation with time series split
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='accuracy')
        
        print(f"CV Mean:        {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Overfitting check
        overfitting = train_acc - test_acc
        if overfitting > 0.1:
            print(f"⚠️ Overfitting detected: {overfitting:.3f}")
        else:
            print(f"✅ Good generalization: {overfitting:.3f}")
        
        return {
            'model_name': model_name,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'overfitting': overfitting
        }
    
    def compare_with_baseline(self, ensemble_results):
        """Compare ensemble results with baseline models."""
        
        print(f"\n📊 ENSEMBLE VS BASELINE COMPARISON")
        print("=" * 50)
        
        # Baseline results (from previous analysis)
        baseline_results = {
            'LightGBM': 0.533,
            'XGBoost': 0.516, 
            'Gradient Boosting': 0.508,
            'Random Forest': 0.457,
            'Logistic Regression': 0.462
        }
        
        print("🏆 RESULTS COMPARISON:")
        print("-" * 30)
        
        for model_name, result in ensemble_results.items():
            test_acc = result['test_accuracy']
            best_baseline = max(baseline_results.values())
            improvement = test_acc - best_baseline
            
            status = "🟢" if improvement > 0 else "🔴"
            print(f"{status} {model_name:20} {test_acc:.3f} ({improvement:+.3f})")
        
        print(f"\nBaseline Best:           {best_baseline:.3f} (LightGBM)")
        
        # Find best ensemble
        best_ensemble = max(ensemble_results.items(), key=lambda x: x[1]['test_accuracy'])
        best_name, best_result = best_ensemble
        
        print(f"🎯 BEST ENSEMBLE: {best_name}")
        print(f"   Accuracy: {best_result['test_accuracy']:.3f}")
        print(f"   Improvement: {best_result['test_accuracy'] - best_baseline:+.3f}")
        
        return best_ensemble
    
    def save_best_model(self, best_ensemble, model_object):
        """Save the best ensemble model."""
        
        model_name, results = best_ensemble
        filename = f"{self.models_dir}ensemble_{model_name.lower().replace(' ', '_')}.pkl"
        
        joblib.dump(model_object, filename)
        print(f"\n✅ Saved best model: {filename}")
        
        # Save results
        results_df = pd.DataFrame([results])
        results_df.to_csv(f"{self.models_dir}ensemble_results.csv", index=False)
        
        return filename
    
    def build_and_evaluate_ensembles(self):
        """Build and evaluate all ensemble models."""
        
        print("🤖 ENSEMBLE MODEL BUILDER - IMMEDIATE ACCURACY BOOST")
        print("=" * 60)
        
        # Load data
        X_train, X_test, y_train, y_test, feature_names = self.load_data()
        if X_train is None:
            return False
        
        # Create base models
        lgb_model, xgb_model, gb_model = self.create_base_models()
        
        # Create ensembles
        voting_ensemble = self.create_voting_ensemble(lgb_model, xgb_model, gb_model)
        stacking_ensemble = self.create_stacking_ensemble(lgb_model, xgb_model, gb_model)
        
        # Evaluate ensembles
        ensemble_results = {}
        
        # Voting ensemble
        voting_result = self.evaluate_ensemble(
            voting_ensemble, X_train, X_test, y_train, y_test, "Voting Ensemble"
        )
        ensemble_results["Voting Ensemble"] = voting_result
        
        # Stacking ensemble  
        stacking_result = self.evaluate_ensemble(
            stacking_ensemble, X_train, X_test, y_train, y_test, "Stacking Ensemble"
        )
        ensemble_results["Stacking Ensemble"] = stacking_result
        
        # Compare with baseline
        best_ensemble = self.compare_with_baseline(ensemble_results)
        
        # Save best model
        if best_ensemble[0] == "Voting Ensemble":
            self.save_best_model(best_ensemble, voting_ensemble)
        else:
            self.save_best_model(best_ensemble, stacking_ensemble)
        
        print("\n🎉 ENSEMBLE MODEL BUILDING COMPLETE!")
        print(f"💡 Achieved {best_ensemble[1]['test_accuracy']:.1%} accuracy")
        
        return True

if __name__ == "__main__":
    builder = EnsembleModelBuilder()
    builder.build_and_evaluate_ensembles()
