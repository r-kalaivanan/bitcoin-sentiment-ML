# scripts/model_trainer.py

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import xgboost as xgb
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Comprehensive machine learning model trainer for Bitcoin price prediction.
    Trains multiple models, performs hyperparameter tuning, and evaluates performance.
    """
    
    def __init__(self, test_size=0.2, validation_size=0.2):
        self.test_size = test_size
        self.validation_size = validation_size
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.results = {}
        
    def load_data(self, filepath='data/merged_features.csv'):
        """Load the merged dataset."""
        try:
            print(f"📊 Loading data from {filepath}...")
            self.data = pd.read_csv(filepath)
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            print(f"✅ Loaded {len(self.data)} rows with {len(self.data.columns)} columns")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
    
    def prepare_features(self, target_col='TARGET'):
        """Prepare features and target variables."""
        print("🔧 Preparing features...")
        
        # Separate features and target
        feature_cols = [col for col in self.data.columns 
                       if col not in ['Date', 'TARGET', 'TARGET_2D', 'TARGET_STRONG']]
        
        self.X = self.data[feature_cols].copy()
        self.y = self.data[target_col].copy()
        self.dates = self.data['Date'].copy()
        
        # Handle missing values
        self.X = self.X.fillna(self.X.median())
        
        # Remove infinite values
        self.X = self.X.replace([np.inf, -np.inf], np.nan)
        self.X = self.X.fillna(self.X.median())
        
        print(f"✅ Prepared {len(feature_cols)} features")
        print(f"Target distribution: {self.y.mean():.1%} up days")
        return True
    
    def create_time_based_splits(self):
        """Create time-based train/validation/test splits."""
        print("📅 Creating time-based data splits...")
        
        n_samples = len(self.data)
        
        # Calculate split indices (chronological order)
        test_start_idx = int(n_samples * (1 - self.test_size))
        val_start_idx = int(n_samples * (1 - self.test_size - self.validation_size))
        
        # Create splits
        self.train_idx = range(0, val_start_idx)
        self.val_idx = range(val_start_idx, test_start_idx)
        self.test_idx = range(test_start_idx, n_samples)
        
        print(f"Train: {len(self.train_idx)} samples ({self.dates.iloc[self.train_idx[0]]} to {self.dates.iloc[self.train_idx[-1]]})")
        print(f"Validation: {len(self.val_idx)} samples ({self.dates.iloc[self.val_idx[0]]} to {self.dates.iloc[self.val_idx[-1]]})")
        print(f"Test: {len(self.test_idx)} samples ({self.dates.iloc[self.test_idx[0]]} to {self.dates.iloc[self.test_idx[-1]]})")
        
        return True
    
    def scale_features(self, scaler_type='robust'):
        """Scale features using the specified scaler."""
        print(f"📏 Scaling features using {scaler_type} scaler...")
        
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            print(f"❌ Unknown scaler type: {scaler_type}")
            return False
        
        # Fit scaler on training data only
        X_train = self.X.iloc[self.train_idx]
        scaler.fit(X_train)
        
        # Transform all data
        self.X_scaled = pd.DataFrame(
            scaler.transform(self.X),
            columns=self.X.columns,
            index=self.X.index
        )
        
        self.scalers['feature_scaler'] = scaler
        print("✅ Feature scaling complete")
        return True
    
    def select_features(self, method='rf_importance', k=50):
        """
        Select the most important features.
        
        Args:
            method: 'rf_importance', 'univariate', 'rfe', or 'all'
            k: Number of features to select
        """
        print(f"🎯 Selecting top {k} features using {method}...")
        
        X_train = self.X_scaled.iloc[self.train_idx]
        y_train = self.y.iloc[self.train_idx]
        
        if method == 'rf_importance':
            # Use Random Forest feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            selected_features = feature_importance.head(k)['feature'].tolist()
            
        elif method == 'univariate':
            # Use univariate statistical test
            selector = SelectKBest(score_func=f_classif, k=k)
            selector.fit(X_train, y_train)
            selected_features = X_train.columns[selector.get_support()].tolist()
            self.feature_selectors['univariate'] = selector
            
        elif method == 'rfe':
            # Use Recursive Feature Elimination
            rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            selector = RFE(estimator=rf, n_features_to_select=k)
            selector.fit(X_train, y_train)
            selected_features = X_train.columns[selector.get_support()].tolist()
            self.feature_selectors['rfe'] = selector
            
        elif method == 'all':
            selected_features = self.X_scaled.columns.tolist()
        
        else:
            print(f"❌ Unknown feature selection method: {method}")
            return False
        
        # Update scaled features to only include selected ones
        self.X_scaled = self.X_scaled[selected_features]
        self.selected_features = selected_features
        
        print(f"✅ Selected {len(selected_features)} features")
        return True
    
    def train_models(self):
        """Train multiple machine learning models."""
        print("🤖 Training machine learning models...")
        
        # Prepare training data
        X_train = self.X_scaled.iloc[self.train_idx]
        y_train = self.y.iloc[self.train_idx]
        X_val = self.X_scaled.iloc[self.val_idx]
        y_val = self.y.iloc[self.val_idx]
        
        # Define models
        models_to_train = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
        }
        
        # Train each model
        for name, model in models_to_train.items():
            print(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                
                # Calculate metrics
                train_accuracy = accuracy_score(y_train, train_pred)
                val_accuracy = accuracy_score(y_val, val_pred)
                val_precision = precision_score(y_val, val_pred)
                val_recall = recall_score(y_val, val_pred)
                val_f1 = f1_score(y_val, val_pred)
                
                # Store model and results
                self.models[name] = model
                self.results[name] = {
                    'train_accuracy': train_accuracy,
                    'val_accuracy': val_accuracy,
                    'val_precision': val_precision,
                    'val_recall': val_recall,
                    'val_f1': val_f1,
                    'train_predictions': train_pred,
                    'val_predictions': val_pred
                }
                
                print(f"✅ {name} - Val Accuracy: {val_accuracy:.3f}, F1: {val_f1:.3f}")
                
            except Exception as e:
                print(f"❌ Error training {name}: {str(e)}")
        
        return True
    
    def evaluate_models(self):
        """Evaluate all trained models on test set."""
        print("📊 Evaluating models on test set...")
        
        X_test = self.X_scaled.iloc[self.test_idx]
        y_test = self.y.iloc[self.test_idx]
        
        test_results = []
        
        for name, model in self.models.items():
            try:
                # Make predictions
                test_pred = model.predict(X_test)
                test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, test_pred)
                precision = precision_score(y_test, test_pred)
                recall = recall_score(y_test, test_pred)
                f1 = f1_score(y_test, test_pred)
                
                # Store test results
                self.results[name].update({
                    'test_accuracy': accuracy,
                    'test_precision': precision,
                    'test_recall': recall,
                    'test_f1': f1,
                    'test_predictions': test_pred,
                    'test_probabilities': test_proba
                })
                
                test_results.append({
                    'Model': name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1': f1
                })
                
                print(f"{name:20s} - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
                
            except Exception as e:
                print(f"❌ Error evaluating {name}: {str(e)}")
        
        # Create results summary
        self.test_results_df = pd.DataFrame(test_results).sort_values('F1', ascending=False)
        
        return True
    
    def create_visualizations(self):
        """Create performance visualizations."""
        print("📈 Creating performance visualizations...")
        
        # Model comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Performance comparison
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        x_pos = np.arange(len(self.test_results_df))
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            bars = ax.bar(x_pos, self.test_results_df[metric], alpha=0.7)
            ax.set_xlabel('Models')
            ax.set_ylabel(metric)
            ax.set_title(f'Model {metric} Comparison')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(self.test_results_df['Model'], rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Confusion matrix for best model
        best_model_name = self.test_results_df.iloc[0]['Model']
        best_model_results = self.results[best_model_name]
        
        y_test = self.y.iloc[self.test_idx]
        test_pred = best_model_results['test_predictions']
        
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved to plots/ directory")
        return True
    
    def save_models(self):
        """Save trained models and results."""
        print("💾 Saving models and results...")
        
        try:
            # Save best model
            best_model_name = self.test_results_df.iloc[0]['Model']
            best_model = self.models[best_model_name]
            
            joblib.dump(best_model, f'models/{best_model_name}_best.pkl')
            joblib.dump(self.scalers['feature_scaler'], 'models/feature_scaler.pkl')
            
            # Save selected features
            pd.Series(self.selected_features).to_csv('models/selected_features.csv', index=False, header=['feature'])
            
            # Save results
            self.test_results_df.to_csv('models/model_results.csv', index=False)
            
            # Save detailed results
            detailed_results = []
            for model_name, results in self.results.items():
                row = {'model': model_name}
                row.update({k: v for k, v in results.items() if not isinstance(v, np.ndarray)})
                detailed_results.append(row)
            
            pd.DataFrame(detailed_results).to_csv('models/detailed_results.csv', index=False)
            
            print(f"✅ Best model ({best_model_name}) and artifacts saved to models/ directory")
            return True
            
        except Exception as e:
            print(f"❌ Error saving models: {str(e)}")
            return False
    
    def run_training_pipeline(self, 
                             data_file='data/merged_features.csv',
                             feature_selection_method='rf_importance',
                             n_features=50):
        """
        Run the complete model training pipeline.
        
        Args:
            data_file: Path to merged features file
            feature_selection_method: Method for feature selection
            n_features: Number of features to select
        """
        print("🚀 Starting model training pipeline...")
        
        # Load and prepare data
        if not self.load_data(data_file):
            return False
        
        if not self.prepare_features():
            return False
        
        if not self.create_time_based_splits():
            return False
        
        if not self.scale_features():
            return False
        
        if not self.select_features(method=feature_selection_method, k=n_features):
            return False
        
        # Train and evaluate models
        if not self.train_models():
            return False
        
        if not self.evaluate_models():
            return False
        
        # Create visualizations and save results
        self.create_visualizations()
        self.save_models()
        
        print(f"🎉 Model training pipeline complete!")
        print(f"\n🏆 BEST MODEL: {self.test_results_df.iloc[0]['Model']}")
        print(f"Test Accuracy: {self.test_results_df.iloc[0]['Accuracy']:.3f}")
        print(f"Test F1 Score: {self.test_results_df.iloc[0]['F1']:.3f}")
        
        return self.test_results_df

if __name__ == "__main__":
    # Create model trainer and run pipeline
    trainer = ModelTrainer()
    results = trainer.run_training_pipeline()
    
    if results is not None:
        print(f"\n📊 MODEL TRAINING SUMMARY:")
        print(results.to_string(index=False))
    else:
        print("❌ Model training pipeline failed")
