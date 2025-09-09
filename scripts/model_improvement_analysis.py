#!/usr/bin/env python3
"""
Model Accuracy Analysis & Improvement Recommendations
Comprehensive evaluation of current model performance and enhancement strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve, validation_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelAnalyzer:
    """Analyze model performance and suggest improvements."""
    
    def __init__(self):
        self.models_dir = "models/"
        self.data_dir = "data/"
        
    def load_results(self):
        """Load model results and performance metrics."""
        try:
            self.model_results = pd.read_csv(f"{self.models_dir}model_results.csv")
            self.detailed_results = pd.read_csv(f"{self.models_dir}detailed_results.csv")
            self.selected_features = pd.read_csv(f"{self.models_dir}selected_features.csv")
            return True
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            return False
    
    def analyze_current_performance(self):
        """Analyze current model performance."""
        print("🔍 CURRENT MODEL PERFORMANCE ANALYSIS")
        print("=" * 50)
        
        # Sort models by test accuracy
        sorted_models = self.model_results.sort_values('Accuracy', ascending=False)
        
        print("📊 MODEL RANKINGS (by Test Accuracy):")
        print("-" * 40)
        for idx, row in sorted_models.iterrows():
            print(f"{idx+1}. {row['Model'].upper():15} | Acc: {row['Accuracy']:.3f} | "
                  f"Prec: {row['Precision']:.3f} | F1: {row['F1']:.3f}")
        
        # Analyze best model
        best_model = sorted_models.iloc[0]
        print(f"\n🏆 BEST MODEL: {best_model['Model'].upper()}")
        print(f"   Accuracy: {best_model['Accuracy']:.3f} (53.3%)")
        print(f"   F1 Score: {best_model['F1']:.3f}")
        
        # Performance assessment
        accuracy = best_model['Accuracy']
        if accuracy > 0.7:
            status = "🟢 EXCELLENT"
        elif accuracy > 0.6:
            status = "🟡 GOOD"
        elif accuracy > 0.5:
            status = "🟠 MODERATE"
        else:
            status = "🔴 POOR"
        
        print(f"\n📈 PERFORMANCE STATUS: {status}")
        return best_model
    
    def identify_issues(self):
        """Identify specific issues with current models."""
        print("\n🔍 PERFORMANCE ISSUE ANALYSIS")
        print("=" * 50)
        
        issues = []
        improvements = []
        
        # Issue 1: Low overall accuracy
        if self.model_results['Accuracy'].max() < 0.6:
            issues.append("❌ Low accuracy (~53%) - barely above random chance")
            improvements.append("✅ Need better feature engineering")
            improvements.append("✅ Increase dataset size")
        
        # Issue 2: Overfitting analysis
        for idx, row in self.detailed_results.iterrows():
            if row['train_accuracy'] > 0.9 and row['test_accuracy'] < 0.6:
                issues.append(f"❌ Severe overfitting in {row['model']}")
                improvements.append("✅ Add regularization")
                improvements.append("✅ Use cross-validation")
        
        # Issue 3: Low precision/recall
        avg_precision = self.model_results['Precision'].mean()
        avg_recall = self.model_results['Recall'].mean()
        
        if avg_precision < 0.6:
            issues.append(f"❌ Low precision ({avg_precision:.3f}) - many false positives")
            improvements.append("✅ Improve class balancing")
        
        if avg_recall < 0.5:
            issues.append(f"❌ Low recall ({avg_recall:.3f}) - missing true positives")
            improvements.append("✅ Better threshold tuning")
        
        # Issue 4: Feature analysis
        num_features = len(self.selected_features)
        if num_features > 30:
            issues.append(f"❌ Too many features ({num_features}) - potential noise")
            improvements.append("✅ More aggressive feature selection")
        
        print("🚨 IDENTIFIED ISSUES:")
        for issue in issues:
            print(f"  {issue}")
        
        print(f"\n💡 IMPROVEMENT OPPORTUNITIES:")
        for improvement in set(improvements):  # Remove duplicates
            print(f"  {improvement}")
        
        return issues, improvements
    
    def suggest_data_improvements(self):
        """Suggest data-related improvements."""
        print("\n📊 DATA IMPROVEMENT STRATEGIES")
        print("=" * 50)
        
        strategies = {
            "🎯 IMMEDIATE (Free Tier)": [
                "Collect more diverse sentiment sources (Reddit, news)",
                "Add crypto-specific indicators (Fear & Greed Index)",
                "Include social media engagement metrics",
                "Add market microstructure features (order book data)",
                "Create time-based features (hour, day of week)"
            ],
            "🚀 MEDIUM TERM (After Twitter Quota Reset)": [
                "Strategic Twitter data collection (high-impact events)",
                "Real-time sentiment during market volatility",
                "Influencer tweet sentiment weighting",
                "Multi-language sentiment analysis"
            ],
            "💰 LONG TERM (If Budget Allows)": [
                "Premium API access for more tweets",
                "Professional market data feeds",
                "Alternative data sources (Telegram, Discord)",
                "High-frequency trading signals"
            ]
        }
        
        for category, items in strategies.items():
            print(f"\n{category}:")
            for item in items:
                print(f"  • {item}")
    
    def suggest_model_improvements(self):
        """Suggest model architecture improvements."""
        print("\n🤖 MODEL IMPROVEMENT STRATEGIES")
        print("=" * 50)
        
        print("🔧 IMMEDIATE IMPROVEMENTS:")
        print("  • Ensemble methods (combine top 3 models)")
        print("  • Hyperparameter optimization with GridSearch")
        print("  • Cross-validation with time series splits")
        print("  • Feature importance analysis and selection")
        print("  • Class balancing (SMOTE, class weights)")
        
        print("\n⚡ ADVANCED TECHNIQUES:")
        print("  • Deep learning models (LSTM, Transformer)")
        print("  • Multi-task learning (price + direction)")
        print("  • Online learning for real-time adaptation")
        print("  • Stacking/blending multiple algorithms")
        print("  • Time series specific models (ARIMA-ML hybrid)")
        
        print("\n🎯 TARGET METRICS:")
        print("  • Accuracy: 65%+ (current: 53%)")
        print("  • F1 Score: 0.60+ (current: 0.50)")
        print("  • Precision: 60%+ (reduce false positives)")
        print("  • Recall: 55%+ (catch more true positives)")
    
    def create_improvement_roadmap(self):
        """Create a practical improvement roadmap."""
        print("\n🗺️ MODEL IMPROVEMENT ROADMAP")
        print("=" * 50)
        
        roadmap = {
            "Week 1-2 (FREE TIER OPTIMIZATION)": [
                "✅ Implement ensemble model combining top 3 algorithms",
                "✅ Add Reddit sentiment data to existing features",
                "✅ Hyperparameter tuning with current data",
                "✅ Cross-validation improvement",
                "Target: 55-58% accuracy"
            ],
            "Week 3-4 (FEATURE ENGINEERING)": [
                "✅ Create interaction features (sentiment × technical indicators)",
                "✅ Add time-based features (volatility regimes)",
                "✅ Implement feature selection algorithms",
                "✅ Add market regime detection",
                "Target: 58-62% accuracy"
            ],
            "Month 2 (ADVANCED MODELING)": [
                "✅ Implement LSTM for time series patterns",
                "✅ Multi-target prediction (price + direction + volatility)",
                "✅ Online learning for real-time adaptation",
                "✅ Custom loss functions for trading objectives",
                "Target: 62-67% accuracy"
            ],
            "Month 3+ (PRODUCTION OPTIMIZATION)": [
                "✅ Real-time prediction pipeline",
                "✅ Model monitoring and retraining",
                "✅ A/B testing different strategies",
                "✅ Risk management integration",
                "Target: 67%+ accuracy"
            ]
        }
        
        for phase, tasks in roadmap.items():
            print(f"\n📅 {phase}:")
            for task in tasks:
                print(f"    {task}")
    
    def estimate_improvement_potential(self):
        """Estimate potential accuracy improvements."""
        print("\n📈 IMPROVEMENT POTENTIAL ANALYSIS")
        print("=" * 50)
        
        current_acc = self.model_results['Accuracy'].max()
        
        improvements = {
            "Ensemble Methods": 0.03,
            "Better Feature Engineering": 0.05,
            "Hyperparameter Tuning": 0.02,
            "More Diverse Data": 0.04,
            "Deep Learning": 0.06,
            "Class Balancing": 0.03
        }
        
        print("🎯 ESTIMATED ACCURACY GAINS:")
        cumulative = current_acc
        for method, gain in improvements.items():
            cumulative += gain
            print(f"  {method:25} +{gain:.1%} → {cumulative:.1%}")
        
        total_potential = cumulative - current_acc
        print(f"\n🚀 TOTAL POTENTIAL IMPROVEMENT: +{total_potential:.1%}")
        print(f"🎯 TARGET ACCURACY: {cumulative:.1%}")
        
        if cumulative > 0.7:
            print("✅ ACHIEVABLE: Professional-grade accuracy")
        elif cumulative > 0.65:
            print("✅ REALISTIC: Strong predictive performance")
        else:
            print("⚠️ CHALLENGING: Requires significant effort")
    
    def run_complete_analysis(self):
        """Run complete model improvement analysis."""
        if not self.load_results():
            return False
        
        print("🤖 BITCOIN SENTIMENT ML - MODEL IMPROVEMENT ANALYSIS")
        print("=" * 60)
        
        # Analyze current performance
        best_model = self.analyze_current_performance()
        
        # Identify issues
        issues, improvements = self.identify_issues()
        
        # Suggest improvements
        self.suggest_data_improvements()
        self.suggest_model_improvements()
        
        # Create roadmap
        self.create_improvement_roadmap()
        
        # Estimate potential
        self.estimate_improvement_potential()
        
        print("\n🎉 ANALYSIS COMPLETE!")
        print("💡 Focus on Week 1-2 improvements first for quick wins!")
        
        return True

if __name__ == "__main__":
    analyzer = ModelAnalyzer()
    analyzer.run_complete_analysis()
