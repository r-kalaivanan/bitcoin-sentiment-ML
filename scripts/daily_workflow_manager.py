import asyncio
import schedule
import time
from datetime import datetime
import logging
import sys
import os

# Add the scripts directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from enhanced_sentiment_analyzer import CryptoSentimentAnalyzer
from enhanced_model_predictor import EnhancedBitcoinPredictor
from feature_engineering import FeatureEngineer

class DailyWorkflowManager:
    def __init__(self):
        self.setup_logging()
        self.sentiment_analyzer = CryptoSentimentAnalyzer()
        self.predictor = EnhancedBitcoinPredictor()
        self.feature_engineer = FeatureEngineer()
        
    def setup_logging(self):
        """Setup logging for the workflow"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/daily_workflow.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def update_bitcoin_data(self):
        """Update Bitcoin price data"""
        try:
            self.logger.info("🔄 Updating Bitcoin price data...")
            
            # Run feature engineering to get latest Bitcoin data
            result = os.system('python scripts/feature_engineering.py')
            
            if result == 0:
                self.logger.info("✅ Bitcoin data updated successfully")
                return True
            else:
                self.logger.error("❌ Bitcoin data update failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Bitcoin data update error: {e}")
            return False
    
    def collect_sentiment_data(self):
        """Collect sentiment data from all sources"""
        try:
            self.logger.info("🔍 Collecting sentiment data...")
            
            # Collect sentiment from all sources
            sentiments = self.sentiment_analyzer.collect_all_sentiment()
            
            if sentiments:
                self.logger.info(f"✅ Collected {len(sentiments)} sentiment data points")
                
                # Get summary
                summary = self.sentiment_analyzer.get_aggregated_sentiment()
                self.logger.info(f"📊 Sentiment: {summary['sentiment_label']} ({summary['overall_sentiment']:+.3f})")
                
                return True
            else:
                self.logger.warning("⚠️ No sentiment data collected")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Sentiment collection error: {e}")
            return False
    
    def retrain_models_if_needed(self):
        """Retrain models if necessary"""
        try:
            self.logger.info("🤖 Checking if model retraining is needed...")
            
            # Check if models exist and are recent
            model_files = [
                'models/lightgbm_enhanced.pkl',
                'models/xgboost_enhanced.pkl',
                'models/random_forest_enhanced.pkl'
            ]
            
            needs_retraining = False
            
            for model_file in model_files:
                if not os.path.exists(model_file):
                    needs_retraining = True
                    break
                    
                # Check if model is older than 7 days
                file_age = time.time() - os.path.getmtime(model_file)
                if file_age > 7 * 24 * 3600:  # 7 days in seconds
                    needs_retraining = True
                    break
            
            if needs_retraining:
                self.logger.info("🎯 Retraining models with latest data...")
                success = self.predictor.train_models()
                
                if success:
                    self.logger.info("✅ Model retraining completed successfully")
                    return True
                else:
                    self.logger.error("❌ Model retraining failed")
                    return False
            else:
                self.logger.info("✅ Models are up to date")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Model retraining error: {e}")
            return False
    
    def generate_predictions(self):
        """Generate daily predictions"""
        try:
            self.logger.info("🔮 Generating daily predictions...")
            
            # Load models if not already loaded
            if not self.predictor.models:
                self.predictor.load_models()
            
            # Generate prediction
            prediction = self.predictor.generate_prediction_report()
            
            if prediction:
                self.logger.info(f"✅ Prediction generated: {prediction['prediction_label']} "
                               f"(confidence: {prediction['confidence']:.1%})")
                return True
            else:
                self.logger.error("❌ Prediction generation failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Prediction generation error: {e}")
            return False
    
    def daily_workflow(self):
        """Execute the complete daily workflow"""
        self.logger.info("🚀 Starting daily Bitcoin ML workflow")
        self.logger.info("=" * 60)
        
        workflow_results = {
            'data_update': False,
            'sentiment_collection': False,
            'model_retraining': False,
            'prediction_generation': False
        }
        
        # Step 1: Update Bitcoin data
        workflow_results['data_update'] = self.update_bitcoin_data()
        
        # Step 2: Collect sentiment data
        workflow_results['sentiment_collection'] = self.collect_sentiment_data()
        
        # Step 3: Retrain models if needed
        workflow_results['model_retraining'] = self.retrain_models_if_needed()
        
        # Step 4: Generate predictions
        workflow_results['prediction_generation'] = self.generate_predictions()
        
        # Summary
        successful_steps = sum(workflow_results.values())
        total_steps = len(workflow_results)
        
        self.logger.info("=" * 60)
        self.logger.info("📊 DAILY WORKFLOW SUMMARY")
        self.logger.info(f"Successful steps: {successful_steps}/{total_steps}")
        
        for step, success in workflow_results.items():
            status = "✅" if success else "❌"
            self.logger.info(f"  {step.replace('_', ' ').title()}: {status}")
        
        if successful_steps == total_steps:
            self.logger.info("🎉 Daily workflow completed successfully!")
        else:
            self.logger.warning("⚠️ Some steps in the daily workflow failed")
        
        self.logger.info(f"Next run scheduled for tomorrow at the same time")
        
        return workflow_results
    
    def quick_update(self):
        """Quick update for real-time monitoring"""
        try:
            self.logger.info("⚡ Running quick update...")
            
            # Quick sentiment collection
            self.sentiment_analyzer.collect_all_sentiment()
            
            # Generate quick prediction
            if not self.predictor.models:
                self.predictor.load_models()
            
            prediction = self.predictor.predict_next_movement()
            
            if prediction:
                self.logger.info(f"⚡ Quick prediction: {prediction['prediction_label']} "
                               f"({prediction['confidence']:.1%})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Quick update error: {e}")
            return False
    
    def start_scheduled_workflow(self):
        """Start the scheduled workflow"""
        self.logger.info("⏰ Setting up scheduled workflow...")
        
        # Schedule daily workflow at 9:00 AM
        schedule.every().day.at("09:00").do(self.daily_workflow)
        
        # Schedule quick updates every hour
        schedule.every().hour.do(self.quick_update)
        
        # Run initial workflow
        self.logger.info("🏃 Running initial workflow...")
        self.daily_workflow()
        
        self.logger.info("⏰ Scheduler started. Waiting for scheduled tasks...")
        self.logger.info("Daily workflow: 09:00 AM")
        self.logger.info("Quick updates: Every hour")
        self.logger.info("Press Ctrl+C to stop")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("⏹️ Workflow scheduler stopped by user")

def main():
    """Main entry point"""
    print("🚀 Bitcoin Sentiment ML - Daily Workflow Manager")
    print("=" * 60)
    
    workflow_manager = DailyWorkflowManager()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'daily':
            workflow_manager.daily_workflow()
        elif command == 'quick':
            workflow_manager.quick_update()
        elif command == 'schedule':
            workflow_manager.start_scheduled_workflow()
        elif command == 'data':
            workflow_manager.update_bitcoin_data()
        elif command == 'sentiment':
            workflow_manager.collect_sentiment_data()
        elif command == 'train':
            workflow_manager.retrain_models_if_needed()
        elif command == 'predict':
            workflow_manager.generate_predictions()
        else:
            print("❌ Unknown command. Available commands:")
            print("  daily    - Run complete daily workflow")
            print("  quick    - Run quick update")
            print("  schedule - Start scheduled workflow")
            print("  data     - Update Bitcoin data only")
            print("  sentiment- Collect sentiment data only")
            print("  train    - Retrain models only")
            print("  predict  - Generate predictions only")
    else:
        # Run daily workflow by default
        workflow_manager.daily_workflow()

if __name__ == "__main__":
    main()