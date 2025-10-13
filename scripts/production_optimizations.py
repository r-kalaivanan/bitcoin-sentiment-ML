"""
Production optimizations for Bitcoin Sentiment ML Dashboard
Includes caching, error handling, performance improvements
"""

import streamlit as st
import functools
import time
import logging
import os
from typing import Any, Callable
import pandas as pd
import numpy as np

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductionOptimizer:
    """Production optimizations for the dashboard."""
    
    @staticmethod
    @st.cache_data(ttl=300, max_entries=50)  # Cache for 5 minutes
    def load_bitcoin_data() -> pd.DataFrame:
        """Cached data loading for Bitcoin prices."""
        try:
            if os.path.exists('data/btc_data.csv'):
                data = pd.read_csv('data/btc_data.csv')
                logger.info(f"Loaded Bitcoin data: {len(data)} records")
                return data
            else:
                logger.warning("Bitcoin data file not found, using demo data")
                return ProductionOptimizer._create_demo_data()
        except Exception as e:
            logger.error(f"Error loading Bitcoin data: {e}")
            return ProductionOptimizer._create_demo_data()
    
    @staticmethod
    @st.cache_data(ttl=600, max_entries=10)  # Cache for 10 minutes
    def load_model_results() -> pd.DataFrame:
        """Cached loading of model performance results."""
        try:
            model_files = [
                'models/sentiment_enhanced_model_results.csv',
                'models/updated_model_results.csv',
                'models/model_results.csv'
            ]
            
            for file_path in model_files:
                if os.path.exists(file_path):
                    results = pd.read_csv(file_path)
                    logger.info(f"Loaded model results from {file_path}")
                    return results
            
            # Fallback to demo data
            logger.warning("No model results found, using demo data")
            return ProductionOptimizer._create_demo_model_results()
            
        except Exception as e:
            logger.error(f"Error loading model results: {e}")
            return ProductionOptimizer._create_demo_model_results()
    
    @staticmethod
    @st.cache_resource
    def load_ml_models():
        """Cached loading of trained ML models."""
        try:
            import joblib
            model_files = [
                'models/lightgbm_sentiment_enhanced.pkl',
                'models/lightgbm_updated.pkl',
                'models/lightgbm_best.pkl'
            ]
            
            for model_file in model_files:
                if os.path.exists(model_file):
                    model = joblib.load(model_file)
                    logger.info(f"Loaded ML model from {model_file}")
                    return model, model_file
            
            logger.warning("No trained models found")
            return None, None
            
        except Exception as e:
            logger.error(f"Error loading ML models: {e}")
            return None, None
    
    @staticmethod
    def _create_demo_data() -> pd.DataFrame:
        """Create demo Bitcoin data."""
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        return pd.DataFrame({
            'Date': dates,
            'Close': np.random.normal(50000, 10000, len(dates)),
            'Volume': np.random.normal(1000000000, 200000000, len(dates)),
            'Open': np.random.normal(50000, 10000, len(dates)),
            'High': np.random.normal(52000, 10000, len(dates)),
            'Low': np.random.normal(48000, 10000, len(dates))
        })
    
    @staticmethod
    def _create_demo_model_results() -> pd.DataFrame:
        """Create demo model results."""
        return pd.DataFrame({
            'Model': ['LightGBM', 'XGBoost', 'Random Forest', 'SVM', 'Logistic Regression'],
            'Accuracy': [0.5655, 0.5326, 0.5244, 0.5082, 0.4620],
            'Precision': [0.5792, 0.5592, 0.5234, 0.5319, 0.4310],
            'Recall': [0.4674, 0.4474, 0.4423, 0.3947, 0.1316],
            'F1': [0.5171, 0.4971, 0.4781, 0.4532, 0.2016]
        })

def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor function performance."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise
    return wrapper

def safe_execute(func: Callable, fallback: Any = None, error_message: str = "Operation failed") -> Any:
    """Safely execute a function with fallback."""
    try:
        return func()
    except Exception as e:
        logger.error(f"{error_message}: {e}")
        st.error(f"⚠️ {error_message}. Using fallback data.")
        return fallback

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage."""
    try:
        # Convert numeric columns to more efficient types
        for col in df.select_dtypes(include=['float64']):
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        for col in df.select_dtypes(include=['int64']):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Convert datetime columns
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        
        logger.info(f"DataFrame optimized: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        return df
        
    except Exception as e:
        logger.error(f"Error optimizing DataFrame: {e}")
        return df

class ErrorHandler:
    """Centralized error handling for production."""
    
    @staticmethod
    def handle_data_error(error: Exception, context: str = "data operation") -> None:
        """Handle data-related errors."""
        error_msg = f"Data error in {context}: {str(error)}"
        logger.error(error_msg)
        st.error(f"⚠️ {error_msg}")
    
    @staticmethod
    def handle_model_error(error: Exception, context: str = "model operation") -> None:
        """Handle ML model errors."""
        error_msg = f"Model error in {context}: {str(error)}"
        logger.error(error_msg)
        st.warning(f"🤖 {error_msg}")
    
    @staticmethod
    def handle_visualization_error(error: Exception, context: str = "visualization") -> None:
        """Handle visualization errors."""
        error_msg = f"Visualization error in {context}: {str(error)}"
        logger.error(error_msg)
        st.info(f"📊 {error_msg}")

def setup_production_environment():
    """Setup production environment configurations."""
    
    # Set page config for production
    st.set_page_config(
        page_title="Bitcoin Sentiment ML - Production",
        page_icon="₿",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            'Get Help': 'https://github.com/r-kalaivanan/bitcoin-sentiment-ML',
            'Report a bug': 'https://github.com/r-kalaivanan/bitcoin-sentiment-ML/issues',
            'About': 'Bitcoin Sentiment ML Dashboard - Production Version'
        }
    )
    
    # Hide Streamlit footer and menu for cleaner production look
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    logger.info("Production environment setup complete")

def get_system_stats():
    """Get system performance statistics."""
    try:
        import psutil
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'process_memory': psutil.Process().memory_info().rss / 1024 / 1024  # MB
        }
    except ImportError:
        logger.warning("psutil not available for system monitoring")
        return {}

# Production health check endpoint
def health_check():
    """Health check for production deployment."""
    try:
        # Check critical components
        checks = {
            'data_available': os.path.exists('data/btc_data.csv'),
            'models_available': any(os.path.exists(f'models/{f}') for f in os.listdir('models') if f.endswith('.pkl')),
            'logs_writable': os.access('logs', os.W_OK),
        }
        
        all_healthy = all(checks.values())
        
        if all_healthy:
            logger.info("Health check passed")
            return {"status": "healthy", "checks": checks}
        else:
            logger.warning(f"Health check failed: {checks}")
            return {"status": "unhealthy", "checks": checks}
            
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {"status": "error", "error": str(e)}