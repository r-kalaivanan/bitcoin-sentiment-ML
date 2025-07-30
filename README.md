# Bitcoin Sentiment ML 🪙📈

A comprehensive machine learning project that predicts Bitcoin's next-day price movement by combining historical price indicators and Twitter sentiment analysis. Features automated data collection, advanced feature engineering, multiple ML models, and real-time predictions.

## 🚀 Features

- **📊 Advanced Feature Engineering**: 100+ technical indicators, price patterns, and market regime features
- **🐦 Real-time Sentiment Analysis**: Twitter sentiment using VADER and engagement weighting
- **🤖 Multiple ML Models**: Random Forest, XGBoost, LightGBM, Gradient Boosting, Logistic Regression
- **📈 Comprehensive Analysis**: Correlation analysis, feature importance, and performance visualization
- **🔮 Real-time Predictions**: Live Bitcoin price direction predictions with confidence scores
- **⚙️ Production-Ready**: Modular Python scripts for deployment and automation

## 📁 Project Structure

```
bitcoin-sentiment-ml/
├── scripts/                    # Core Python modules
│   ├── feature_engineering.py  # Price data & technical indicators
│   ├── sentiment_analysis.py   # Twitter sentiment pipeline
│   ├── data_merger.py          # Data integration & feature creation
│   ├── model_trainer.py        # ML model training & evaluation
│   ├── predictor.py            # Real-time predictions
│   ├── scrape.py               # Enhanced Twitter scraping
│   └── utils.py                # Utility functions
├── data/                       # Data storage
│   ├── btc_features_enhanced.csv
│   ├── sentiment_features.csv
│   └── merged_features.csv
├── models/                     # Trained models
│   ├── best_model.pkl
│   ├── feature_scaler.pkl
│   └── model_results.csv
├── plots/                      # Visualizations
├── predictions/                # Prediction outputs
├── main.py                     # Main pipeline orchestrator
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

## 🛠️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/r-kalaivanan/bitcoin-sentiment-ML.git
cd bitcoin-sentiment-ml
```

2. **Create virtual environment**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Setup Twitter API (Optional)**
   Create a `.env` file:

```env
X_BEARER_TOKEN=your_twitter_bearer_token_here
```

## 🎯 Quick Start

### Run Complete Pipeline

```bash
python main.py --mode pipeline
```

### Train Models Only

```bash
python main.py --mode train
```

### Make Prediction Only

```bash
python main.py --mode predict
```

### Individual Components

```bash
# Feature engineering
python scripts/feature_engineering.py

# Sentiment analysis
python scripts/sentiment_analysis.py

# Model training
python scripts/model_trainer.py

# Make prediction
python scripts/predictor.py
```

## 📊 Features Created

### 🔧 Technical Indicators (40+ features)

- **Momentum**: RSI, Stochastic, Williams %R, ROC, CCI
- **Trend**: SMA/EMA (multiple periods), MACD, ADX, Aroon
- **Volatility**: Bollinger Bands, ATR, Donchian/Keltner Channels
- **Volume**: OBV, VPT, CMF, VWAP, Volume ratios

### 💰 Price Features (25+ features)

- **Returns**: 1D, 3D, 7D, 14D, 30D percentage and log returns
- **Volatility**: Rolling volatility, High-Low volatility
- **Price Patterns**: Price position, distance from MAs, crossovers
- **Candlestick**: Body size, wick analysis, gap detection

### 📅 Time Features (15+ features)

- **Calendar**: Year, month, day, quarter, day of week
- **Cyclical**: Sin/cos encoding for temporal patterns
- **Market**: Weekend indicators, specific day patterns

### 🐦 Sentiment Features (20+ features)

- **Basic Sentiment**: Positive, negative, neutral, compound scores
- **Engagement**: Like/retweet weighted sentiment
- **Temporal**: Lagged sentiment, rolling averages, momentum
- **Interactions**: Sentiment × volatility, sentiment × RSI

### 📈 Market Regime Features (10+ features)

- **Trend Strength**: Bull/bear market indicators
- **Volatility Regime**: High/low volatility periods
- **Momentum Regime**: Overbought/oversold conditions

## 🤖 Machine Learning Models

The system trains and compares multiple models:

1. **Logistic Regression** - Baseline linear model
2. **Random Forest** - Ensemble with feature importance
3. **Gradient Boosting** - Sequential ensemble learning
4. **XGBoost** - Optimized gradient boosting
5. **LightGBM** - Fast gradient boosting

### 📊 Model Selection

- **Time-series cross-validation** for robust evaluation
- **Feature selection** using Random Forest importance
- **Hyperparameter tuning** for optimal performance
- **Multiple metrics**: Accuracy, Precision, Recall, F1-score

## 📈 Performance Metrics

The system provides comprehensive evaluation:

- **Classification Metrics**: Accuracy, Precision, Recall, F1
- **Confusion Matrix**: Visual performance breakdown
- **Feature Importance**: Top contributing features
- **Time-series Validation**: Realistic backtesting
- **Confidence Scores**: Prediction reliability

## 🔮 Making Predictions

The prediction system provides:

```python
{
    'date': '2025-01-30',
    'current_price': 98750.50,
    'prediction': 'UP',
    'confidence': 0.73,
    'probability_up': 0.73,
    'probability_down': 0.27,
    'model_used': 'XGBClassifier'
}
```

## 📝 Usage Examples

### Custom Feature Engineering

```python
from scripts.feature_engineering import FeatureEngineer

engineer = FeatureEngineer(start_date='2023-01-01')
data = engineer.generate_features()
```

### Sentiment Analysis

```python
from scripts.sentiment_analysis import SentimentAnalyzer

analyzer = SentimentAnalyzer()
sentiment_data = analyzer.run_sentiment_pipeline()
```

### Model Training

```python
from scripts.model_trainer import ModelTrainer

trainer = ModelTrainer()
results = trainer.run_training_pipeline()
```

### Real-time Prediction

```python
from scripts.predictor import predict_bitcoin_price

result = predict_bitcoin_price()
print(f"Bitcoin predicted to go {result['prediction']}")
```

## 🔧 Configuration

### Model Parameters

Edit `scripts/model_trainer.py` to adjust:

- Train/validation/test split ratios
- Feature selection methods
- Model hyperparameters
- Cross-validation settings

### Data Sources

- **Price Data**: Yahoo Finance (yfinance)
- **Sentiment Data**: Twitter API v2 (tweepy)
- **Fallback**: Simulated historical sentiment

## 📊 Data Requirements

- **Minimum**: 2+ years of price data for robust training
- **Recommended**: 3+ years for better model performance
- **Features**: 100+ engineered features per day
- **Target**: Binary classification (price up/down next day)

## 🔄 Automation

For production deployment:

1. **Daily Data Update**: Run feature engineering daily
2. **Model Retraining**: Weekly/monthly model updates
3. **Prediction Schedule**: Daily predictions before market open
4. **Monitoring**: Track prediction accuracy over time

## 🚨 Disclaimer

⚠️ **Important**: This project is for educational and research purposes only.

- **Not Financial Advice**: Do not use for actual trading decisions
- **No Guarantees**: Past performance doesn't predict future results
- **High Risk**: Cryptocurrency trading involves significant risk
- **Research Only**: Use for learning ML and data science concepts

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Technical Analysis**: TA-Lib library
- **Sentiment Analysis**: VADER sentiment analyzer
- **Data Source**: Yahoo Finance, Twitter API
- **ML Libraries**: Scikit-learn, XGBoost, LightGBM

---

**Made with ❤️ for the crypto and ML community**
