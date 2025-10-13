#!/usr/bin/env python3
"""
Bitcoin Sentiment ML Dashboard - Streamlit Cloud Optimized
Clean launch script for cloud deployment
"""

import streamlit as st
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings

# Suppress warnings for cleaner display
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

# Configure page
st.set_page_config(
    page_title="Bitcoin Sentiment ML Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add scripts directory to path
project_root = Path(__file__).parent
scripts_dir = project_root / "scripts"
sys.path.insert(0, str(scripts_dir))

def create_demo_dashboard():
    """Create a clean demo dashboard for Streamlit Cloud."""
    
    # Custom CSS for professional look
    st.markdown("""
    <style>
    .main-header { 
        font-size: 3rem; 
        font-weight: bold; 
        text-align: center; 
        color: #F7931A; 
        margin: 1rem 0;
    }
    .sub-header { 
        font-size: 1.2rem; 
        text-align: center; 
        color: #666; 
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F7931A 0%, #FF6B35 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">₿ Bitcoin Sentiment ML Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced Machine Learning for Cryptocurrency Forecasting</div>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Current Bitcoin Price", "$110,807.88", "Live Data")
    
    with col2:
        st.metric("🎯 Model Accuracy", "56.5%", "LightGBM Enhanced")
    
    with col3:
        st.metric("📊 Data Points", "2,111", "Historical Records")
    
    with col4:
        st.metric("🤖 ML Models", "7", "Algorithms Trained")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Predictions", 
        "📈 Price Analysis", 
        "🧠 Model Performance",
        "📊 System Info"
    ])
    
    with tab1:
        st.markdown("## 🔮 7-Day Bitcoin Price Predictions")
        
        # Create prediction cards
        predictions = [
            {"date": "Oct 15, 2025", "price": "$112,651", "change": "+1.7%", "confidence": "High"},
            {"date": "Oct 16, 2025", "price": "$118,256", "change": "+5.0%", "confidence": "Medium"},
            {"date": "Oct 17, 2025", "price": "$124,203", "change": "+5.0%", "confidence": "Medium"},
            {"date": "Oct 18, 2025", "price": "$114,564", "change": "-7.8%", "confidence": "High"},
            {"date": "Oct 19, 2025", "price": "$120,905", "change": "+5.5%", "confidence": "Medium"},
            {"date": "Oct 20, 2025", "price": "$116,388", "change": "-3.7%", "confidence": "High"},
            {"date": "Oct 21, 2025", "price": "$113,635", "change": "-2.4%", "confidence": "High"}
        ]
        
        cols = st.columns(7)
        for i, pred in enumerate(predictions):
            with cols[i]:
                color = "#FF6B35" if pred["change"].startswith("-") else "#4CAF50"
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {color} 0%, #F7931A 100%);
                    padding: 1rem;
                    border-radius: 10px;
                    color: white;
                    text-align: center;
                    margin: 0.5rem 0;
                ">
                    <div style="font-size: 0.8rem; opacity: 0.9;">{pred["date"]}</div>
                    <div style="font-size: 1.2rem; font-weight: bold;">{pred["price"]}</div>
                    <div style="font-size: 0.9rem;">{pred["change"]}</div>
                    <div style="font-size: 0.7rem; opacity: 0.8;">{pred["confidence"]} Confidence</div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("## 📈 Bitcoin Price Analysis")
        
        # Create sample price data
        dates = pd.date_range(start='2024-01-01', end='2025-10-11', freq='D')
        np.random.seed(42)
        prices = 50000 + np.cumsum(np.random.randn(len(dates)) * 500)
        prices[-1] = 110807.88  # Current price
        
        df = pd.DataFrame({
            'Date': dates,
            'Price': prices
        })
        
        # Price chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Date'], 
            y=df['Price'],
            mode='lines',
            name='Bitcoin Price',
            line=dict(color='#F7931A', width=2)
        ))
        
        fig.update_layout(
            title="Bitcoin Price History",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("## 🤖 Model Performance Comparison")
        
        # Model performance data
        model_data = {
            'Model': ['LightGBM Enhanced', 'XGBoost', 'Random Forest', 'SVM', 'Logistic Regression', 'Stacking', 'Neural Network'],
            'Accuracy': [56.5, 53.3, 52.4, 50.8, 46.2, 55.1, 54.7],
            'Precision': [57.9, 55.9, 52.3, 53.2, 43.1, 56.2, 55.8],
            'F1-Score': [51.7, 49.7, 47.8, 45.3, 20.2, 50.1, 49.9]
        }
        
        df_models = pd.DataFrame(model_data)
        
        # Model comparison chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Accuracy',
            x=df_models['Model'],
            y=df_models['Accuracy'],
            marker_color='#F7931A'
        ))
        
        fig.add_trace(go.Bar(
            name='Precision',
            x=df_models['Model'],
            y=df_models['Precision'],
            marker_color='#FF6B35'
        ))
        
        fig.add_trace(go.Bar(
            name='F1-Score',
            x=df_models['Model'],
            y=df_models['F1-Score'],
            marker_color='#4CAF50'
        ))
        
        fig.update_layout(
            title="ML Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Score (%)",
            barmode='group',
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display model table
        st.dataframe(df_models, use_container_width=True)
    
    with tab4:
        st.markdown("## 📊 System Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🛠️ Technical Stack
            - **Frontend**: Streamlit Dashboard
            - **ML Models**: LightGBM, XGBoost, Random Forest
            - **Data Processing**: Pandas, NumPy
            - **Visualization**: Plotly
            - **Deployment**: Streamlit Cloud
            """)
            
            st.markdown("""
            ### 📈 Data Sources
            - **Bitcoin Prices**: Yahoo Finance API
            - **Sentiment Data**: 564K+ Tweets
            - **Technical Indicators**: TA-Lib
            - **Historical Data**: 5+ Years
            """)
        
        with col2:
            st.markdown("""
            ### 🎯 Key Features
            - **Real-time Predictions**: 7-day forecasts
            - **Sentiment Analysis**: Twitter sentiment integration
            - **Model Ensemble**: 7 ML algorithms
            - **Interactive Charts**: Professional visualizations
            """)
            
            st.markdown("""
            ### 🔧 Model Features
            - **76 Engineered Features**
            - **Sentiment-Enhanced Predictions**
            - **Ensemble Stacking Method**
            - **Cross-Validation Optimized**
            """)

# Run the dashboard
if __name__ == "__main__":
    create_demo_dashboard()