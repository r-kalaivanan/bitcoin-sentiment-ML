#!/usr/bin/env python3
"""
Bitcoin Sentiment ML Dashboard
Streamlit Cloud optimized launch script
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Configure page first
st.set_page_config(
    page_title="Bitcoin Sentiment ML Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add the scripts directory to Python path
project_root = Path(__file__).parent
scripts_dir = project_root / "scripts"
sys.path.insert(0, str(scripts_dir))

def main():
    """Main function to launch the dashboard."""
    try:
        # Import and run the dashboard
        from dashboard import main as dashboard_main
        
        # Run the dashboard
        dashboard_main()
        
    except ImportError as e:
        st.error(f"❌ Import Error: {e}")
        st.info("🔧 Make sure all required packages are installed from requirements.txt")
        st.code("pip install -r requirements.txt")
    except Exception as e:
        st.error(f"❌ Application Error: {e}")
        st.info("🔍 Check the console for detailed error information")
        st.exception(e)

if __name__ == "__main__":
    main()