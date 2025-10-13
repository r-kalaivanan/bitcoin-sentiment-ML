#!/usr/bin/env python3
"""
Simple Dashboard Launch Script
Launches the Bitcoin Sentiment ML Dashboard
"""

import subprocess
import sys
import os
from pathlib import Path
import webbrowser
import time

def launch_dashboard():
    """Launch the streamlit dashboard with proper error handling."""
    print("🚀 Launching Bitcoin Sentiment ML Dashboard...")
    
    # Change to project directory
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    print(f"📁 Working directory: {project_dir}")
    
    # Check if streamlit is available
    try:
        result = subprocess.run([sys.executable, "-c", "import streamlit"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Streamlit not found. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"])
    except Exception as e:
        print(f"⚠️  Warning: {e}")
    
    # Launch dashboard
    try:
        print("🌐 Starting Streamlit server...")
        print("📊 Dashboard will open at: http://localhost:8501")
        print("🔄 Press Ctrl+C to stop the server")
        
        # Open browser after a delay
        def open_browser():
            time.sleep(3)
            webbrowser.open("http://localhost:8501")
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.start()
        
        # Run streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "scripts/dashboard.py"])
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        print("💡 Try running manually: streamlit run scripts/dashboard.py")

if __name__ == "__main__":
    launch_dashboard()