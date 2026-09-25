"""
Streamlit Web App Launcher Helper
Allows running the web app cleanly via `python run_app.py` or double-clicking `run_app.bat`.
"""

import sys
import subprocess

if __name__ == '__main__':
    print("=" * 60)
    print("  🛣️ Launching YOLO Pothole Detection Web Dashboard")
    print("=" * 60)
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
