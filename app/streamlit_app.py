# streamlit_app.py

import sys
import os

# Ensure the app folder is accessible
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app import app

if __name__ == "__main__":
    app.main()
