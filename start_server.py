#!/usr/bin/env python3
"""Start FastAPI server for KG viz testing."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Minimal startup without metrics

# Import the function that creates the app
from web_server import main as create_app_and_run

# Just run main() which handles everything
if __name__ == "__main__":
    create_app_and_run()
