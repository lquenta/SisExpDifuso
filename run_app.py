#!/usr/bin/env python3
"""
Simple script to run the fuzzy panel application.
"""

import sys
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the application
from app.views import FuzzyPanelApp
from app.main import main

if __name__ == "__main__":
    main()

