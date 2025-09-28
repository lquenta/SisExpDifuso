#!/usr/bin/env python3
"""
Launcher script for the fuzzy panel application.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the application components directly
import panel as pn
from core.schema import create_example_config
from fuzzy_io.loader import FuzzyConfigLoader
from app.views import FuzzyPanelApp

def main():
    """Main function to run the fuzzy panel application."""
    # Configure Panel
    pn.extension('tabulator', sizing_mode='stretch_width')
    
    # Create the main application
    app = FuzzyPanelApp()
    
    # Serve the application
    pn.serve(app.get_layout(), show=True, port=5006, allow_websocket_origin=["*"])

if __name__ == "__main__":
    main()
