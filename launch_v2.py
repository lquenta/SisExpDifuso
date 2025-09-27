#!/usr/bin/env python3
"""
Launcher para la versión moderna del Sistema Experto Difuso.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the modern application components
import panel as pn
from core.schema import create_example_config
from fuzzy_io.loader import FuzzyConfigLoader
from app.views_v2 import ModernFuzzyPanel

def main():
    """Main function to run the modern fuzzy panel application."""
    # Configure Panel with modern extensions
    pn.extension(
        'tabulator', 
        'plotly', 
        'ace', 
        sizing_mode='stretch_width',
        notifications=True
    )
    
    # Set theme
    pn.config.theme = 'default'
    
    # Create the modern application
    app = ModernFuzzyPanel()
    
    # Serve the application with better configuration
    pn.serve(
        app.get_layout(), 
        show=True, 
        port=5007,  # Different port to avoid conflicts
        allow_websocket_origin=["*"],
        title="Sistema Experto Difuso - Versión Moderna",
        autoreload=True
    )

if __name__ == "__main__":
    main()

