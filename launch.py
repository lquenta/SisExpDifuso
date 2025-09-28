#!/usr/bin/env python3
"""
Launcher para la versión moderna simplificada del Sistema Experto Difuso.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the modern application components
import panel as pn
from app.views import ModernFuzzyPanel

def main():
    """Main function to run the modern fuzzy panel application."""
    # Configure Panel
    pn.extension(
        'tabulator', 
        sizing_mode='stretch_width',
        notifications=True
    )
    
    # Create the modern application
    app = ModernFuzzyPanel()
    
    # Serve the application
    pn.serve(
        app.get_layout(), 
        show=True, 
        port=5008,  # Different port
        allow_websocket_origin=["*"],
        title="Sistema Experto Difuso - Moderno",
        autoreload=True
    )

if __name__ == "__main__":
    main()

