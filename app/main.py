"""
Main entry point for the fuzzy panel application.
"""

import panel as pn
import sys
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from .views import FuzzyPanelApp
from ..core.schema import create_example_config
from fuzzy_io.loader import FuzzyConfigLoader


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
