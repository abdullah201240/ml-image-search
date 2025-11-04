#!/usr/bin/env python3
"""
Main entry point for the ML Image Search Server
This file makes it easy to start the server with different configurations
"""

import os
import sys
import logging

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the Flask app and configurations from app.py and config.py
from app import app, logger
from config import FLASK_HOST, FLASK_PORT, DEBUG

if __name__ == '__main__':
    # Run the Flask application with the configured host and port
    logger.info("Starting ML Image Search Server from main.py")
    logger.info(f"Server: http://{FLASK_HOST}:{FLASK_PORT}")
    logger.info(f"Debug: {DEBUG}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=DEBUG, threaded=True)