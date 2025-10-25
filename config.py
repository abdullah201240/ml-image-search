"""
Configuration for ML Image Search Server
"""

# CLIP Model Configuration
MODEL_NAME = 'openai/clip-vit-base-patch32'
DEVICE = 'cpu'  # Change to 'cuda' if you have GPU

# Server Configuration
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
DEBUG = True

# Search Configuration
SIMILARITY_THRESHOLD = 0.15  # Minimum similarity score (0-1)
MAX_RESULTS = 10  # Maximum number of results to return

# NestJS Server Configuration
NESTJS_SERVER = 'http://localhost:3000'
MEDICINES_ENDPOINT = '/medicines'

# Upload Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'jfif'}
