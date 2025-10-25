"""
Configuration for ML Image Search Server
"""

# CLIP Model Configuration
MODEL_NAME = 'openai/clip-vit-base-patch32'
DEVICE = 'cpu'  # Change to 'cuda' if you have GPU

# Server Configuration
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
DEBUG = False  # Set to False for production

# Search Configuration
SIMILARITY_THRESHOLD = 0.10  # Lowered for better sensitivity with large datasets
MAX_RESULTS = 5  # Maximum number of results to return (most accurate 5)

# Performance Configuration for Large Datasets
BATCH_SIZE = 64  # Increased batch size for faster processing of large datasets
CACHE_REFRESH_SECONDS = 1800  # 30 minutes - longer cache for large datasets

# NestJS Server Configuration
NESTJS_SERVER = 'http://localhost:3000'
MEDICINES_ENDPOINT = '/medicines'

# Upload Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'jfif'}