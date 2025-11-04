"""
Configuration for ML Image Search Server
All configurations can be overridden via environment variables
"""

import os

# CLIP Model Configuration
# Default to openai model which doesn't require authentication
MODEL_NAME = os.getenv('CLIP_MODEL', 'openai/clip-vit-base-patch32')

# Alternative models that can be used
# MODEL_NAME = os.getenv('CLIP_MODEL', 'EVA02-CLIP-L-14')  # Requires authentication

# Device Configuration
def get_default_device():
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'

DEVICE = os.getenv('DEVICE', get_default_device())

# Server Configuration
FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
FLASK_PORT = int(os.getenv('FLASK_PORT', 5001))
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'

# Search Configuration
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', 0.25))  # Increased from 0.10 to 0.25 (25%)
MAX_RESULTS = int(os.getenv('MAX_RESULTS', 5))

# Performance Configuration for Large Datasets
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 64))  # Increased batch size for faster processing of large datasets
CACHE_REFRESH_SECONDS = int(os.getenv('CACHE_REFRESH_SECONDS', 1800))  # 30 minutes - longer cache for large datasets

# NestJS Server Configuration
NESTJS_SERVER = os.getenv('NESTJS_SERVER', 'http://localhost:3000')
MEDICINES_ENDPOINT = os.getenv('MEDICINES_ENDPOINT', '/medicines')

# Upload Configuration
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads/query')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'jfif'}

# Cache Files Configuration
EMBEDDINGS_FILE = os.getenv('EMBEDDINGS_FILE', 'cache/medicine_embeddings.npy')
METADATA_FILE = os.getenv('METADATA_FILE', 'cache/medicine_metadata.json')
FAISS_INDEX_FILE = os.getenv('FAISS_INDEX_FILE', 'cache/faiss.index')

# Medicine Image Folder (absolute path to medicine images)
MEDICINE_IMAGE_FOLDER = os.getenv('MEDICINE_IMAGE_FOLDER', os.path.abspath('../midi-vision-server/uploads/medicines'))