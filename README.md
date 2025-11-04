# ML Image Search Server

This is the ML-based image search server for finding similar medicine images.

## Configuration

The server can be configured using environment variables. All configurations are centralized in `config.py`.

### Available Configuration Options

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `CLIP_MODEL` | `openai/clip-vit-base-patch32` | CLIP model to use for image embeddings |
| `DEVICE` | `cpu` or `cuda` (auto-detected) | Device to run the model on |
| `FLASK_HOST` | `0.0.0.0` | Host to bind the server to |
| `FLASK_PORT` | `5001` | Port to run the server on |
| `DEBUG` | `false` | Enable debug mode |
| `SIMILARITY_THRESHOLD` | `0.25` | Minimum similarity threshold for results |
| `MAX_RESULTS` | `5` | Maximum number of results to return |
| `BATCH_SIZE` | `64` | Batch size for processing images |
| `NESTJS_SERVER` | `http://localhost:3000` | Backend server URL |
| `MEDICINE_IMAGE_FOLDER` | `../midi-vision-server/uploads/medicines` | Path to medicine images |

## Supported Models

### 1. OpenAI CLIP (Default)
- Model: `openai/clip-vit-base-patch32`
- No authentication required
- Good general-purpose model

### 2. EVA02-CLIP (Alternative)
- Model: `EVA02-CLIP-L-14`
- Requires Hugging Face authentication
- Potentially better accuracy but needs setup

## Usage

### Starting the Server

1. **Using the main script:**
   ```bash
   python main.py
   ```

2. **Using the startup script:**
   ```bash
   ./start_server.sh
   ```

3. **With custom configuration:**
   ```bash
   export CLIP_MODEL="EVA02-CLIP-L-14"
   export FLASK_PORT=5002
   python main.py
   ```

### Using EVA02-CLIP Model

To use the EVA02-CLIP model:

1. Uncomment and set the model in `start_server.sh`:
   ```bash
   export CLIP_MODEL="EVA02-CLIP-L-14"
   ```

2. Or set it as an environment variable:
   ```bash
   export CLIP_MODEL="EVA02-CLIP-L-14"
   python main.py
   ```

3. If the model requires authentication, you may need to log in to Hugging Face:
   ```bash
   huggingface-cli login
   ```

## API Endpoints

- `GET /health` - Health check
- `GET /preload-model` - Preload the CLIP model
- `POST /rebuild-index` - Rebuild the search index
- `POST /refresh-medicines` - Refresh medicine data from backend
- `POST /search` - Search for similar medicines by image

## Error Handling

The server includes fallback mechanisms:
- If the specified model fails to load, it will fall back to `openai/clip-vit-base-patch32`
- Network errors are handled gracefully with retries
- Invalid images are skipped during processing

## Setup Instructions

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate virtual environment:**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the server:**
   ```bash
   python app.py
   ```

## API Endpoints

- `GET /health` - Check server health
- `POST /search` - Search medicines by image
- `POST /refresh-medicines` - Force refresh medicine cache and rebuild index
- `POST /rebuild-index` - Rebuild embeddings index (same as refresh-medicines)

## Supported Image Formats

- JPG/JPEG
- PNG
- GIF
- WEBP
- JFIF
- JFIF

## Git Ignore

This project includes a `.gitignore` file to prevent committing:
- Virtual environment files
- Cache and compiled files
- Model files
- Log files
- IDE/editor files
- OS generated files

## Features

- **CLIP Model**: State-of-the-art vision-language model for image understanding
- **FAISS Indexing**: Fast similarity search using Facebook AI Similarity Search library
- **Full Database Search**: Searches across all medicines in the database, not just a limited subset
- **Automatic Cache Refresh**: Integrates with NestJS server to automatically refresh when medicines change
- **Scalable Architecture**: Optimized for large datasets (tested with 10,000+ medicines)
- **Cosine Similarity**: Accurate image matching using embedding vectors
- **Flask API**: Simple REST API for image search
- **CORS Enabled**: Works with frontend applications

## Installation

### 1. Create Virtual Environment

```bash
cd d:\my\ml-image-search
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: Installing FAISS may take several minutes as it's a large library with native extensions.
If you have a CUDA-enabled GPU, you can install the GPU version for faster processing:

```bash
pip install torch torchvision faiss-gpu --index-url https://download.pytorch.org/whl/cu118
```

### 3. Configure Settings

Edit environment variables to customize:
- `DEVICE`: Set to `'cuda'` if you have GPU
- `SIMILARITY_THRESHOLD`: Adjust sensitivity (0.10 recommended for large datasets)
- `BATCH_SIZE`: Set to 64 or higher for faster processing of large datasets
- `CACHE_REFRESH_SECONDS`: Set to 1800 (30 minutes) for large datasets
- `NESTJS_SERVER`: Your NestJS server URL

## Usage

### Start the Server

```bash
python app.py
```

Server will start on `http://localhost:5000`

### API Endpoints

#### 1. Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model": "openai/clip-vit-base-patch32",
  "device": "cpu"
}
```

#### 2. Preload Model
```bash
GET /preload-model
```

Loads the CLIP model into memory (optional, happens automatically on first search).

#### 3. Search by Image
```bash
POST /search
Content-Type: multipart/form-data

Body:
  image: [image file]
```

Response:
```json
[
  {
    "id": "uuid",
    "name": "Paracetamol",
    "nameBn": "প্যারাসিটামল",
    "brand": "Napa",
    "similarity": 0.87,
    "confidence": "87.00%",
    "images": [...]
  }
]
```

#### 4. Refresh Medicine Cache
```bash
POST /refresh-medicines
```

Forces the ML service to refresh its medicine cache and rebuild the search index.
This is automatically called by the NestJS server when medicines are created, updated, or deleted.

Response:
```json
{
  "status": "success",
  "message": "Medicine cache refreshed and index rebuilt",
  "medicines_count": 15
}
```

## How It Works

1. **Image Upload**: User uploads a medicine image
2. **Embedding Extraction**: CLIP model converts image to 512-dimensional vector
3. **Database Query**: Fetches ALL medicines from NestJS server (no pagination limits)
4. **Similarity Calculation**: Compares uploaded image with each medicine image using cosine similarity with FAISS for fast search
5. **Results Ranking**: Returns top matches sorted by similarity score

## Model Information

- **Model**: openai/clip-vit-base-patch32
- **Architecture**: Vision Transformer (ViT)
- **Embedding Size**: 512 dimensions
- **Similarity Metric**: Cosine similarity
- **Threshold**: 0.10 (10% minimum similarity for large datasets)

## Integration with NestJS

The server automatically fetches medicines from your NestJS backend at `http://localhost:3000/medicines`.

The NestJS server automatically notifies the ML service when medicines are created, updated, or deleted, ensuring the search index is always up-to-date.

Make sure your NestJS server is running before starting the ML server.

## Performance Optimization for Large Datasets

### Batch Processing
- Processes images in batches of 64 for memory efficiency
- Adjustable via `BATCH_SIZE` environment variable

### Caching Strategy
- Embeddings cached to disk for persistence
- Medicine data cached for 30 minutes to reduce API calls
- Automatic cache refresh to keep data current
- Immediate refresh when medicines change

### FAISS Indexing
- Vector similarity search optimized for large datasets
- Sub-second search times even with 10,000+ medicines
- Disk-persistent indexes for fast startup

## Troubleshooting

### Out of Memory Error
- Reduce batch size or use CPU instead of GPU
- Close other applications to free up memory

### Slow Performance
- Use GPU by setting `DEVICE = 'cuda'` in environment variables
- Reduce `MAX_RESULTS` to process fewer comparisons
- FAISS library provides fast similarity search; without it, searches fall back to slower numpy implementation

### Model Download Issues
- Ensure stable internet connection
- Models are downloaded to `~/.cache/huggingface/`

## Performance Tips

1. **First Run**: Model download and loading takes 2-5 minutes
2. **Subsequent Runs**: Model loads from cache in ~10 seconds
3. **Search Speed**: ~100-500ms for 10,000+ medicines (with GPU)
4. **GPU Acceleration**: 5-10x faster with CUDA GPU
5. **FAISS Indexing**: Provides 10-50x faster similarity search compared to brute-force approach
6. **Full Database Search**: Searches across all medicines in your database for comprehensive results
7. **Batch Processing**: Efficiently handles large datasets with optimized memory usage
8. **Automatic Updates**: Search index automatically refreshes when medicines change

## Requirements

- Python 3.8+
- 4GB+ RAM (8GB+ recommended for large datasets)
- Internet connection (for first-time model download)
- Windows/Linux/Mac compatible

## License

MIT License