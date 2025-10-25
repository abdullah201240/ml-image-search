# ML Image Search Server

This is a Python Flask server that provides image-based medicine search using the CLIP model.

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

**Note**: PyTorch installation may take some time. If you have a CUDA-enabled GPU, you can install the GPU version for faster processing:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Configure Settings

Edit `config.py` to customize:
- `DEVICE`: Set to `'cuda'` if you have GPU
- `SIMILARITY_THRESHOLD`: Adjust sensitivity (0.1-0.3 recommended)
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

## How It Works

1. **Image Upload**: User uploads a medicine image
2. **Embedding Extraction**: CLIP model converts image to 512-dimensional vector
3. **Database Query**: Fetches all medicines from NestJS server
4. **Similarity Calculation**: Compares uploaded image with each medicine image using cosine similarity
5. **Results Ranking**: Returns top matches sorted by similarity score

## Model Information

- **Model**: openai/clip-vit-base-patch32
- **Architecture**: Vision Transformer (ViT)
- **Embedding Size**: 512 dimensions
- **Similarity Metric**: Cosine similarity
- **Threshold**: 0.15 (15% minimum similarity)

## Integration with NestJS

The server automatically fetches medicines from your NestJS backend at `http://localhost:3000/medicines`.

Make sure your NestJS server is running before starting the ML server.

## Troubleshooting

### Out of Memory Error
- Reduce batch size or use CPU instead of GPU
- Close other applications to free up memory

### Slow Performance
- Use GPU by setting `DEVICE = 'cuda'` in config.py
- Reduce `MAX_RESULTS` to process fewer comparisons

### Model Download Issues
- Ensure stable internet connection
- Models are downloaded to `~/.cache/huggingface/`

## Performance Tips

1. **First Run**: Model download and loading takes 2-5 minutes
2. **Subsequent Runs**: Model loads from cache in ~10 seconds
3. **Search Speed**: ~1-3 seconds for 10-20 medicines (CPU)
4. **GPU Acceleration**: 5-10x faster with CUDA GPU

## Requirements

- Python 3.8+
- 4GB+ RAM (8GB+ recommended)
- Internet connection (for first-time model download)
- Windows/Linux/Mac compatible

## License

MIT License
