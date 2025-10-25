# Quick Setup Guide

## Step 1: Install Python Dependencies

Open a new terminal in this directory and run:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies (this may take 5-10 minutes)
pip install -r requirements.txt
```

**Note:** Installing FAISS may take several minutes as it's a large library with native extensions.

## Step 2: Start the ML Server

```bash
python app.py
```

The server will:
- Start on http://localhost:5000
- Download CLIP model on first run (~400MB)
- Preload the model into memory
- Fetch ALL medicines from the NestJS server (no pagination limits)
- Optimize for large datasets with batch processing (64 items per batch)
- Automatically refresh when medicines change

## Step 3: Start NestJS Server

In another terminal:

```bash
cd d:\my\midi-vision-server
npm run start:dev
```

## Step 4: Start Admin Frontend

In another terminal:

```bash
cd d:\my\admin
npm run dev
```

## Step 5: Test Image Search

1. Go to http://localhost:3001 (admin dashboard)
2. Click the "Click Here to Search by Image" button
3. Upload a medicine image
4. View accurate ML-powered search results across your FULL medicine database!

## How It Works

1. User uploads image → Admin frontend
2. Frontend sends image → NestJS backend (`/medicines/search-by-image`)
3. NestJS forwards image → Python ML server (`http://localhost:5000/search`)
4. ML server:
   - Extracts CLIP embeddings from uploaded image
   - Fetches ALL medicines from NestJS server (no pagination limits)
   - Compares with all medicine images in database
   - Calculates cosine similarity scores using FAISS for fast similarity search
   - Returns top matches (>10% similarity)
5. Results displayed with confidence scores

## Automatic Cache Refresh

The ML service now automatically refreshes its medicine cache when medicines are:
- Created (new medicines added)
- Updated (existing medicines modified)
- Deleted (medicines removed)

This ensures that search results are always up-to-date without requiring manual refresh or waiting for cache expiration.

## Performance Optimizations for Large Datasets

### Batch Processing
- Images processed in batches of 64 for memory efficiency
- Reduces memory usage while maintaining performance
- Configurable via `BATCH_SIZE` environment variable

### Caching Strategy
- Medicine data cached for 30 minutes to reduce API calls
- Embeddings saved to disk for persistence
- Automatic cache refresh keeps data current
- Immediate refresh when medicines change

### FAISS Indexing
- Vector similarity search optimized for large datasets
- Sub-second search times even with 10,000+ medicines
- Disk-persistent indexes for fast startup

## Environment Variables

You can customize the ML server behavior with these environment variables:

- `DEVICE`: Set to 'cuda' for GPU acceleration
- `BATCH_SIZE`: Increase for faster processing (default: 64)
- `CACHE_REFRESH_SECONDS`: How often to refresh medicine data (default: 1800)
- `SIMILARITY_THRESHOLD`: Minimum similarity score (default: 0.10)
- `MAX_RESULTS`: Number of results to return (default: 5)

## Troubleshooting

**ML Server won't start:**
- Make sure virtual environment is activated
- Check Python version (3.8+ required)
- Reinstall dependencies: `pip install -r requirements.txt --upgrade`

**No search results:**
- Check if medicines have images in database
- Verify images exist in `d:\my\midi-vision-server\uploads\medicines\`
- Lower SIMILARITY_THRESHOLD in config.py (e.g., 0.10)

**Slow searches:**
- First search takes longer (model loading)
- Consider using GPU (set DEVICE='cuda' in config.py)
- FAISS is used for fast similarity search; without it, searches fall back to slower numpy implementation

**Connection refused:**
- Ensure ML server is running on port 5000
- Check NestJS server is running on port 3000
- Verify no firewall blocking localhost connections

**Incomplete search results:**
- The ML service now fetches ALL medicines from your database, not just the first 10
- If you have many medicines, the initial indexing might take longer but will cover your entire database

**Memory issues with large datasets:**
- Reduce BATCH_SIZE if running out of memory
- Ensure at least 8GB RAM for 10,000+ medicines
- Consider using a machine with more RAM for very large datasets

**Search results not updating after adding new medicines:**
- The ML service should automatically refresh when medicines change
- If automatic refresh fails, you can manually trigger it with:
  ```bash
  curl -X POST http://localhost:5000/refresh-medicines
  ```
- Check NestJS server logs for any errors in notifying the ML service