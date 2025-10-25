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

## Step 2: Start the ML Server

```bash
python app.py
```

The server will:
- Start on http://localhost:5000
- Download CLIP model on first run (~400MB)
- Preload the model into memory

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
4. View accurate ML-powered search results!

## How It Works

1. User uploads image → Admin frontend
2. Frontend sends image → NestJS backend (`/medicines/search-by-image`)
3. NestJS forwards image → Python ML server (`http://localhost:5000/search`)
4. ML server:
   - Extracts CLIP embeddings from uploaded image
   - Compares with all medicine images in database
   - Calculates cosine similarity scores
   - Returns top matches (>15% similarity)
5. Results displayed with confidence scores

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

**Connection refused:**
- Ensure ML server is running on port 5000
- Check NestJS server is running on port 3000
- Verify no firewall blocking localhost connections
