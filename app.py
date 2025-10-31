
import os
import time
import json
import math
import logging
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image
from io import BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

import requests

import torch
from transformers import CLIPProcessor, CLIPModel

# Try to import faiss; if not available, we'll use numpy fallback
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

# ----------------------
# Config
# ----------------------
MODEL_NAME = os.getenv('CLIP_MODEL', 'openai/clip-vit-base-patch32')
DEVICE = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads/query')
MEDICINE_IMAGE_FOLDER = os.getenv('MEDICINE_IMAGE_FOLDER', '../midi-vision-server/uploads/medicines')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'webp'])
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', 0.10))  # Lowered for better sensitivity
MAX_RESULTS = int(os.getenv('MAX_RESULTS', 5))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 64))  # Increased for large datasets
EMBEDDINGS_FILE = os.getenv('EMBEDDINGS_FILE', 'cache/medicine_embeddings.npy')
METADATA_FILE = os.getenv('METADATA_FILE', 'cache/medicine_metadata.json')
FAISS_INDEX_FILE = os.getenv('FAISS_INDEX_FILE', 'cache/faiss.index')
CACHE_REFRESH_SECONDS = int(os.getenv('CACHE_REFRESH_SECONDS', 60 * 30))  # 30 minutes for large datasets
NESTJS_SERVER = os.getenv('NESTJS_SERVER', 'http://localhost:3000')
MEDICINES_ENDPOINT = os.getenv('MEDICINES_ENDPOINT', '/medicines')
FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
FLASK_PORT = int(os.getenv('FLASK_PORT', 5001))
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'

# ----------------------
# Logging setup
# ----------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('ml-image-search')

# ----------------------
# Globals
# ----------------------
app = Flask(__name__)
CORS(app)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)

clip_model = None
clip_processor = None

# Embeddings and metadata
embeddings: np.ndarray = None  # shape (N, D)
metadata: List[Dict[str, Any]] = []  # list length N, metadata mapping to embeddings rows
faiss_index = None
embeddings_loaded_at = 0
medicines_cache = []
medicines_cache_time = 0

# ----------------------
# Utilities
# ----------------------

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_clip():
    global clip_model, clip_processor
    if clip_model is None or clip_processor is None:
        logger.info(f"Loading CLIP model: {MODEL_NAME} to {DEVICE}")
        clip_model = CLIPModel.from_pretrained(MODEL_NAME)
        clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        clip_model.to(DEVICE)
        clip_model.eval()
        logger.info("CLIP loaded")
    return clip_model, clip_processor


def image_to_pil(image_input) -> Image.Image:
    """Accepts path or bytes and returns PIL Image RGB"""
    if isinstance(image_input, (bytes, bytearray)):
        return Image.open(BytesIO(image_input)).convert('RGB')
    elif isinstance(image_input, str):
        return Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        return image_input.convert('RGB')
    else:
        raise ValueError('Unsupported image input type')


def batch_get_image_embeddings(paths: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Compute embeddings for a list of image paths (or bytes) in batches.
    Returns a numpy array shape (len(paths), D) with L2-normalized vectors.
    """
    model, processor = load_clip()
    all_embeddings = []
    batch_images = []
    batch_idx_map = []

    for idx, p in enumerate(paths):
        try:
            img = image_to_pil(p)
            batch_images.append(img)
            batch_idx_map.append(idx)
        except Exception as e:
            logger.warning(f"Skipping image {p}: {e}")
            all_embeddings.append(None)
            continue

        if len(batch_images) >= batch_size:
            with torch.no_grad():
                inputs = processor(images=batch_images, return_tensors='pt', padding=True).to(DEVICE)
                feats = model.get_image_features(**inputs)
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                feats = feats.cpu().numpy()
                # append to proper indices
                for i, emb in zip(batch_idx_map, feats):
                    all_embeddings.append(emb)
            batch_images = []
            batch_idx_map = []

    # final batch
    if batch_images:
        with torch.no_grad():
            inputs = processor(images=batch_images, return_tensors='pt', padding=True).to(DEVICE)
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            feats = feats.cpu().numpy()
            for i, emb in zip(batch_idx_map, feats):
                all_embeddings.append(emb)

    # Replace None entries with zero vectors if any
    D = all_embeddings[0].shape[0] if any(e is not None for e in all_embeddings) else 512
    final = np.zeros((len(paths), D), dtype=np.float32)
    for i, v in enumerate(all_embeddings):
        if v is None:
            final[i] = np.zeros((D,), dtype=np.float32)
        else:
            final[i] = v

    return final


# ----------------------
# FAISS helpers
# ----------------------

def build_faiss_index(vecs: np.ndarray) -> faiss.Index:
    """Build a FAISS index for inner-product search. Vectors are expected normalized.
    Returns index.
    """
    if not _HAS_FAISS:
        raise RuntimeError('FAISS not available')

    d = vecs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vecs)
    return index


def faiss_search(query_vec: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    # query_vec shape (1, D)
    if faiss_index is None:
        raise RuntimeError('FAISS index not built')
    scores, idxs = faiss_index.search(query_vec.astype(np.float32), top_k)
    return idxs[0], scores[0]


# ----------------------
# Data loading & caching
# ----------------------

def fetch_medicines_from_nestjs() -> List[Dict[str, Any]]:
    global medicines_cache, medicines_cache_time
    try:
        if time.time() - medicines_cache_time < CACHE_REFRESH_SECONDS and medicines_cache:
            return medicines_cache

        # Fetch medicines with pagination to avoid memory issues
        url = f"{NESTJS_SERVER.rstrip('/')}/{MEDICINES_ENDPOINT.lstrip('/')}"
        logger.info(f"Fetching medicines from: {url}")
        
        all_medicines = []
        page = 1
        limit = 50  # Fetch in smaller batches to avoid memory issues
        total_fetched = 0
        max_total = 1000  # Safety limit to prevent infinite loops
        
        while total_fetched < max_total:
            page_url = f"{url}?page={page}&limit={limit}"
            logger.info(f"Fetching page {page} from: {page_url}")
            
            try:
                r = requests.get(page_url, timeout=15)  # Increased timeout
                r.raise_for_status()
                data = r.json()
                
                # Handle both paginated and non-paginated responses
                if isinstance(data, dict) and 'data' in data:
                    medicines = data['data']
                    total = data.get('total', len(medicines))
                    total_pages = data.get('totalPages', 1)
                    
                    logger.info(f"Page {page}: fetched {len(medicines)} medicines")
                    all_medicines.extend(medicines)
                    total_fetched += len(medicines)
                    
                    # If this is the last page, break
                    if page >= total_pages:
                        break
                else:
                    # Non-paginated response
                    medicines = data if isinstance(data, list) else []
                    logger.info(f"Fetched {len(medicines)} medicines (non-paginated)")
                    all_medicines.extend(medicines)
                    total_fetched += len(medicines)
                    break
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to fetch page {page}: {e}")
                break
            except Exception as e:
                logger.error(f"Error parsing page {page}: {e}")
                break
                
            page += 1
            
            # Safety check to prevent infinite loops
            if page > 50:  # Max 50 pages
                logger.warning("Reached maximum page limit (50)")
                break
        
        if not isinstance(all_medicines, list):
            all_medicines = []
            
        medicines_cache = all_medicines
        medicines_cache_time = time.time()
        logger.info(f"Total fetched medicines: {len(all_medicines)}")
        return all_medicines
    except Exception as e:
        logger.exception(f"Failed to fetch medicines: {e}")
        # fallback to cached if present
        if medicines_cache:
            logger.info('Using cached medicines')
            return medicines_cache
        return []


def build_embeddings_and_index(force_rebuild: bool = False):
    """Build or load embeddings and FAISS index. Saves to disk for persistence."""
    global embeddings, metadata, faiss_index, embeddings_loaded_at

    # load from disk if exists and not forcing rebuild
    if not force_rebuild and os.path.exists(EMBEDDINGS_FILE) and os.path.exists(METADATA_FILE):
        try:
            logger.info('Loading embeddings & metadata from disk')
            embeddings = np.load(EMBEDDINGS_FILE)
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            embeddings_loaded_at = time.time()
            if _HAS_FAISS and os.path.exists(FAISS_INDEX_FILE):
                try:
                    logger.info('Loading FAISS index from disk')
                    faiss_index = faiss.read_index(FAISS_INDEX_FILE)
                except Exception as e:
                    logger.warning(f'Failed to load FAISS index: {e}, rebuilding...')
                    faiss_index = build_faiss_index(embeddings)
                    if _HAS_FAISS:
                        faiss.write_index(faiss_index, FAISS_INDEX_FILE)
            elif _HAS_FAISS:
                faiss_index = build_faiss_index(embeddings)
                faiss.write_index(faiss_index, FAISS_INDEX_FILE)
            logger.info('Embeddings loaded')
            return
        except Exception as e:
            logger.exception(f'Error loading cached embeddings: {e} — rebuilding')

    # Otherwise build fresh
    logger.info('Building embeddings from medicine images')
    medicines = fetch_medicines_from_nestjs()

    rows = []
    meta = []
    image_paths = []
    mapping = []  # map image index -> (medicine index, image_url)

    for mi, med in enumerate(medicines):
        imgs = med.get('images') or []
        for image_url in imgs:
            image_filename = image_url.split('/')[-1]
            candidate_path = os.path.join(MEDICINE_IMAGE_FOLDER, image_filename)
            if not os.path.exists(candidate_path):
                logger.debug(f'Image not found: {candidate_path}')
                continue
            image_paths.append(candidate_path)
            mapping.append((mi, image_url))
            meta.append({
                'medicine_index': mi,
                'image_url': image_url,
                'image_path': candidate_path,
                'medicine': med  # full medicine object (could be heavy)
            })

    if not image_paths:
        logger.warning('No medicine images found to build embeddings')
        embeddings = np.zeros((0, 512), dtype=np.float32)
        metadata = []
        return

    vecs = batch_get_image_embeddings(image_paths, batch_size=BATCH_SIZE)

    # Normalize (should already be normalized by model but ensure float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = (vecs / norms).astype(np.float32)

    embeddings = vecs
    metadata = meta

    # save to disk
    np.save(EMBEDDINGS_FILE, embeddings)
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # build faiss index if available
    if _HAS_FAISS and embeddings.shape[0] > 0:
        faiss_index = build_faiss_index(embeddings)
        try:
            faiss.write_index(faiss_index, FAISS_INDEX_FILE)
        except Exception as e:
            logger.warning(f'Could not write FAISS index to disk: {e}')

    embeddings_loaded_at = time.time()
    logger.info('Embeddings & index build complete')


# ----------------------
# Search helpers
# ----------------------

def search_top_k(query_embedding: np.ndarray, top_k: int = MAX_RESULTS) -> List[Dict[str, Any]]:
    """Return top_k nearest medicines with similarity scores and confidence."""
    global embeddings, metadata

    if embeddings is None or embeddings.shape[0] == 0:
        return []

    # ensure query is normalized
    q = query_embedding.astype(np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return []
    q = q / q_norm
    q = q.reshape(1, -1)

    results = []

    if _HAS_FAISS and faiss_index is not None:
        idxs, scores = faiss_search(q, top_k)
        for idx, score in zip(idxs, scores):
            # score is inner product (cosine because vectors normalized)
            m = metadata[idx]
            med = m['medicine']
            results.append({
                **med,
                'matched_image': m['image_url'],
                'similarity': float(score),
                'confidence': f"{float(score) * 100:.2f}%"
            })
    else:
        # brute-force using numpy dot product
        sims = embeddings.dot(q.T).squeeze(1)
        idxs = np.argsort(-sims)[:top_k]
        for idx in idxs:
            score = float(sims[idx])
            m = metadata[idx]
            med = m['medicine']
            results.append({
                **med,
                'matched_image': m['image_url'],
                'similarity': score,
                'confidence': f"{score * 100:.2f}%"
            })

    # Filter by threshold
    filtered = [r for r in results if r['similarity'] >= SIMILARITY_THRESHOLD]
    return filtered[:top_k]


# ----------------------
# Flask routes
# ----------------------

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model': MODEL_NAME,
        'device': DEVICE,
        'embeddings_loaded_at': embeddings_loaded_at
    })


@app.route('/preload-model', methods=['GET'])
def preload_model():
    try:
        load_clip()
        return jsonify({'status': 'success', 'message': 'Model loaded'}), 200
    except Exception as e:
        logger.exception(e)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/rebuild-index', methods=['POST'])
def rebuild_index_endpoint():
    try:
        build_embeddings_and_index(force_rebuild=True)
        return jsonify({'status': 'success', 'message': 'Index rebuilt'}), 200
    except Exception as e:
        logger.exception(e)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/refresh-medicines', methods=['POST'])
def refresh_medicines():
    """Force refresh the medicine cache and rebuild embeddings index"""
    global medicines_cache, medicines_cache_time
    try:
        logger.info("Starting medicine cache refresh")
        # Clear the cache
        medicines_cache = []
        medicines_cache_time = 0
        
        # Rebuild embeddings and index
        build_embeddings_and_index(force_rebuild=True)
        
        medicines_count = len(medicines_cache) if medicines_cache else 0
        logger.info(f"Medicine cache refresh completed. Medicines count: {medicines_count}")
        
        return jsonify({
            'status': 'success', 
            'message': 'Medicine cache refreshed and index rebuilt',
            'medicines_count': medicines_count
        }), 200
    except Exception as e:
        logger.exception(f"Failed to refresh medicines: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/search', methods=['POST'])
def search_by_image():
    # Accept either uploaded file or URL (json 'image_url')
    if 'image' not in request.files and not request.json:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # get image bytes
        if 'image' in request.files:
            f = request.files['image']
            if f.filename == '' or not allowed_file(f.filename):
                return jsonify({'error': 'Invalid file'}), 400
            fn = secure_filename(f.filename)
            save_path = os.path.join(UPLOAD_FOLDER, fn)
            f.save(save_path)
            query_img_input = save_path
        else:
            body = request.get_json()
            image_url = body.get('image_url')
            if not image_url:
                return jsonify({'error': 'No image_url provided'}), 400
            r = requests.get(image_url, timeout=10)
            r.raise_for_status()
            query_img_input = r.content

        # compute embedding
        query_emb = batch_get_image_embeddings([query_img_input], batch_size=1)[0]

        if embeddings is None or embeddings.shape[0] == 0:
            # attempt to build embeddings if not present
            build_embeddings_and_index()

        results = search_top_k(query_emb, top_k=MAX_RESULTS)

        # optional: cleanup uploaded file
        if isinstance(query_img_input, str) and os.path.exists(query_img_input):
            try:
                os.remove(query_img_input)
            except Exception:
                pass

        return jsonify(results), 200
    except Exception as e:
        logger.exception(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500


# ----------------------
# Startup
# ----------------------

if __name__ == '__main__':
    logger.info('=' * 60)
    logger.info('Starting ML Image Search Server')
    logger.info(f'Model: {MODEL_NAME} | Device: {DEVICE}')
    logger.info(f'Server: http://{FLASK_HOST}:{FLASK_PORT}')
    logger.info('=' * 60)

    # Preload model and build embeddings/index in startup
    try:
        load_clip()
    except Exception:
        logger.exception('Failed to load model at startup')

    try:
        build_embeddings_and_index(force_rebuild=False)
    except Exception:
        logger.exception('Failed to build embeddings at startup')

    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=DEBUG)
