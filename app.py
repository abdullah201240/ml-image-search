"""
ML Image Search Server using CLIP Model
Flask server for accurate medicine image similarity search
"""

import os
import torch
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from werkzeug.utils import secure_filename
import config

app = Flask(__name__)
CORS(app)

# Global model cache
clip_model = None
clip_processor = None


def load_clip_model():
    """Load CLIP model and processor (cached)"""
    global clip_model, clip_processor
    
    if clip_model is None or clip_processor is None:
        print(f"Loading CLIP model: {config.MODEL_NAME}...")
        clip_model = CLIPModel.from_pretrained(config.MODEL_NAME)
        clip_processor = CLIPProcessor.from_pretrained(config.MODEL_NAME)
        clip_model.to(config.DEVICE)
        clip_model.eval()
        print("CLIP model loaded successfully!")
    
    return clip_model, clip_processor


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS


def get_image_embedding(image_path):
    """Extract image embedding using CLIP"""
    try:
        model, processor = load_clip_model()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt").to(config.DEVICE)
        
        # Get image features
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            # Normalize the features
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()
    except Exception as e:
        print(f"Error extracting image embedding: {e}")
        return None


def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    if embedding1 is None or embedding2 is None:
        return 0.0
    
    # Cosine similarity
    similarity = np.dot(embedding1.flatten(), embedding2.flatten())
    return float(similarity)


def fetch_medicines_from_nestjs():
    """Fetch all medicines from NestJS server"""
    try:
        url = f"{config.NESTJS_SERVER}{config.MEDICINES_ENDPOINT}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Handle both direct array and paginated response
        if isinstance(data, dict) and 'data' in data:
            return data['data']
        return data
    except Exception as e:
        print(f"Error fetching medicines from NestJS: {e}")
        return []


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': config.MODEL_NAME,
        'device': config.DEVICE
    }), 200


@app.route('/preload-model', methods=['GET'])
def preload_model():
    """Preload CLIP model into memory"""
    try:
        load_clip_model()
        return jsonify({
            'status': 'success',
            'message': 'Model preloaded successfully'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/search', methods=['POST'])
def search_by_image():
    """Search medicines by uploaded image using CLIP model"""
    
    filepath = None
    
    # Check if image file is present
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Create upload directory if it doesn't exist
        os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        print(f"Processing uploaded image: {filename}")
        
        # Extract embedding from uploaded image
        query_embedding = get_image_embedding(filepath)
        
        if query_embedding is None:
            os.remove(filepath)
            return jsonify({'error': 'Failed to process image'}), 500
        
        # Fetch all medicines from NestJS
        medicines = fetch_medicines_from_nestjs()
        print(f"Found {len(medicines)} medicines in database")
        
        # Calculate similarities
        results = []
        for medicine in medicines:
            # Check if medicine has images
            if not medicine.get('images') or len(medicine['images']) == 0:
                continue
            
            # Process each image of the medicine
            medicine_similarities = []
            
            for image_url in medicine['images']:
                # Construct full image path
                # Assuming images are stored in uploads/medicines/
                image_filename = image_url.split('/')[-1]
                medicine_image_path = os.path.join('..', 'midi-vision-server', 'uploads', 'medicines', image_filename)
                
                if not os.path.exists(medicine_image_path):
                    print(f"Image not found: {medicine_image_path}")
                    continue
                
                # Get embedding for medicine image
                medicine_embedding = get_image_embedding(medicine_image_path)
                
                if medicine_embedding is not None:
                    # Calculate similarity
                    similarity = calculate_similarity(query_embedding, medicine_embedding)
                    medicine_similarities.append(similarity)
            
            # Use the highest similarity score for this medicine
            if medicine_similarities:
                max_similarity = max(medicine_similarities)
                
                # Only include if similarity is above threshold
                if max_similarity >= config.SIMILARITY_THRESHOLD:
                    results.append({
                        **medicine,
                        'similarity': max_similarity,
                        'confidence': f"{max_similarity * 100:.2f}%"
                    })
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Limit results
        results = results[:config.MAX_RESULTS]
        
        # Clean up uploaded file
        os.remove(filepath)
        
        print(f"Found {len(results)} matching medicines")
        
        return jsonify(results), 200
        
    except Exception as e:
        print(f"Error in search_by_image: {e}")
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("ML Image Search Server Starting...")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Device: {config.DEVICE}")
    print(f"Server: http://{config.FLASK_HOST}:{config.FLASK_PORT}")
    print("=" * 60)
    
    # Preload model at startup
    print("\nPreloading CLIP model...")
    load_clip_model()
    
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.DEBUG
    )
