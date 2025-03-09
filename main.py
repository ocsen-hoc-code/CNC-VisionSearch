from fastapi import FastAPI, UploadFile, File, Form
import faiss
import numpy as np
import torch
import logging
import sqlite3
import io
import os
import pytesseract
from transformers import ViTImageProcessor, ViTModel
from PIL import Image

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Automatically select the best available device (CUDA, MPS, or CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use CUDA for NVIDIA GPUs
    logger.info(f"🚀 Using CUDA (GPU) - {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Use MPS for Apple Silicon (Mac)
    logger.info("🚀 Using MPS (Mac GPU)")
else:
    device = torch.device("cpu")  # Default to CPU
    logger.info("🚀 Using CPU")

# FAISS Parameters
D_IMAGE = 2048  # Vector size after concatenating Mean + Max Pooling
M = 32  # Number of neighbors for HNSW
efSearch = 300  # Number of candidates during search (higher = more accurate)
INDEX_FILE = "faiss_index_hnsw.faiss"
DB_FILE = "id_mapping.db"

# Initialize SQLite Database
def initialize_database():
    """ Creates or connects to an SQLite database to store image metadata. """
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS id_mapping (
            str_id TEXT PRIMARY KEY, 
            faiss_id INTEGER UNIQUE,
            text_content TEXT
        )
        """
    )
    conn.commit()
    return conn, cur

conn, cur = initialize_database()

# Load FAISS Index
try:
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
        logger.info("✅ FAISS HNSW index loaded from file!")
    else:
        index = faiss.IndexHNSWFlat(D_IMAGE, M)
        index.hnsw.efSearch = efSearch  # Improve search accuracy
        index = faiss.IndexIDMap(index)
        logger.info("🆕 New FAISS HNSW index initialized!")
except Exception as e:
    logger.error(f"Error loading FAISS index: {e}")

# Load ViT-Large Model
logger.info("🚀 Loading ViT-Large model...")
vit_model = ViTModel.from_pretrained('google/vit-large-patch16-224-in21k').to(device)
vit_processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224-in21k')
logger.info("✅ ViT-Large model loaded successfully!")

# Function to Extract Image Features
def extract_vit_embedding(image_bytes):
    """ Extracts image embeddings using ViT-Large with Mean + Max Pooling. """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = vit_processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = vit_model(**inputs)
            mean_features = outputs.last_hidden_state.mean(dim=1)  # Mean Pooling
            max_features = outputs.last_hidden_state.max(dim=1).values  # Max Pooling

        combined_features = torch.cat((mean_features, max_features), dim=1)  # Concatenate Mean + Max
        embedding = combined_features.squeeze(0).cpu().numpy().astype(np.float32)
        embedding /= np.linalg.norm(embedding)  # Normalize vector for better FAISS search

        return embedding
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None

# Function to Extract Text from Image using OCR
def extract_text_from_image(image_bytes):
    """ Extracts text from an image using OCR (Tesseract). """
    image = Image.open(io.BytesIO(image_bytes))
    return pytesseract.image_to_string(image).strip()

# Initialize FastAPI
app = FastAPI()

# API: Add Image Vector
@app.post("/add_drawing")
async def add_vector(id: str = Form(...), file: UploadFile = File(...)):
    """
    Adds an image to the FAISS index along with its extracted text.
    - If the image ID already exists, only the text is updated.
    """
    image_bytes = await file.read()
    image_vector = extract_vit_embedding(image_bytes)
    image_text = extract_text_from_image(image_bytes)
    
    if image_vector is None:
        return {"message": "Error processing image", "id": None}

    image_vector = np.expand_dims(image_vector, axis=0)
    
    # Check if the image ID already exists
    cur.execute("SELECT faiss_id FROM id_mapping WHERE str_id=?", (id,))
    existing_record = cur.fetchone()
    
    if existing_record:
        cur.execute("UPDATE id_mapping SET text_content = ? WHERE str_id = ?", (image_text, id))
        conn.commit()
        return {"message": "Image ID exists, text updated", "id": id}

    # Generate a new FAISS ID
    cur.execute("SELECT COALESCE(MAX(faiss_id), 0) FROM id_mapping")
    faiss_id = cur.fetchone()[0] + 1  
    cur.execute("INSERT INTO id_mapping (str_id, faiss_id, text_content) VALUES (?, ?, ?)", (id, faiss_id, image_text))
    conn.commit()

    index.add_with_ids(image_vector, np.array([faiss_id], dtype=np.int64))
    return {"message": "Vector added successfully", "id": id}

# API: Search by Image
@app.post("/search_drawing")
async def search_by_image(file: UploadFile = File(...), top_k: int = 10):
    """
    Searches for similar images in the FAISS index using image embeddings.
    Returns the top_k most similar images along with their similarity scores.
    """
    image_bytes = await file.read()
    query_vector = extract_vit_embedding(image_bytes)
    
    if query_vector is None:
        return {"message": "Error processing image"}
    
    query_vector = np.expand_dims(query_vector, axis=0)

    if index.ntotal == 0:
        return {"message": "No images in database"}

    distances, indices = index.search(query_vector, k=top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx != -1:
            cur.execute("SELECT str_id FROM id_mapping WHERE faiss_id=?", (int(idx),))
            result = cur.fetchone()
            if result:
                results.append({"id": result[0], "distance": float(dist)})
    
    return {"similar_drawings": results}

# Save FAISS Index on Shutdown
@app.on_event("shutdown")
def save_faiss_index():
    """ Saves the FAISS index and closes the database connection upon shutdown. """
    faiss.write_index(index, INDEX_FILE)
    conn.close()
    logger.info("✅ FAISS HNSW index & SQLite mapping saved!")
