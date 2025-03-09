# CNC VISION SEARCH

CNC-VisionSearch is an AI-powered search engine for CNC drawings, enabling users to search by image or text description. It leverages deep learning models (ViT & SBERT) and vector search (FAISS) to find similar mechanical drawings quickly and accurately.

This repository contains a FastAPI service for:
- Extracting image embeddings using a Vision Transformer (ViT).
- Extracting text embeddings using SBERT (Sentence-BERT).
- Storing and querying embeddings in a FAISS index for fast similarity search.

The service supports searching by images or text and includes a health-check endpoint to ensure everything is running correctly.

## Features

1. **Vision Transformer (ViT) Embeddings**  
   - Uses [google/vit-large-patch16-224-in21k](https://huggingface.co/google/vit-large-patch16-224-in21k).  
   - Processes images to produce 2048-dimensional embeddings.  
   - Normalizes embeddings before indexing or searching.

2. **Sentence-BERT (SBERT) Embeddings**  
   - Uses [`all-MiniLM-L6-v2`](https://www.sbert.net/docs/pretrained_models.html).  
   - Converts text queries into 384-dimensional vectors.  
   - Embeddings are normalized for improved similarity search.

3. **FAISS for Similarity Search**  
   - Builds two FAISS indexes (for images and text).  
   - Allows fast *k*-nearest-neighbor lookups using `IndexHNSWFlat`.  
   - The index data is saved to and loaded from `faiss_index_hnsw.faiss`.

4. **MPS / CPU Support**  
   - Uses MPS on Apple Silicon if available; otherwise defaults to CPU.  
   - FAISS runs on CPU because MPS is not supported.

5. **FastAPI Endpoints**  
   - **GET /health**: Checks FAISS index status, model availability, and device info.
   - **POST /add_drawing**: Accepts an image file and an ID, extracts the image embedding with ViT, and adds it to the FAISS index.
   - **POST /search_drawing**: Accepts an image file as a query and returns the most similar items (default top 10).
   - **GET /search_by_text**: Accepts a text query and returns the most similar items in the text index (under development).

## Setup & Installation

1. Install Tesseract
   - ***Ubuntu/Debian**
   ```bash
   sudo apt update
   sudo apt install -y tesseract-ocr
   ```
   - ***macOS (Homebrew)**
   ```bash
   brew install tesseract
   ```

2. **Install Dependencies**. Ensure you have Python 3.8+ installed:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Service**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Test the API**:
   ```bash
   curl http://localhost:8000/health
   ```

## API Usage

### Health Check
- **GET /health**: Returns system status.
   ```bash
   curl http://localhost:8000/health
   ```

### Add Vector (Image)
- **POST /add_vector**: Uploads an image and adds it to FAISS.
   ```bash
  curl -X 'POST' \
  'http://localhost:8000/add_drawing' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'id=123' \
  -F 'file=@path/to/your/image.jpg'
   ```

### Search by Image
- **POST /search_by_image**: Finds the most similar images in FAISS.
   ```bash
   curl -X 'POST' \
  'http://localhost:8000/search_drawing' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@path/to/your/query_image.jpg'
   ```

### Search by Text (Under Development)
- **GET /search_by_text**: Finds the most similar text matches in FAISS.
   ```bash
   curl -X 'GET' \
  'http://localhost:8000/search_by_text?query=your_text_query' \
  -H 'accept: application/json'
   ```

## License

This project is provided "as is" under the [MIT License](LICENSE).
