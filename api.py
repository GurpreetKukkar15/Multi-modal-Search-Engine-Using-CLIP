import torch
import chromadb
from transformers import CLIPProcessor, CLIPModel
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from config import *

# --- Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Initialize FastAPI App ---
app = FastAPI()

# Add CORS middleware to allow requests from your front end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Serve Static Files ---
app.mount("/data", StaticFiles(directory="data"), name="data")

# --- Load CLIP Model (run once at startup) ---
print("Loading CLIP model for API...")
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
print("CLIP model loaded.")

# --- Initialize ChromaDB Client ---
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
try:
    collection = client.get_collection(name=COLLECTION_NAME)
    doc_count = collection.count()
    print(f"✅ Connected to ChromaDB collection '{COLLECTION_NAME}' with {doc_count} items.")
    if doc_count == 0:
        print("⚠️  WARNING: Collection is empty! Run ingest_data.py first.")
except Exception as e:
    print(f"❌ ERROR: Could not connect to collection '{COLLECTION_NAME}': {e}")
    print("   Make sure to run ingest_data.py first to create the database.")
    collection = None

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """Check if the API and database are working properly."""
    if collection is None:
        return {"status": "error", "message": "Database not connected. Run ingest_data.py first."}
    
    try:
        doc_count = collection.count()
        return {
            "status": "healthy", 
            "database": "connected",
            "documents": doc_count,
            "collection": COLLECTION_NAME
        }
    except Exception as e:
        return {"status": "error", "message": f"Database error: {str(e)}"}

# --- Search Endpoint ---
@app.get("/search")
async def search_images_api(query: str = Query(..., min_length=1), k: int = Query(5, ge=1, le=20)):
    """
    Search the image collection based on a text query.
    k: Number of results to return (1-20)
    """
    # Check if database is available
    if collection is None:
        return {"error": "Database not available. Please run ingest_data.py first."}
    
    try:
        # Generate text embedding from the query
        inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_embedding = model.get_text_features(**inputs).cpu().numpy().tolist()[0]

        # Query the ChromaDB collection (get more results to filter duplicates)
        results = collection.query(
            query_embeddings=[text_embedding],
            n_results=min(k * 3, 100),  # Get 3x more results to filter duplicates
            include=['metadatas', 'distances', 'documents']
        )

        # Process results and return unique images with captions
        retrieved_metadata = results['metadatas'][0]
        retrieved_distances = results['distances'][0]
        retrieved_documents = results['documents'][0]

        # Deduplicate by image_id
        unique_images = {}
        for i, meta in enumerate(retrieved_metadata):
            image_id = meta['image_id']
            image_path = meta['image_path']
            caption = retrieved_documents[i]  # Get the caption from documents
            distance = retrieved_distances[i]
            
            # Keep the closest match per image_id
            if image_id not in unique_images or distance < unique_images[image_id]['distance']:
                unique_images[image_id] = {
                    'path': image_path,
                    'caption': caption,
                    'distance': distance
                }

        # Sort by similarity (distance)
        sorted_images = sorted(unique_images.values(), key=lambda x: x['distance'])

        # Return top-k unique results with web-friendly paths
        retrieved_results = []
        for img in sorted_images[:k]:
            # Normalize path for web (replace backslashes with forward slashes)
            web_path = img['path'].replace("\\", "/")
            retrieved_results.append({
                "path": web_path,
                "caption": img['caption']
            })

        return {"results": retrieved_results}
        
    except Exception as e:
        print(f"Search error: {e}")
        return {"error": f"Search failed: {str(e)}"}