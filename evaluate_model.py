import torch
import json
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import chromadb
from config import *

# --- Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
K = 10  # Set the value for K (e.g., top 10 results)

# --- Load Pre-trained CLIP Model ---
print("Loading CLIP model for evaluation...")
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# --- Initialize ChromaDB Client ---
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

# --- Load COCO annotations for ground truth ---
with open(DATASET_PATH, 'r') as f:
    coco_data = json.load(f)

print("Starting evaluation...")
hits = 0
total_queries = 0

for item in coco_data['annotations']:
    # Get the image ID from the annotation for ground-truth comparison
    ground_truth_image_id = str(item['image_id'])
    caption = item['caption']
    
    # Generate the embedding for the text query
    inputs = processor(text=[caption], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs).cpu().numpy().tolist()[0]

    # Perform a search in the vector database
    results = collection.query(
        query_embeddings=[text_embedding],
        n_results=K,
        # Now include metadata to get the image_id
        include=['metadatas']
    )

    # Check if the original image ID is in the search results
    retrieved_metadata = results['metadatas'][0]
    retrieved_image_ids = [meta['image_id'] for meta in retrieved_metadata]
    
    if ground_truth_image_id in retrieved_image_ids:
        hits += 1

    total_queries += 1
    
    if total_queries % 100 == 0:
        print(f"Processed {total_queries} queries. Current Recall@{K}: {(hits / total_queries) * 100:.2f}%")

# --- Final Results ---
final_recall = (hits / total_queries) * 100 if total_queries > 0 else 0
print("\n--- Final Evaluation Results ---")
print(f"Total queries: {total_queries}")
print(f"Successful retrievals (hits): {hits}")
print(f"Final Recall@{K}: {final_recall:.2f}%")