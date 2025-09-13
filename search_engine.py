import torch
import chromadb
from transformers import CLIPProcessor, CLIPModel
import os
import matplotlib.pyplot as plt
from PIL import Image
from config import *

# --- Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load Pre-trained CLIP Model ---
print("Loading CLIP model for search...")
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
print("CLIP model loaded.")

# --- Initialize ChromaDB Client ---
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)
print(f"Connected to ChromaDB collection '{COLLECTION_NAME}' with {collection.count()} items.")

# --- Main Search Function ---
def search_images(query_text):
    """
    Takes a text query, performs a CLIP embedding, and searches ChromaDB.
    """
    print(f"\nSearching for: '{query_text}'")

    # 1. Generate text embedding from the query
    inputs = processor(text=[query_text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs).cpu().numpy().tolist()[0]

    # 2. Query the ChromaDB collection
    results = collection.query(
        query_embeddings=[text_embedding],
        n_results=K_RESULTS,
        include=['metadatas']
    )
    
    # 3. Process and display the results
    retrieved_paths = [meta['image_path'] for meta in results['metadatas'][0]]
    retrieved_distances = results['distances'][0]

    if not retrieved_paths:
        print("No results found.")
        return

    print("\n--- Top Results ---")
    
    # Display the images using matplotlib
    fig, axes = plt.subplots(1, len(retrieved_paths), figsize=(15, 5))
    if len(retrieved_paths) == 1:
        axes = [axes] # Ensure axes is an array for single result case
    
    for i, path in enumerate(retrieved_paths):
        try:
            image = Image.open(path)
            axes[i].imshow(image)
            axes[i].set_title(f"Rank {i+1}\nDistance: {retrieved_distances[i]:.2f}")
            axes[i].axis('off')
        except FileNotFoundError:
            print(f"Error: Image not found at {path}")
            axes[i].set_title("Image Not Found")
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    while True:
        user_query = input("\nEnter your search query (or type 'quit' to exit): ")
        if user_query.lower() == 'quit':
            break
        search_images(user_query)