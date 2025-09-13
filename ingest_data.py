import torch
import os
import json
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import chromadb
from config import *

# --- Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load Pre-trained CLIP Model ---
print("Loading pre-trained CLIP model...")
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
print("CLIP model loaded.")

# --- Initialize ChromaDB Client ---
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection_exists = False

try:
    collection = client.get_collection(name=COLLECTION_NAME)
    doc_count = collection.count()
    print(f"✅ Collection '{COLLECTION_NAME}' already exists with {doc_count} documents.")
    print("🚀 Skipping ingestion - database is ready to use!")
    collection_exists = True
except Exception:
    # If the collection does not exist, create it
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    print(f"📦 ChromaDB collection '{COLLECTION_NAME}' created. Starting ingestion...")

# Only run ingestion if the collection did not already exist
if not collection_exists:
    # --- Load COCO annotations and prepare data ---
    with open(DATASET_PATH, 'r') as f:
        coco_data = json.load(f)

    print("Starting data ingestion...")

    # Lists to hold data for batch processing
    batch_embeddings = []
    batch_documents = []
    batch_metadatas = []
    batch_ids = []

    # Statistics tracking
    total_annotations = len(coco_data['annotations'])
    processed_count = 0
    skipped_count = 0

    print(f"Starting data ingestion for {total_annotations} annotations...")

    for i, item in enumerate(coco_data['annotations']):
        image_id = item['image_id']
        caption = item['caption']
        
        # Use the unique annotation ID instead of the image ID
        annotation_id = item['id']

        # Format image filename with leading zeros (e.g., 9 -> 000000000009)
        image_filename = f"{image_id:012d}.jpg"
        absolute_image_path = get_absolute_image_path(image_filename)
        relative_image_path = get_relative_image_path(image_filename)

        if not os.path.exists(absolute_image_path):
            skipped_count += 1
            if skipped_count % 100 == 0:
                print(f"Skipped {skipped_count} missing images so far...")
            continue # Skip if image file not found

        try:
            # Load and process the image
            image = Image.open(absolute_image_path).convert("RGB")
            
            # Use CLIP to get the image embedding
            inputs = processor(text=[caption], images=image, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                image_embedding = outputs.image_embeds.cpu().numpy().tolist()[0]
                
            # Add to the current batch
            batch_embeddings.append(image_embedding)
            batch_documents.append(caption) # Store the caption as a document
            # Store relative path for web serving and image_id for evaluation
            batch_metadatas.append({
                "image_id": str(image_id), 
                "image_path": relative_image_path  # Use relative path for web serving
            })
            # Use the unique annotation_id for the ChromaDB id
            batch_ids.append(str(annotation_id))
            
            processed_count += 1
            
            # Check if the batch is full
            if len(batch_ids) >= BATCH_SIZE:
                print(f"Adding batch of {len(batch_ids)} documents to ChromaDB... (Processed: {processed_count}/{total_annotations})")
                collection.add(
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                
                # Clear lists for next batch
                batch_embeddings = []
                batch_documents = []
                batch_metadatas = []
                batch_ids = []

        except Exception as e:
            print(f"Error processing image {image_filename}: {e}")
            skipped_count += 1

    # Add any remaining items in the last batch
    if len(batch_ids) > 0:
        print(f"Adding final batch of {len(batch_ids)} documents...")
        collection.add(
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas,
            ids=batch_ids
        )

    print("\n" + "="*50)
    print("🎉 INGESTION COMPLETE!")
    print("="*50)
    print(f"✅ Total annotations processed: {processed_count}")
    print(f"⚠️  Images skipped (missing): {skipped_count}")
    print(f"📊 Success rate: {(processed_count/(processed_count+skipped_count)*100):.1f}%")
    print(f"🗄️  Total documents in collection: {collection.count()}")
    print("="*50)

# Always show final collection status
print(f"\n📊 Final collection status: {collection.count()} documents in '{COLLECTION_NAME}'")