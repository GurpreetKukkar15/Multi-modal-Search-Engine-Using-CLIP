"""
Configuration file for the Multi-Modal Search Engine
Centralizes all configuration to avoid hardcoded values across files.
"""

import os

# --- Data Configuration ---
DATA_SPLIT = 'val'  # Change to 'train' for larger dataset
DATASET_PATH = f"data/annotations/captions_{DATA_SPLIT}2017.json"
IMAGE_DIR = f"data/{DATA_SPLIT}2017/"

# --- Model Configuration ---
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda"  # Will be set to "cpu" if CUDA not available

# --- Database Configuration ---
COLLECTION_NAME = f"image_search_{DATA_SPLIT}"
CHROMA_DB_PATH = "./chroma_db"  # Persistent database directory

# --- API Configuration ---
API_HOST = "127.0.0.1"
API_PORT = 8000
K_RESULTS = 5  # Number of search results to return

# --- Processing Configuration ---
BATCH_SIZE = 50  # Batch size for data ingestion

# --- Path Configuration ---
def get_relative_image_path(image_filename):
    """Convert image filename to relative path for web serving"""
    return f"data/{DATA_SPLIT}2017/{image_filename}"

def get_absolute_image_path(image_filename):
    """Convert image filename to absolute path for file operations"""
    return os.path.join(IMAGE_DIR, image_filename)
