import faiss
import os
import logging
from google.cloud import storage
import requests
import numpy as np
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Kubo API URL
ipfs_url = 'http://127.0.0.1:5001/api/v0'

logger.info("Setting up Google Cloud Storage client...")
client = storage.Client()
bucket_name = 'mouseion-embeddings'
bucket = client.get_bucket(bucket_name)

# List all blobs in the bucket
logger.info("Listing all blobs in the bucket...")
blobs = bucket.list_blobs()

# Extract index file names and sort them
logger.info("Extracting index file names...")
index_files = [blob.name for blob in blobs if 'paraphrase_test_index_' in blob.name]
index_files.sort(key=lambda name: int(name.split('_')[-1].split('.')[0]))
logger.info(f"Found {len(index_files)} index files.")

# Define shards
d = 384  # Dimension of the embeddings
index_shards = faiss.IndexShards(d)
logger.info("Index shards defined.")

# Load processed index files from a JSON file, if it exists
try:
    with open("processed_indexes.json", "r") as f:
        processed_indexes = json.load(f)
    logger.info(f"Loaded {len(processed_indexes)} processed index files from processed_indexes.json.")
except FileNotFoundError:
    processed_indexes = {}

# Iterate through index files and add them to the shards
for index_file_name in index_files:
    logger.info(f"Processing {index_file_name}...")

    # Check if this index file has already been processed
    if index_file_name in processed_indexes:
        temp_path = processed_indexes[index_file_name]
        logger.info(f"Index file {index_file_name} already processed.")
    else:
        # Download the index file
        blob = bucket.blob(index_file_name)
        temp_data = blob.download_as_bytes()

        # Add the index file to IPFS
        try:
            response = requests.post(f'{ipfs_url}/add', files={'file': temp_data})
            response.raise_for_status()
            temp_path = response.json()['Hash']
        except (requests.HTTPError, json.JSONDecodeError):
            logger.error(f"Failed to add index file {index_file_name} to IPFS.")
            continue

        # Save the IPFS path of the index file
        processed_indexes[index_file_name] = temp_path
        with open("processed_indexes.json", "w") as f:
            json.dump(processed_indexes, f)

        logger.info(f"Index file {index_file_name} added to IPFS at {temp_path}.")

    # Read the index from IPFS
    try:
        response = requests.get(f'{ipfs_url}/cat?arg={temp_path}')
        response.raise_for_status()
        index_data = response.content
        index_data = np.frombuffer(index_data, dtype=np.uint8)  # Convert bytes to numpy array
        index = faiss.deserialize_index(index_data.tobytes())  # Convert numpy array to bytes and deserialize
        index_shards.add_index(index)
    except (requests.HTTPError, ValueError):
        logger.error(f"Failed to read index from IPFS at {temp_path}.")
        continue

    logger.info(f"Index from {temp_path} added to the shards.")

# Save the final index to IPFS
logger.info("Saving the final index to IPFS...")
index_binary = faiss.serialize_index(index_shards)
try:
    response = requests.post(f'{ipfs_url}/add', files={'file': index_binary})
    response.raise_for_status()
    final_index_path = response.json()['Hash']
except (requests.HTTPError, json.JSONDecodeError):
    logger.error("Failed to save the final index to IPFS.")
else:
    logger.info(f"Final FAISS index stored in IPFS at {final_index_path}.")
