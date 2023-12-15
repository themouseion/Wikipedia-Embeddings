import json
import numpy as np
import logging
import gc
import os
from google.cloud import storage
from usearch.index import Index

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Google Cloud Storage Client
client = storage.Client()
bucket_name = 'mouseion-embeddings'
bucket = client.bucket(bucket_name)

# Function to load embeddings
def load_embeddings(file_number):
    emb_blob_name = f'embeddings_paraphrase_{file_number}.json'
    emb_blob = bucket.blob(emb_blob_name)
    if emb_blob.exists():
        logging.info(f"Loading file: {emb_blob_name}")
        embeddings_data = json.loads(emb_blob.download_as_text())
        logging.info(f"Loaded {len(embeddings_data)} embeddings from file {file_number}")
        return {data['unique_id']: np.array(data['embedding'], dtype=np.float32) for data in embeddings_data}
    else:
        logging.warning(f"File not found: {emb_blob_name}")
        return {}

# Initialize a new USEarch index
index = Index(ndim=384, metric='cos', dtype='f32', expansion_add=128, expansion_search=64)
logging.info("USEarch index initialized.")

# Process the first 100 files
total_files_to_process = 100

for file_number in range(1, total_files_to_process + 1):
    embeddings_dict = load_embeddings(file_number)
    if embeddings_dict:
        new_keys = []
        new_embeddings = []
        for key, embedding in embeddings_dict.items():
            new_keys.append(key)
            new_embeddings.append(embedding)
        if new_keys:
            index.add(new_keys, np.array(new_embeddings), threads=4)
            logging.info(f"Added {len(new_keys)} new embeddings from file {file_number} to index")

    gc.collect()

# Save the index
index_file_name = "index_first_100_files.usearch"
index.save(index_file_name)
logging.info(f"Index for the first 100 files saved as {index_file_name}")

# Optionally, upload the index to Google Cloud Storage
blob = bucket.blob(index_file_name)
blob.upload_from_filename(index_file_name)
logging.info(f"Index uploaded to Google Cloud Storage: {blob.name}")

logging.info("Index creation for the first 100 files completed.")
