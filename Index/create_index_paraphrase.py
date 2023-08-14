import numpy as np
import faiss
import json
import os
import logging
from google.cloud import storage
from google.api_core.exceptions import NotFound
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Setting up Google Cloud Storage client...")
client = storage.Client()
bucket_name = 'mouseion-embeddings'
bucket = client.get_bucket(bucket_name)

# List all blobs in the bucket
blobs = bucket.list_blobs()

# Extract file_ids from blob names, excluding those containing 'v2'
logger.info("Extracting file IDs...")
file_ids = sorted(list(set(int(blob.name.split('_')[-1].split('.')[0]) for blob in blobs
                         if 'chunk_new_' in blob.name and 'embeddings' not in blob.name and 'v2' not in blob.name)))

# Ensure unique file IDs
file_ids = list(dict.fromkeys(file_ids))

d = 384
nlist = 10000
quantizer = faiss.IndexFlatL2(d)
index_combined = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

# Define a directory to save on-disk data
on_disk_index_path = "/home/alexcarrabre/on_disk_index"
os.makedirs(on_disk_index_path, exist_ok=True)

# Create an on-disk inverted list object
on_disk_ivf_data_path = os.path.join(on_disk_index_path, "ivf_data")
invlists = faiss.OnDiskInvertedLists(index_combined.nlist, index_combined.code_size, on_disk_ivf_data_path)
index_combined.replace_invlists(invlists)

# Ensure that index_combined is writable
index_combined.make_direct_map()
index_on_disk = index_combined

# Randomly sample 200 file_ids for training
training_file_ids = random.sample(file_ids, 200)
training_data = []

# Iterate through the randomly selected training file_ids for training
for file_id in training_file_ids:
    try:
        logger.info(f"Processing file_id: {file_id} for training data...")

        # Fetch and load embeddings
        blob_emb = bucket.blob(f'embeddings_paraphrase_{file_id}.json')
        emb_text = blob_emb.download_as_text()
        emb_data = json.loads(emb_text)

        # Convert list of embeddings to numpy array and append to training data
        training_data.append(np.asarray(emb_data, dtype=np.float32))

    except NotFound:
        logger.info(f"File_id: {file_id} not found. Skipping to next file_id...")

# Concatenate training data
training_data = np.vstack(training_data)

# Ensure that training_data has enough samples
if training_data.shape[0] < nlist:
    raise RuntimeError(f"Number of training points ({training_data.shape[0]}) must be at least as large as number of clusters ({nlist})")

index_combined.train(training_data)

# Iterate through the file_ids and add vectors in batches
batch_size = 1000
for file_id in file_ids:
    try:
        logger.info(f"Processing file_id: {file_id}...")

        # Fetch and load embeddings
        blob_emb = bucket.blob(f'embeddings_paraphrase_{file_id}.json')
        emb_text = blob_emb.download_as_text()
        emb_data = json.loads(emb_text)

        # Convert list of embeddings to numpy array
        embeddings_np = np.asarray(emb_data, dtype=np.float32)

        # Add vectors in batches
        for i in range(0, len(embeddings_np), batch_size):
            index_on_disk.add(embeddings_np[i:i + batch_size])

        # Explicitly delete to free up memory
        del embeddings_np, emb_data

    except NotFound:
        logger.info(f"File_id: {file_id} not found. Skipping to next file_id...")

logger.info("Finished processing all files.")
