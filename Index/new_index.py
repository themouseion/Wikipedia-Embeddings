import numpy as np
import faiss
import json
import os
import logging
from google.cloud import storage
from google.api_core.exceptions import NotFound
import random
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_blob_with_retry(blob, max_retries=5, delay=5):
    for i in range(max_retries):
        try:
            return blob.download_as_text()
        except Exception as e:
            if i < max_retries - 1:  # i is zero indexed
                time.sleep(delay)  # wait before trying again
                continue
            else:
                raise

try:
    logger.info("Initializing Google Cloud Storage client.")
    client = storage.Client()
    bucket_name = 'mouseion-embeddings'
    bucket = client.get_bucket(bucket_name)
except Exception as e:
    logger.error(f"Error initializing Google Cloud Storage: {e}")
    exit(1)

try:
    logger.info("Listing all blobs in the bucket.")
    blobs = bucket.list_blobs()
except Exception as e:
    logger.error(f"Error listing blobs: {e}")
    exit(1)

try:
    logger.info("Starting to extract unique file IDs.")
    file_ids = []
    blob_count = 0
    for blob in blobs:
        blob_count += 1
        if blob_count % 100000 == 0:
            logger.info(f"Processed {blob_count} blobs so far.")
        
        if 'chunk_new_' in blob.name and 'embeddings' not in blob.name and 'v2' not in blob.name:
            file_id = int(blob.name.split('_')[-1].split('.')[0])
            file_ids.append(file_id)
    file_ids = sorted(list(set(file_ids)))
    logger.info(f"Successfully extracted {len(file_ids)} unique file IDs.")
except Exception as e:
    logger.error(f"Error during file ID extraction: {e}")
    exit(1)

# Initialize Faiss index and unique identifier list
unique_ids = []
d = 384
nlist = 10000
quantizer = faiss.IndexFlatL2(d)
index_combined = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

on_disk_index_path = "./on_disk_index"
trained_index_file = os.path.join(on_disk_index_path, "trained.index")
os.makedirs(on_disk_index_path, exist_ok=True)

# Check for existing trained index
if os.path.exists(trained_index_file):
    logger.info("Found existing trained index. Loading it.")
    index_combined = faiss.read_index(trained_index_file)
else:
    logger.info("No trained index found. Training a new one.")
    on_disk_ivf_data_path = os.path.join(on_disk_index_path, "ivf_data")
    invlists = faiss.OnDiskInvertedLists(index_combined.nlist, index_combined.code_size, on_disk_ivf_data_path)
    index_combined.replace_invlists(invlists)
    index_combined.make_direct_map()

    # Train the index
    try:
        logger.info("Gathering training data.")
        training_file_ids = random.sample(file_ids, 200)
        training_data = []
        for file_id in training_file_ids:
            blob_emb = bucket.blob(f'embeddings_paraphrase_{file_id}.json')
            emb_text = download_blob_with_retry(blob_emb)
            emb_data = json.loads(emb_text)
            training_data.append(np.asarray(emb_data, dtype=np.float32))
        training_data = np.vstack(training_data)
        index_combined.train(training_data)
        logger.info("Successfully trained the index.")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        exit(1)

    # Save the trained index
    faiss.write_index(index_combined, trained_index_file)
    logger.info("Saved the trained index.")

# Populate the index with embeddings
try:
    logger.info("Populating the index with embeddings.")
    for file_id in file_ids:
        blob_emb = bucket.blob(f'embeddings_paraphrase_{file_id}.json')
        emb_text = download_blob_with_retry(blob_emb)
        emb_data = json.loads(emb_text)
        embeddings_np = np.asarray(emb_data, dtype=np.float32)
        batch_ids = [f"{file_id}_{i}" for i in range(len(embeddings_np))]
        unique_ids.extend(batch_ids)

        batch_size = 1000
        for i in range(0, len(embeddings_np), batch_size):
            index_combined.add(embeddings_np[i:i + batch_size])
    logger.info("Successfully populated the index.")
except Exception as e:
    logger.error(f"Error populating index: {e}")
    exit(1)

# Save unique_ids to a local file
unique_ids_file = "unique_ids.json"
with open(unique_ids_file, 'w') as f:
    json.dump(unique_ids, f)
logger.info(f"Unique ID list saved to {unique_ids_file}.")

# Upload the unique_ids file to Google Cloud Storage
blob_ids = bucket.blob(unique_ids_file)
blob_ids.upload_from_filename(unique_ids_file)
logger.info(f"Unique ID list uploaded to Google Cloud Storage.")

# Upload the populated index to GCS
try:
    logger.info("Uploading index to Google Cloud Storage.")
    for root, dirs, files in os.walk(on_disk_index_path):
        for filename in files:
            local_file_path = os.path.join(root, filename)
            blob_path = os.path.relpath(local_file_path, start=on_disk_index_path)
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_file_path)
except Exception as e:
    logger.error(f"Error uploading index to Google Cloud Storage: {e}")
    exit(1)