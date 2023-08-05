import numpy as np
import faiss
import json
import os
import logging
from google.cloud import storage
from google.api_core.exceptions import NotFound

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Setting up Google Cloud Storage client...")
client = storage.Client()
bucket_name = 'mouseion-embeddings'
bucket = client.get_bucket(bucket_name)

# List all blobs in the bucket
blobs = bucket.list_blobs()

# Extract file_ids from blob names
logger.info("Extracting file IDs...")
file_ids = sorted(list(set(int(blob.name.split('_')[-1].split('.')[0]) for blob in blobs
                         if 'chunk_new_' in blob.name and 'embeddings' not in blob.name)))

# Define the range of files you want to process
start_index = 11000  # Starting index of the range (inclusive)
end_index = 13627   # Ending index of the range (exclusive)

# Extract the specific range of file_ids you want to process
file_ids_range = file_ids[start_index:end_index]

nlist = 100  # You can modify this number based on your requirements
quantizer = faiss.IndexFlatL2(384)  # 384 is the dimension of the embeddings

for file_id in file_ids_range:
    try:
        # Check if the index file already exists
        index_file = f'paraphrase_test_index_{file_id}.faiss'
        blob_index = bucket.blob(index_file)
        if blob_index.exists():
            logger.info(f"Index file {index_file} already exists. Skipping...")
            continue

        logger.info(f"Processing file_id: {file_id}...")

        # Fetch and load associated data
        blob_data = bucket.blob(f'chunk_new_{file_id}.json')
        data_text = blob_data.download_as_text()
        data = json.loads(data_text)

        # Fetch and load embeddings
        blob_emb = bucket.blob(f'embeddings_paraphrase_{file_id}.json')
        emb_text = blob_emb.download_as_text()
        emb_data = json.loads(emb_text)

        # Convert list of embeddings to numpy array
        embeddings_np = np.asarray(emb_data, dtype=np.float32)

        # Build FAISS index
        logger.info("Building FAISS index...")
        index = faiss.IndexIVFFlat(quantizer, 384, nlist, faiss.METRIC_L2)
        index.train(embeddings_np)  # Training step
        index.add(embeddings_np)
        logger.info(f"FAISS IVF index built with {index.ntotal} vectors.")

        # Save the index to a file
        faiss.write_index(index, index_file)
        logger.info(f"FAISS index file {index_file} created.")

        # Upload the index file to Google Cloud Storage
        blob_index.upload_from_filename(index_file)
        logger.info(f"FAISS index file {index_file} uploaded to Google Cloud Storage.")

        # Save associated data to a file
        data_file = f'associated_data_{file_id}.json'
        with open(data_file, 'w') as f:
            json.dump(data, f)
        logger.info(f"Associated data file {data_file} created.")

        # Upload the associated data file to Google Cloud Storage
        blob_data = bucket.blob(data_file)
        blob_data.upload_from_filename(data_file)
        logger.info(f"Associated data file {data_file} uploaded to Google Cloud Storage.")

        # Delete local index and data files
        os.remove(index_file)
        os.remove(data_file)
        logger.info("Local files deleted.")

    except NotFound:
        logger.info(f"File_id: {file_id} not found. Skipping to next file_id...")

logger.info("Finished processing selected range of files.")
