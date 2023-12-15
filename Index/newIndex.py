import json
import numpy as np
import logging
import gc
import os
import psutil
from google.cloud import storage
from usearch.index import Index

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Google Cloud Storage Client
client = storage.Client()
bucket_name = 'mouseion-embeddings'
bucket = client.bucket(bucket_name)

def load_embeddings(file_number):
    emb_blob_name = f'embeddings_paraphrase_{file_number}.json'
    emb_blob = bucket.blob(emb_blob_name)
    if emb_blob.exists():
        logging.info(f'Loading file: {emb_blob_name}')
        embeddings_data = json.loads(emb_blob.download_as_text())
        return {data['unique_id']: np.array(data['embedding'], dtype=np.float32) for data in embeddings_data}
    else:
        logging.warning(f'File not found: {emb_blob_name}')
        return {}

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / 1024 ** 2  # Convert RSS to MB
    logging.info(f'Memory usage: {mem_usage:.2f} MB')
    return mem_usage

# Initialize or load the index
index_file_name = 'index.usearch'
index = Index(ndim=384, metric='cos', dtype='f32', expansion_add=128, expansion_search=64)

total_files = 13627
batch_size = 10

for i in range(1, total_files + 1, batch_size):
    logging.info(f'Starting batch processing from file {i}')
    for file_number in range(i, min(i + batch_size, total_files + 1)):
        embeddings_dict = load_embeddings(file_number)
        if embeddings_dict:
            new_keys = [key for key in embeddings_dict.keys()]
            new_embeddings = np.array(list(embeddings_dict.values()))
            index.add(new_keys, new_embeddings, threads=4)

            # Clear numpy arrays and force garbage collection
            del new_keys, new_embeddings, embeddings_dict
            gc.collect()

        log_memory_usage()

    # Save and close index to free up memory
    index.save(index_file_name)
    del index
    gc.collect()
    index = Index.restore(index_file_name)
    logging.info(f'Index saved and reloaded after processing file {min(i + batch_size - 1, total_files)}')

logging.info('Index creation completed.')
