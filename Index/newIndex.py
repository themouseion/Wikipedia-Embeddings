import json
import numpy as np
import logging
import gc
from google.cloud import storage
from usearch.index import Index

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

client = storage.Client()
bucket_name = 'mouseion-embeddings'
bucket = client.bucket(bucket_name)

def load_embeddings_and_metadata(file_number):
    embeddings_dict = {}
    metadata_dict = {}

    emb_blob_name = f'embeddings_paraphrase_{file_number}.json'
    emb_blob = bucket.blob(emb_blob_name)
    logging.info(f"Loading embeddings from file: {emb_blob_name}")

    embeddings_data = json.loads(emb_blob.download_as_text())
    for data in embeddings_data:
        embeddings_dict[data['unique_id']] = np.array(data['embedding'], dtype=np.float32)

    # Load the associated metadata
    metadata_blob_name = f'associated_data_{file_number}.json'
    metadata_blob = bucket.blob(metadata_blob_name)
    metadata_data = json.loads(metadata_blob.download_as_text())
    for data in metadata_data:
        metadata_dict[data['unique_id']] = data

    return embeddings_dict, metadata_dict

# Process the first 500 files
for file_number in range(1, 500):
    index = Index(ndim=384, metric='cos', dtype='f32', expansion_add=128, expansion_search=64)
    logging.info("USearch index initialized.")

    embeddings_dict, metadata_dict = load_embeddings_and_metadata(file_number)

    keys = list(embeddings_dict.keys())
    embeddings_array = np.array(list(embeddings_dict.values()))

    if len(keys) != len(embeddings_array):
        logging.error(f"Mismatch in number of keys and embeddings: {len(keys)} keys, {len(embeddings_array)} embeddings")
    else:
        index.add(keys, embeddings_array, threads=4)
        logging.info(f"Processed file {file_number}")

        # Perform clustering
        clustering = index.cluster(min_count=300, max_count=600e)
        centroid_keys, sizes = clustering.centroids_popularity
        clustering.plot_centroids_popularity()

        # Save the index and clustering to disk
        index.save(f"my_usearch_index_{file_number}.usearch")
        with open(f'clustering_{file_number}.json', 'w') as f:
            json.dump({"centroid_keys": centroid_keys.tolist(), "sizes": sizes.tolist()}, f)        

        logging.info(f"Index and clustering saved to disk for file {file_number}")

        # Clear memory of large objects and run garbage collection
        del embeddings_dict, metadata_dict, keys, embeddings_array, clustering
        gc.collect()