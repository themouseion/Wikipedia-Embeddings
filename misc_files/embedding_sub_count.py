# embedding_sub_count.py

import json
from google.cloud import storage

# Define GCS client
client = storage.Client()
bucket_name = 'mouseion-embeddings'  # replace with your bucket name
bucket = client.get_bucket(bucket_name)

# List of files
file_ids = [1, 10, 100, 1000, 10000, 10001, 10002, 10003, 10004, 10005, 10006]

# Initialize counter
num_vectors = 0

for file_id in file_ids:
    # Fetch and load embeddings
    blob_emb = bucket.blob(f'chunk_new_{file_id}.json_embeddings.json')
    emb_text = blob_emb.download_as_text()
    emb_data = json.loads(emb_text)

    # Count embeddings in each file
    num_vectors += len(emb_data)

print(f"Total number of vectors: {num_vectors}")
