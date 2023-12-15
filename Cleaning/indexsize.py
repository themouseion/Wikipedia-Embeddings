import os
import json
import numpy as np
from google.cloud import storage

def estimate_index_size(file_number, total_files):
    client = storage.Client()
    bucket_name = 'mouseion-embeddings'
    bucket = client.bucket(bucket_name)
    emb_blob_name = f'embeddings_paraphrase_{file_number}.json'
    emb_blob = bucket.blob(emb_blob_name)

    if emb_blob.exists():
        # Download and load a single file
        single_file_data = json.loads(emb_blob.download_as_text())
        single_file_size = sum(len(json.dumps(entry)) for entry in single_file_data)
        
        # Estimate total size for all files
        total_size_bytes = single_file_size * total_files
        total_size_gb = total_size_bytes / (1024 ** 3)  # Convert bytes to gigabytes

        return total_size_gb

    else:
        print(f"File not found: {emb_blob_name}")
        return None

# Example usage
file_number = 1  # Example file number to use for estimation
total_files = 13627  # Total number of files
estimated_total_size = estimate_index_size(file_number, total_files)

if estimated_total_size is not None:
    print(f"Estimated total size for {total_files} files: {estimated_total_size:.2f} GB")
