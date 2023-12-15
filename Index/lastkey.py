import numpy as np
from google.cloud import storage
from usearch.index import Index

# Initialize Google Cloud Storage client and specify your bucket
client = storage.Client()
bucket_name = 'your-bucket-name'  # Replace with your bucket name
bucket = client.bucket(bucket_name)

# Path to your USearch index file
index_file_path = 'index_full.usearch'  # Replace with your index file name
index_file_blob = bucket.blob(index_file_path)

# Download the index file from Google Cloud Storage
index_file_blob.download_to_filename(index_file_path)

# Load the index
index = Index.restore(index_file_path)

# Check if the index is empty
if len(index) == 0:
    print("The index is empty.")
else:
    # Retrieve the last key (unique ID)
    last_key = max(index.keys())
    # Retrieve the corresponding embedding
    last_embedding = index[last_key]

    print(f"Last Unique ID: {last_key}")
    print(f"Last Embedding: {last_embedding}")
