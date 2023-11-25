from google.cloud import storage

def get_file_sizes(bucket_name, file_prefix, num_files):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=file_prefix))[:num_files]

    total_size_bytes = sum(blob.size for blob in blobs)
    return total_size_bytes

# Your Google Cloud Storage bucket name
bucket_name = 'mouseion-embeddings'

# Get total sizes for the first 50 embedding files
embedding_prefix = 'embeddings_paraphrase'
embedding_total_size_bytes = get_file_sizes(bucket_name, embedding_prefix, 50)

# Get total sizes for the first 50 associated data files
data_prefix = 'associated_data'
data_total_size_bytes = get_file_sizes(bucket_name, data_prefix, 50)

# Calculate total size in gigabytes
total_size_gb = (embedding_total_size_bytes + data_total_size_bytes) / (1024**3)

print(f"Total size of embedding files: {embedding_total_size_bytes / (1024**3)} GB")
print(f"Total size of associated data files: {data_total_size_bytes / (1024**3)} GB")
print(f"Combined total size: {total_size_gb} GB")
