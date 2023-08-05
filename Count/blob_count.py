from google.cloud import storage

def count_blobs(bucket_name, prefix):
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
    count = sum(1 for _ in blobs)
    print(f"Found {count} blobs with prefix {prefix}")

bucket_name = "mouseion-embeddings"  # Your bucket name
prefix = "chunk_new_"  # The prefix of your files

# Call the function to count the blobs
count_blobs(bucket_name, prefix)
