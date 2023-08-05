from google.cloud import storage

def count_chunk_files(bucket_name, prefix, exclude_term):
    # Create a client
    client = storage.Client()

    # Access the bucket
    bucket = client.get_bucket(bucket_name)

    # List all files in the bucket
    blobs = bucket.list_blobs(prefix=prefix)

    # Count the files that don't contain the exclude_term
    count = sum(1 for blob in blobs if exclude_term not in blob.name)
    print(f"There are {count} files with prefix '{prefix}' (excluding '{exclude_term}') in bucket '{bucket_name}'")

# Call the function
count_chunk_files('mouseion-embeddings', 'chunk_new_', 'embeddings')
