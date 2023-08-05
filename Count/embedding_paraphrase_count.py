from google.cloud import storage

def count_embeddings_files(bucket_name, prefix):
    # Create a client
    client = storage.Client()

    # Access the bucket
    bucket = client.get_bucket(bucket_name)

    # List all files in the bucket
    blobs = bucket.list_blobs(prefix=prefix)

    # Count and print the number of files
    count = sum(1 for _ in blobs)
    print(f"There are {count} files with prefix '{prefix}' in bucket '{bucket_name}'")

# Call the function
count_embeddings_files('mouseion-embeddings', 'embeddings_paraphrase_')
