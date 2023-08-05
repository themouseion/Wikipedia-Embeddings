from google.cloud import storage

def remove_files(bucket_name, prefix="paraphrase_test_index_", suffix=".faiss"):
    # Create a client
    client = storage.Client()

    # Access the bucket
    bucket = client.get_bucket(bucket_name)

    # List all files in the bucket
    blobs = bucket.list_blobs(prefix=prefix)

    # Remove files with specified pattern
    for blob in blobs:
        if blob.name.startswith(prefix) and blob.name.endswith(suffix):
            print(f"Deleting {blob.name}")
            blob.delete()

    print(f"All files with prefix '{prefix}' and suffix '{suffix}' have been deleted from bucket '{bucket_name}'")

# Call the function
remove_files('mouseion-embeddings')
