from google.cloud import storage

# def remove_files(bucket_name, prefix="paraphrase_test_index_", suffix="_v2.faiss"):
# def remove_files(bucket_name, prefix="associated_data_", suffix="_v2.json"):
def remove_files(bucket_name, prefix="associated_data_", suffix="_v2.json"):
    # Create a client
    client = storage.Client()

    # Access the bucket
    bucket = client.get_bucket(bucket_name)

    # List all files in the bucket with the given prefix
    blobs = bucket.list_blobs(prefix=prefix)

    # Remove files with the specified pattern
    for blob in blobs:
        # Check the suffix to ensure it matches the desired pattern
        if blob.name.endswith(suffix):
            print(f"Deleting {blob.name}")
            blob.delete()

    print(f"All files with prefix '{prefix}' and suffix '{suffix}' have been deleted from bucket '{bucket_name}'")

# Call the function
remove_files('mouseion-embeddings')
