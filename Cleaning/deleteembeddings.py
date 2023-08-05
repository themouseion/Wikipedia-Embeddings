from google.cloud import storage

def delete_blob(bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete()
    print(f"Blob {blob_name} deleted.")

def delete_matching_blobs(bucket_name, prefix):
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
    for blob in blobs:
        if "embeddings" in blob.name:
            delete_blob(bucket_name, blob.name)

bucket_name = "mouseion-embeddings"  # Your bucket name
prefix = "chunk_new_"  # The prefix of your files

# Call the function to delete the blobs
delete_matching_blobs(bucket_name, prefix)
