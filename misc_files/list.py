from google.cloud import storage

def list_blobs_with_prefix(bucket_name, prefix, delimiter=None):
    """Lists all the blobs in the bucket that begin with the prefix."""
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)

    for blob in blobs:
        print(f"Blob: {blob.name}")
        if blob.metadata:
            print(f"Associated ID: {blob.metadata.get('associated_id', 'No ID found')}")
        else:
            print("No metadata found for this blob.")

# Your bucket name
bucket_name = 'mouseion-embeddings'
prefix = 'embeddings_paraphrase_'  # Prefix for your files

list_blobs_with_prefix(bucket_name, prefix)

