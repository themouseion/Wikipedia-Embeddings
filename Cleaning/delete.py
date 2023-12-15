from google.cloud import storage
import re

def delete_matching_files(bucket_name, pattern):
    """Delete files matching the given pattern from the specified Google Cloud Storage bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Compile the pattern
    regex = re.compile(pattern)

    # List all files in the bucket and delete those matching the pattern
    blobs = bucket.list_blobs()
    for blob in blobs:
        if regex.match(blob.name):
            blob.delete()
            print(f"Deleted: {blob.name}")

# Replace with your bucket name
bucket_name = "mouseion-embeddings"
pattern = r'chunk_new_\d+\.json_embeddings\.json_embeddings\.json'

delete_matching_files(bucket_name, pattern)
