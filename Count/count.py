from google.cloud import storage
import operator

def list_top_files(bucket_name):
    """List the top 10 files in the given Google Cloud Storage bucket."""
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    # Get list of files in the bucket
    blobs = bucket.list_blobs()

    # Create a dictionary to hold file names and sizes
    files_and_sizes = {blob.name: blob.size for blob in blobs}

    # Sort files by size
    sorted_files = sorted(files_and_sizes.items(), key=operator.itemgetter(1), reverse=True)

    # Print the top 10 files
    for file_name, size in sorted_files[:10]:
        print(f"File: {file_name}, Size: {size} bytes")

# Replace with your bucket name
bucket_name = "mouseion-embeddings"
list_top_files(bucket_name)
