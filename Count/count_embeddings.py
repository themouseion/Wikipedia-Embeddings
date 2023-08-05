import datetime
from google.cloud import storage

# Instantiate a storage client
storage_client = storage.Client()

# The name of your bucket
bucket_name = "mouseion-embeddings"

# Get the bucket
bucket = storage_client.bucket(bucket_name)

# Define a prefix (if all relevant files are in a subdirectory)
prefix = "" # if no subdirectory, keep it empty

# Define a list to collect the names of relevant files
relevant_files = []

# List blobs in the bucket
blobs = bucket.list_blobs(prefix=prefix)

# Iterate over blobs, checking if the filename matches the pattern
for blob in blobs:
    filename = blob.name
    if filename.startswith("chunk_new_") and filename.endswith("_embeddings.json"):
        relevant_files.append(filename)

# Print the count of matching files with timestamp
print(f"{datetime.datetime.now()}: {len(relevant_files)}")
