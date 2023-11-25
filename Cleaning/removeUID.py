import json
import logging
from google.cloud import storage

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize Google Cloud Storage client and bucket
client = storage.Client()
bucket_name = "mouseion-embeddings"  # Replace with your bucket name
bucket = client.bucket(bucket_name)

def remove_id_from_file(blob):
    logging.info(f"Processing file: {blob.name}")
    
    # Read the data
    data = json.loads(blob.download_as_text())

    # Check if data is a list of dictionaries with 'unique_id' key
    if isinstance(data, list) and isinstance(data[0], dict):
        for item in data:
            if "unique_id" in item:
                del item["unique_id"]
            if "embedding" in item:
                item.pop('embedding', None)

    # Write the updated data back to the file
    blob.upload_from_string(json.dumps(data))
    
    logging.info(f"Finished processing file: {blob.name}")

# Process all files
total_files = 823  # Total number of files

for i in range(757, total_files + 1):
    emb_blob_name = f'embeddings_paraphrase_{i}.json'
    metadata_blob_name = f'associated_data_{i}.json'

    # Get blobs
    emb_blob = bucket.blob(emb_blob_name)
    metadata_blob = bucket.blob(metadata_blob_name)

    # Remove unique ID from files
    remove_id_from_file(emb_blob)
    remove_id_from_file(metadata_blob)