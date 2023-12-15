import json
import logging
from google.cloud import storage

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Google Cloud Storage client
client = storage.Client()
bucket_name = 'mouseion-embeddings'
bucket = client.bucket(bucket_name)

def is_valid_unique_id(unique_id):
    """Checks if the unique_id is an integer."""
    return isinstance(unique_id, int)

def check_embeddings_and_data(file_number):
    """Loads embeddings and associated data, then checks for unique ID issues."""
    try:
        # Load embeddings
        emb_blob_name = f'embeddings_paraphrase_{file_number}.json'
        emb_blob = bucket.blob(emb_blob_name)
        embeddings_data = json.loads(emb_blob.download_as_text())
        logging.info(f"Loaded embeddings from {emb_blob_name}")

        # Load associated data
        metadata_blob_name = f'associated_data_{file_number}.json'
        metadata_blob = bucket.blob(metadata_blob_name)
        metadata_data = json.loads(metadata_blob.download_as_text())
        logging.info(f"Loaded associated data from {metadata_blob_name}")

        # Check each unique_id in embeddings and metadata
        for data in embeddings_data + metadata_data:
            unique_id = data.get('unique_id')
            if not is_valid_unique_id(unique_id):
                logging.error(f"Invalid or missing unique_id in file {file_number}: {unique_id}")

    except Exception as e:
        logging.error(f"Error processing file {file_number}: {e}")

# Adjust the range based on how many files you have
for file_number in range(1, 500):
    check_embeddings_and_data(file_number)
