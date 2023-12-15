import json
import gc  
import logging
import hashlib
import time
import requests
from google.cloud import storage

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize Google Cloud Storage client and bucket
client = storage.Client()
bucket_name = "mouseion-embeddings" 
bucket = client.bucket(bucket_name)


start_file_number = 2519  
end_file_number = 2579
chunk_size = 10
embeddings_per_file = 10000

def add_id_to_embedding(embedding_blob, start_id):
    logging.info(f"Processing embedding file: {embedding_blob.name}")

    # Read the embedding data
    data = json.loads(embedding_blob.download_as_text())

    # Check if data is a dictionary with 'unique_id' and 'embedding' keys
    if isinstance(data, dict) and "unique_id" in data and "embedding" in data:
        embeddings = data["embedding"]
    else:
        embeddings = data

    # Overwrite the unique ID for each embedding
    for index, embedding in enumerate(embeddings):
        # If the embedding is a dictionary with 'unique_id' and 'embedding' keys
        if isinstance(embedding, dict) and "unique_id" in embedding and "embedding" in embedding:
            # Overwrite the 'unique_id' and keep the 'embedding' as is
            embeddings[index] = {"unique_id": start_id + index, "embedding": embedding["embedding"]}
        else:
            # Assign a new numerical 'unique_id' and the embedding
            embeddings[index] = {"unique_id": start_id + index, "embedding": embedding}

    # Write the updated embeddings back to the file
    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            embedding_blob.upload_from_string(json.dumps(embeddings))
            break
        except requests.exceptions.ConnectionError:
            if attempt < retry_attempts - 1:  # i.e. if it's not the last attempt
                time.sleep(2)  # wait for 2 seconds before retrying
                continue
            else:
                raise

    logging.info(f"Updated embedding file: {embedding_blob.name} with unique IDs")

    return [embedding["unique_id"] for embedding in embeddings]

def add_id_to_associated_data(associated_data_blob, unique_ids):
    logging.info(f"Processing associated data file: {associated_data_blob.name}")

    # Read the associated data
    data = json.loads(associated_data_blob.download_as_text())

    # Assign the unique ID for each dictionary in the data list
    for index, item in enumerate(data):
        # Assign the corresponding 'unique_id' from unique_ids
        item['unique_id'] = unique_ids[index]

    # Write the updated data back to the file
    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            associated_data_blob.upload_from_string(json.dumps(data))
            break
        except requests.exceptions.ConnectionError:
            if attempt < retry_attempts - 1:  # i.e. if it's not the last attempt
                time.sleep(2)  # wait for 2 seconds before retrying
                continue
            else:
                raise

    logging.info(f"Updated associated data file: {associated_data_blob.name} with unique IDs")

def process_chunk(start, end):
    logging.info(f"Processing chunk from {start} to {end}")
    for i in range(start, end + 1):
        start_id = (i - 1) * embeddings_per_file  # Calculate start_id based on file number

        emb_blob_name = f"embeddings_paraphrase_{i}.json"
        associated_data_blob_name = f"associated_data_{i}.json"

        # Get blobs
        emb_blob = bucket.blob(emb_blob_name)
        associated_data_blob = bucket.blob(associated_data_blob_name)

        # Add unique ID to embedding file and get the unique IDs
        unique_ids = add_id_to_embedding(emb_blob, start_id)

        # Add unique ID to associated data file
        add_id_to_associated_data(associated_data_blob, unique_ids)

    # Clear memory of large objects and run garbage collection
    gc.collect()
    logging.info(f"Finished processing chunk from {start} to {end}")

# Process files in chunks of 1
for i in range(start_file_number, end_file_number + 1, chunk_size):
    process_chunk(i, min(i + chunk_size - 1, end_file_number))

logging.info(f"Completed processing files from {start_file_number} to {end_file_number}")