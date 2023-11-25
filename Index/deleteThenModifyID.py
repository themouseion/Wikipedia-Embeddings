import json
import gc  # Garbage Collector
import logging
import hashlib
from google.cloud import storage

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize Google Cloud Storage client and bucket
client = storage.Client()
bucket_name = "mouseion-embeddings"  # Replace with your bucket name
bucket = client.bucket(bucket_name)


def add_id_to_embedding(embedding_blob):
    logging.info(f"Processing embedding file: {embedding_blob.name}")

    # Read the embedding data
    data = json.loads(embedding_blob.download_as_text())

    # Add a unique ID to each embedding
    for index, embedding in enumerate(data):
        embedding["unique_id"] = index

    # Write the updated embeddings back to the file
    embedding_blob.upload_from_string(json.dumps(data))
    logging.info(f"Updated embedding file: {embedding_blob.name} with unique IDs")

    return [embedding["unique_id"] for embedding in data]


def add_id_to_associated_data(associated_data_blob, unique_ids):
    logging.info(f"Processing associated data file: {associated_data_blob.name}")

    # Read the associated data
    data = json.loads(associated_data_blob.download_as_text())

    # Add the unique ID to each dictionary in the data list
    for item, unique_id in zip(data, unique_ids):
        item["unique_id"] = unique_id

    # Write the updated data back to the file
    associated_data_blob.upload_from_string(json.dumps(data))
    logging.info(f"Updated associated data file: {associated_data_blob.name} with unique IDs")


def process_chunk(start, end):
    logging.info(f"Processing chunk from {start} to {end}")
    for i in range(start, end + 1):
        emb_blob_name = f"embeddings_paraphrase_{i}.json"
        associated_data_blob_name = f"associated_data_{i}.json"

        # Get blobs
        emb_blob = bucket.blob(emb_blob_name)
        associated_data_blob = bucket.blob(associated_data_blob_name)

        # Add unique ID to embedding file and get the unique IDs
        unique_ids = add_id_to_embedding(emb_blob)

        # Add unique ID to associated data file
        add_id_to_associated_data(associated_data_blob, unique_ids)

    # Clear memory of large objects and run garbage collection
    gc.collect()
    logging.info(f"Finished processing chunk from {start} to {end}")


# Process file number 1
file_number = 1
process_chunk(file_number, file_number)
logging.info(f"Completed processing file number {file_number}")