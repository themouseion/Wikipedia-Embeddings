import json
import numpy as np
import logging
import argparse
import os
from google.cloud import storage
from usearch.index import Index
from sentence_transformers import SentenceTransformer

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

# Function to load metadata
def load_metadata(bucket, metadata_file_name):
    metadata_blob = bucket.blob(metadata_file_name)
    if metadata_blob.exists():
        logging.info(f"Loading metadata file: {metadata_file_name}")
        return json.loads(metadata_blob.download_as_text())
    else:
        logging.warning(f"Metadata file not found: {metadata_file_name}")
        return []

# Function to process query
def process_query(index, query_vector, metadata_dict):
    matches = index.search(query_vector, 10)
    for match in matches:
        unique_id = match.key
        metadata = metadata_dict.get(str(unique_id), {})
        print(f"Match: {unique_id}, Score: {match.distance}, Metadata: {metadata}")

# Main function
def main(query_text):
    # Convert query text to an embedding vector
    query_vector = model.encode(query_text)

    # Google Cloud Storage Client
    client = storage.Client()
    bucket_name = 'mouseion-embeddings'
    bucket = client.bucket(bucket_name)

    # Load the index
    index_file_name = "index_first_100_files.usearch"
    index = None
    if os.path.exists(index_file_name):
        try:
            index = Index.restore(index_file_name)
            logging.info("Index loaded.")
        except Exception as e:
            logging.error(f"Failed to load index: {e}")
            return

    if not index:
        logging.error("Index is not available. Exiting.")
        return

    # Load metadata
    metadata_list = load_metadata(bucket, 'first_100_metadata_combined.json')
    metadata_dict = {str(item['unique_id']): item for item in metadata_list}

    # Process the query
    process_query(index, query_vector, metadata_dict)

# Argument parser for command-line interaction
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query USEarch Index")
    parser.add_argument("query_text", type=str, help="Text to query")
    args = parser.parse_args()
    main(args.query_text)
