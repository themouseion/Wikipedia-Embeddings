import json
import os
from google.cloud import storage
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Google Cloud Storage Client
client = storage.Client()
bucket_name = 'mouseion-embeddings'
bucket = client.bucket(bucket_name)

def load_metadata(file_number):
    """ Load metadata from a specific file. """
    metadata_blob_name = f'associated_data_{file_number}.json'
    metadata_blob = bucket.blob(metadata_blob_name)
    if metadata_blob.exists():
        logging.info(f"Loading file: {metadata_blob_name}")
        metadata_data = json.loads(metadata_blob.download_as_text())
        return metadata_data
    else:
        logging.warning(f"File not found: {metadata_blob_name}")
        return None

# Path to store combined metadata on your server
output_file_path = '/home/alexcarrabre/first_100_metadata_combined.json'

# Processing only the first 100 files
total_files_to_process = 100
logging.info(f"Processing the first {total_files_to_process} files.")

# Open the output file in write mode
with open(output_file_path, 'w') as output_file:
    output_file.write('[')  # Start of JSON array
    first_entry = True

    for file_number in range(1, total_files_to_process + 1):
        file_metadata = load_metadata(file_number)
        if file_metadata:
            for item in file_metadata:
                if not first_entry:
                    output_file.write(',')
                json.dump(item, output_file)
                first_entry = False
            logging.info(f"Processed file number: {file_number}")

    output_file.write(']')  # End of JSON array
    logging.info(f"Completed writing to output file at {output_file_path}.")

logging.info(f"Metadata for the first 100 files combined and saved to disk at {output_file_path}.")
