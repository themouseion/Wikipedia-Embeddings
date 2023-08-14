import logging
from google.cloud import storage
import tempfile
import os
import pandas as pd
import dask.dataframe as dd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define GCS client and bucket
client = storage.Client()
bucket_name = "mouseion-embeddings"
bucket = client.get_bucket(bucket_name)

# A set to ensure uniqueness of file numbers
processed_file_numbers = set()

# Output Parquet path
output_path = "combined_data.parquet"

# Batch size
batch_size = 50
current_batch = []
batch_number = 1

# Iterate over the blobs in the bucket
logger.info("Fetching blobs...")
blobs = bucket.list_blobs(prefix="associated_data_")

for blob in blobs:
    if 'v2' in blob.name:
        continue

    # Extract file number
    file_number = blob.name.split('_')[-1].split('.')[0]

    # Skip redundant files
    if file_number in processed_file_numbers:
        continue

    processed_file_numbers.add(file_number)
    logger.info(f"Processing blob: {blob.name}")

    # Download the JSON file
    json_path = tempfile.mktemp(suffix=".json")
    blob.download_to_filename(json_path)
    current_batch.append(json_path)

    if len(current_batch) == batch_size:
        logger.info(f"Processing batch {batch_number}")
        combined_data = dd.concat([dd.from_pandas(pd.read_json(file), npartitions=1) for file in current_batch])
        combined_data.to_parquet(output_path, engine="pyarrow", append=(batch_number > 1))
        current_batch = []
        batch_number += 1
        logger.info(f"Batch {batch_number} processed.")

# Process the remaining batch
if current_batch:
    logger.info(f"Processing last batch")
    combined_data = dd.concat([dd.from_pandas(pd.read_json(file), npartitions=1) for file in current_batch])
    combined_data.to_parquet(output_path, engine="pyarrow", append=(batch_number > 1))
    logger.info(f"Last batch processed.")

# Remove the temporary files
for file in current_batch:
    os.remove(file)

logger.info(f"Combined data saved to {output_path}")
