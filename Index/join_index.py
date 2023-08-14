import faiss
import os
import logging
from google.cloud import storage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a directory for on-disk storage
on_disk_dir = "temp_shards"

# Check if the directory exists, create if not
if not os.path.exists(on_disk_dir):
    os.makedirs(on_disk_dir)

logger.info("Setting up Google Cloud Storage client...")
client = storage.Client()
bucket_name = 'mouseion-embeddings'
bucket = client.get_bucket(bucket_name)

# List all blobs in the bucket
blobs = bucket.list_blobs()

# Extract index file names and sort them
logger.info("Extracting index file names...")
index_files = [blob.name for blob in blobs if 'paraphrase_test_index_' in blob.name]
index_files.sort(key=lambda name: int(name.split('_')[-1].split('.')[0]))

# Define shards
d = 384  # Dimension of the embeddings

# List to hold the on-disk indices
on_disk_indices = []

# Iterate through index files and add them to the shards
for index_file_name in index_files:
    logger.info(f"Processing index file: {index_file_name}...")

    # Check if on-disk file already exists
    on_disk_file_path = f"{on_disk_dir}/{index_file_name}"
    if not os.path.exists(on_disk_file_path):
        # Download the index file from Google Cloud Storage
        blob_index = bucket.blob(index_file_name)
        local_file_name = f'local_{index_file_name}'
        blob_index.download_to_filename(local_file_name)

        # Read the index
        index = faiss.read_index(local_file_name)

        # Create an empty on-disk index with the same parameters
        quantizer = faiss.IndexFlatL2(d)
        on_disk_index = faiss.IndexIVFFlat(quantizer, d, index.nlist)
        on_disk_index.own_invlists = False

        # Add to the list of on-disk indices
        on_disk_indices.append(on_disk_index)

        # Delete the local file
        os.remove(local_file_name)
        logger.info(f"Processed index file: {index_file_name}.")

# Create a final concatenated index
final_index = faiss.IndexFlatL2(d)
for on_disk_index in on_disk_indices:
    final_index = faiss.index_cpu_to_all_gpus(final_index)
    final_index.add(on_disk_index.reconstruct_n(0, on_disk_index.ntotal))

# Define the combined index file name
combined_index_file = 'paraphrase_test_index_shards.faiss'

# Save the combined index to a file
faiss.write_index(final_index, combined_index_file)
logger.info(f"Combined FAISS index file {combined_index_file} created.")

# Upload the combined index file to Google Cloud Storage
blob_combined_index = bucket.blob(combined_index_file)
blob_combined_index.upload_from_filename(combined_index_file)
logger.info(f"Combined FAISS index file {combined_index_file} uploaded to Google Cloud Storage.")

# Delete local combined index file
os.remove(combined_index_file)
logger.info("Local files deleted.")
logger.info("Finished processing all index files.")
