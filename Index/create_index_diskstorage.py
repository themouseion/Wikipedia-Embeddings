import numpy as np
import faiss
import json
import os
import logging
from google.cloud import storage
from google.api_core.exceptions import NotFound

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Setting up Google Cloud Storage client...")
client = storage.Client()
bucket_name = 'mouseion-embeddings'
bucket = client.get_bucket(bucket_name)

# List all blobs in the bucket
blobs = bucket.list_blobs()

# Extract file_ids from blob names
logger.info("Extracting file IDs...")
file_ids = sorted(list(set(int(blob.name.split('_')[-1].split('.')[0]) for blob in blobs
                         if 'chunk_new_' in blob.name and 'embeddings' not in blob.name)))

nlist = 100  # You can modify this number based on your requirements
quantizer = faiss.IndexFlatL2(384)  # 384 is the dimension of the embeddings

suffix = "_v2"  # Suffix to append to the filenames to avoid clashes

# Directory to store on-disk indices
on_disk_dir = "faiss_on_disk_indices"
# Check if the directory exists, create if not
if not os.path.exists(on_disk_dir):
    os.makedirs(on_disk_dir)

for file_id in file_ids:
    try:
        # Check if the index file already exists with the new suffix
        index_file = f'paraphrase_test_index_{file_id}{suffix}.faiss'
        blob_index = bucket.blob(index_file)
        if blob_index.exists():
            logger.info(f"Index file {index_file} already exists. Skipping...")
            continue

        logger.info(f"Processing file_id: {file_id}...")

        # Fetch and load associated data
        blob_data = bucket.blob(f'chunk_new_{file_id}.json')
        data_text = blob_data.download_as_text()
        data = json.loads(data_text)

        # Fetch and load embeddings
        blob_emb = bucket.blob(f'embeddings_paraphrase_{file_id}.json')
        emb_text = blob_emb.download_as_text()
        emb_data = json.loads(emb_text)

        # Convert list of embeddings to numpy array
        embeddings_np = np.asarray(emb_data, dtype=np.float32)

        # Build FAISS index
        logger.info("Building FAISS index...")

        # Create an empty on-disk index with the same parameters
        on_disk_index_file_path = os.path.join(on_disk_dir, f"on_disk_index_{file_id}.faiss")
        on_disk_index = faiss.IndexIVFScalarQuantizer(quantizer, 384, nlist, faiss.ScalarQuantizer.QT_8bit)
        on_disk_index.own_invlists = False
        
        # Create the OnDiskInvertedLists object
        on_disk_invlists = faiss.OnDiskInvertedLists(on_disk_index.nlist, on_disk_index.code_size, on_disk_index_file_path)
        
        # Replace the inverted lists of the new index with the on-disk one
        on_disk_index.replace_invlists(on_disk_invlists)

        on_disk_index.train(embeddings_np)  # Training step
        on_disk_index.add(embeddings_np)
        logger.info(f"FAISS IVFScalarQuantizer index built with {on_disk_index.ntotal} vectors and saved on disk.")

        # Upload the on-disk index file to Google Cloud Storage
        blob_on_disk_index = bucket.blob(on_disk_index_file_path)
        blob_on_disk_index.upload_from_filename(on_disk_index_file_path)
        logger.info(f"On-disk FAISS index file {on_disk_index_file_path} uploaded to Google Cloud Storage.")

        # Save associated data to a file with the new suffix
        data_file = f'associated_data_{file_id}{suffix}.json'
        with open(data_file, 'w') as f:
            json.dump(data, f)
        logger.info(f"Associated data file {data_file} created.")

        # Upload the associated data file to Google Cloud Storage
        blob_data = bucket.blob(data_file)
        blob_data.upload_from_filename(data_file)
        logger.info(f"Associated data file {data_file} uploaded to Google Cloud Storage.")

        # Delete local index and data files
        os.remove(on_disk_index_file_path)
        os.remove(data_file)
        logger.info("Local files deleted.")

    except NotFound:
        logger.info(f"File_id: {file_id} not found. Skipping to next file_id...")

logger.info("Finished processing all files.")
