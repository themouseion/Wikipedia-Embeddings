import logging
from google.cloud import storage
import faiss
import numpy as np
import tempfile
import gc  # Garbage collection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define GCS client and bucket
client = storage.Client()
bucket_name = "mouseion-embeddings"  # Replace with your bucket name
bucket = client.get_bucket(bucket_name)

dimension = 384  # Dimension of the vectors
nlist = 100    # Number of clustering centroids; you may need to adjust this
BATCH_SIZE = 100 # Batch size for processing files

# Create a quantizer and combined index using IndexIVFFlat
logger.info("Creating quantizer and combined index with IndexIVFFlat...")
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)

# Fetch blobs
logger.info("Fetching blobs...")
blobs_list = list(bucket.list_blobs(prefix="paraphrase_test_index_"))

# Function to process a batch of blobs
def process_batch(blobs_batch):
    for blob in blobs_batch:
        logger.info(f"Processing blob: {blob.name}")
        index_path = tempfile.mktemp()
        blob.download_to_filename(index_path)
        sub_index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
        vectors = np.zeros((sub_index.ntotal, dimension), dtype=np.float32)
        
        if not sub_index.direct_map.hashtable:
            raise Exception("Direct map is not initialized in the index file; reconstruction is not possible.")
            
        for i in range(sub_index.ntotal):
            vectors[i] = sub_index.reconstruct(i)
        index.add(vectors)
        logger.info(f"Added {sub_index.ntotal} vectors from {blob.name}")
        del sub_index
        gc.collect()

# Iterate over the blobs in the bucket in batches
for i in range(0, len(blobs_list), BATCH_SIZE):
    blobs_batch = blobs_list[i:i + BATCH_SIZE]
    process_batch(blobs_batch)

# Create a Product Quantizer
logger.info("Creating Product Quantizer...")
pq = faiss.ProductQuantizer(dimension, m, 8)

# Assign the Product Quantizer to the combined index
index.pq = pq

# Save the combined index
output_path = "combined_index.faiss"
faiss.write_index(index, output_path)
logger.info(f"Combined index saved to {output_path}")

# Load index as read-only if needed (outside of this script)
# index = faiss.read_index(output_path, faiss.OMP_READ_ONLY)
