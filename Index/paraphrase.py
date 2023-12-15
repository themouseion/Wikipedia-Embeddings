import json
import logging
from sentence_transformers import SentenceTransformer
from google.cloud import storage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Sentence Transformer model
logger.info("Loading Sentence Transformer model...")
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

# Define GCS client
logger.info("Setting up Google Cloud Storage client...")
client = storage.Client()
bucket_name = 'mouseion-embeddings'
bucket = client.get_bucket(bucket_name)

# Get all blobs in the bucket
blobs = bucket.list_blobs(prefix='chunk_new_')
file_ids = []
for blob in blobs:
    blob_name_parts = blob.name.split('_')
    if len(blob_name_parts) < 3:
        # Skip blobs that don't have the format chunk_new_{id}.json
        continue
    try:
        # Extract file id
        file_id = int(blob_name_parts[-1].split('.')[0])
        file_ids.append(file_id)
    except ValueError:
        continue

# Sorting file_ids to ensure consistency across different machines
file_ids.sort()

file_ids_to_process = range(1, 2)
# range(1, 12)  # specify the file IDs you want to process

for file_id in file_ids_to_process:
    logger.info(f"Processing file_id: {file_id}...")

    blob_data = bucket.blob(f'chunk_new_{file_id}.json')
    if not blob_data.exists(client):
        logger.warning(f"Missing: chunk_new_{file_id}.json")
        continue

    # Fetch and load data from GCS
    data_text = blob_data.download_as_text()
    data = json.loads(data_text)

    # Create embeddings
    texts = [item['text'] for item in data]
    embeddings = model.encode(texts)

    # Save embeddings to a new JSON file
    embeddings = embeddings.tolist()
    embeddings_blob_name = f'embeddings_paraphrase_{file_id}.json'
    blob_emb = bucket.blob(embeddings_blob_name)
    blob_emb.upload_from_string(json.dumps(embeddings))

    logger.info(f"Finished processing file_id: {file_id}.")

logger.info("All files processed.")



# import json
# import logging
# from sentence_transformers import SentenceTransformer
# from google.cloud import storage

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load Sentence Transformer model
# logger.info("Loading Sentence Transformer model...")
# model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

# # Define GCS client
# logger.info("Setting up Google Cloud Storage client...")
# client = storage.Client()
# bucket_name = 'mouseion-embeddings'
# bucket = client.get_bucket(bucket_name)

# # Get all blobs in the bucket
# blobs = bucket.list_blobs(prefix='chunk_new_')
# file_ids = []
# for blob in blobs:
#     blob_name_parts = blob.name.split('_')
#     if len(blob_name_parts) < 3:
#         # Skip blobs that don't have the format chunk_new_{id}.json
#         continue
#     try:
#         # Extract file id
#         file_id = int(blob_name_parts[-1].split('.')[0])
#         file_ids.append(file_id)
#     except ValueError:
#         continue

# # Sorting file_ids to ensure consistency across different machines
# file_ids.sort()

# # Divide the workload across different machines
# # num_machines = 8  # adjust this to your actual number of machines
# # machine_id = 8  # adjust this to each specific machine: 1, 2, ..., num_machines
# # total_files = len(file_ids)
# # start = int((machine_id - 1) * total_files / num_machines)
# # end = int(machine_id * total_files / num_machines)
# file_ids_to_process = range(1, 10)  # specify the file IDs you want to process

# for file_id in file_ids_to_process:
#     logger.info(f"Processing file_id: {file_id}...")

#     # Check if the embeddings for this file_id already exist
#     embeddings_blob_name = f'embeddings_paraphrase_{file_id}.json'
#     if storage.Blob(bucket=bucket, name=embeddings_blob_name).exists(client):
#         logger.info(f"Embeddings for file_id {file_id} already exist. Skipping...")
#         continue

#     blob_data = bucket.blob(f'chunk_new_{file_id}.json')
#     if not blob_data.exists(client):
#         logger.warning(f"Missing: chunk_new_{file_id}.json")
#         continue

#     # Fetch and load data from GCS
#     data_text = blob_data.download_as_text()
#     data = json.loads(data_text)

#     # Create embeddings
#     texts = [item['text'] for item in data]
#     embeddings = model.encode(texts)

#     # Save embeddings to a new JSON file
#     embeddings = embeddings.tolist()
#     blob_emb = bucket.blob(embeddings_blob_name)
#     blob_emb.upload_from_string(json.dumps(embeddings))

#     logger.info(f"Finished processing file_id: {file_id}.")

# logger.info("All files processed.")
