import numpy as np
import faiss
import json
import os
from google.cloud import storage

# Define GCS client
client = storage.Client()
bucket_name = 'mouseion-embeddings'  # replace with your bucket name
bucket = client.get_bucket(bucket_name)

# List of files
file_ids = [1, 10, 100, 1000, 10000, 10001, 10002, 10003, 10004, 10005, 10006]

associated_data = []
embeddings = []

for file_id in file_ids:
    # Fetch and load associated data
    blob_data = bucket.blob(f'chunk_new_{file_id}.json')
    data_text = blob_data.download_as_text()
    data = json.loads(data_text)

    # Fetch and load embeddings
    blob_emb = bucket.blob(f'chunk_new_{file_id}.json_embeddings.json')
    emb_text = blob_emb.download_as_text()
    emb_data = json.loads(emb_text)
    
    # Store data and embeddings
    for item, emb in zip(data, emb_data):
        associated_data.append(item)  # associated data
        embeddings.append(emb['embedding'])  # embeddings

# Convert list of embeddings to numpy array
embeddings_np = np.asarray(embeddings, dtype=np.float32)

# Define parameters for the FAISS index
dim = len(embeddings_np[0])  # dimension of the vectors
nlist = 50  # Number of Voronoi cells (i.e., clusters)
k = 4  # Number of cells to visit during search

# Build FAISS index
quantizer = faiss.IndexFlatL2(dim)  # Quantizer defines the Voronoi cells
index = faiss.IndexIVFFlat(quantizer, dim, nlist)

assert not index.is_trained
index.train(embeddings_np)
assert index.is_trained

index.nprobe = k  # Set the number of cells to visit during search

index.add(embeddings_np)

print(f"FAISS index built with {index.ntotal} vectors.")

# Save the index to a file
index_file = 'my_index.faiss'
faiss.write_index(index, index_file)

# Upload the index file to Google Cloud Storage
blob_index = bucket.blob(index_file)
blob_index.upload_from_filename(index_file)

print(f"FAISS index file {index_file} uploaded to Google Cloud Storage.")

# Save associated data to a file
data_file = 'associated_data.json'
with open(data_file, 'w') as f:
    json.dump(associated_data, f)

# Upload the associated data file to Google Cloud Storage
blob_data = bucket.blob(data_file)
blob_data.upload_from_filename(data_file)

print(f"Associated data file {data_file} uploaded to Google Cloud Storage.")

# Delete local index and data files
os.remove(index_file)
os.remove(data_file)
