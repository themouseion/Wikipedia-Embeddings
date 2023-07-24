import numpy as np
import faiss
from google.cloud import storage

# Load the embeddings from first script
embeddings_array = np.load('embeddings.npy')

# Initialize a FAISS index
index = faiss.IndexFlatL2(embeddings_array.shape[1])

# Add vectors to the FAISS index
index.add(np.asarray(embeddings_array).astype('float32'))

# Save the index to a file
faiss.write_index(index, 'faiss_index')

# Save the FAISS index to GCS
client = storage.Client()
output_bucket_name = 'mouseion-embeddings-output'
output_bucket = client.get_bucket(output_bucket_name)
with open('faiss_index', 'rb') as index_file:
    output_bucket.blob('faiss_index').upload_from_file(index_file)
