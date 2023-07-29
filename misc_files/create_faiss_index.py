import os
import pandas as pd
import numpy as np
import faiss
from google.cloud import storage

output_bucket_name = 'mouseion-embeddings-output'
client = storage.Client()
output_bucket = client.get_bucket(output_bucket_name)

num_chunks = len([name for name in os.listdir('.') if name.startswith('data_') and name.endswith('.npy')])

# Process each chunk separately
for chunk_num in range(num_chunks):
    embeddings_array = np.load(f'embeddings_{chunk_num}.npy', allow_pickle=True)

    # Create and train the index
    index = faiss.IndexFlatL2(embeddings_array.shape[1])
    index.add(embeddings_array)

    #
