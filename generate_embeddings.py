import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from google.cloud import storage
from io import BytesIO

# Load SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

data = []
bucket_name = 'mouseion-embeddings'
client = storage.Client()
bucket = client.get_bucket(bucket_name)
lines_processed = 0

blobs = bucket.list_blobs(prefix='output_directory_newcode')
for blob in blobs:
    if blob.name.endswith('.json'):
        text = blob.download_as_text()
        for line in text.splitlines():
            json_line = json.loads(line)
            data.append(json_line)
            lines_processed += 1
            if lines_processed % 10000 == 0:
                print(f'Processed {lines_processed} lines')

# Check if there's any data to process
if data:
    # Split data into three parts
    data_1, data_2, data_3 = np.array_split(data, 3)

    # Generate embeddings for each string in the data list for each part
    embeddings_1 = model.encode([item['text'] for item in data_1])
    embeddings_2 = model.encode([item['text'] for item in data_2])
    embeddings_3 = model.encode([item['text'] for item in data_3])

    # Convert the list of arrays to a single 2D array for each part
    embeddings_array_1 = np.vstack(embeddings_1)
    embeddings_array_2 = np.vstack(embeddings_2)
    embeddings_array_3 = np.vstack(embeddings_3)

    # Save the data and embeddings locally for next script for each part
    np.save('data_1.npy', data_1)
    np.save('embeddings_1.npy', embeddings_array_1)

    np.save('data_2.npy', data_2)
    np.save('embeddings_2.npy', embeddings_array_2)

    np.save('data_3.npy', data_3)
    np.save('embeddings_3.npy', embeddings_array_3)

else:
    print("No valid JSON files found in the directory.")
