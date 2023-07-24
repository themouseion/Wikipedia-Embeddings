import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from google.cloud import storage
import faiss
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
    # Generate embeddings for each string in the data list
    embeddings = model.encode([item['text'] for item in data])

    # Convert the list of arrays to a single 2D array
    embeddings_array = np.vstack(embeddings)

    # Create a DataFrame with the embeddings and the actual data
    df = pd.DataFrame(data)
    df['embeddings'] = list(embeddings_array)

    # Save the DataFrame to GCS
    output_bucket_name = 'mouseion-embeddings-output'
    output_bucket = client.get_bucket(output_bucket_name)
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer)
    csv_buffer.seek(0)
    output_bucket.blob('DataFrameWithEmbeddings.csv').upload_from_file(csv_buffer, content_type='text/csv')
    
    # Initialize a FAISS index
    index = faiss.IndexFlatL2(embeddings_array.shape[1])

    # Add vectors to the FAISS index
    index.add(np.asarray(embeddings).astype('float32'))

    # Save the index to a file
    faiss.write_index(index, 'faiss_index')

    # Save the FAISS index to GCS
    with open('faiss_index', 'rb') as index_file:
        output_bucket.blob('faiss_index').upload_from_file(index_file)
else:
    print("No valid JSON files found in the directory.")
