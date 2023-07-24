import os
import json
from transformers import AutoTokenizer
import numpy as np
from sentence_transformers import SentenceTransformer
from google.cloud import storage
import pandas as pd
from io import BytesIO

# Disable parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
max_tokens = 512

bucket_name = 'mouseion-embeddings'
folder_path = 'output_directory_newcode'
batch_size = 100

client = storage.Client()
bucket = client.get_bucket(bucket_name)

embeddings = []
lines_processed = 0

blobs = bucket.list_blobs(prefix=folder_path)

for blob in blobs:
    if blob.name.endswith('.json'):
        json_data = [json.loads(line) for line in blob.download_as_text().splitlines()]
        data = []
        text = ''  # Set a default value for 'text'
        for json_line in json_data:
            text = json_line['text']
            inputs = tokenizer.encode_plus(
                text, 
                max_length=max_tokens,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            inputs = inputs.input_ids.squeeze().tolist()
            text = tokenizer.decode(inputs)
            data.append(text)
            
            if len(data) == batch_size:
                batch_embeddings = model.encode(data)
                embeddings.extend(batch_embeddings)
                data = []

            lines_processed += 1

            if lines_processed % 1000 == 0:
                print(f'Processed {lines_processed} lines')

# Process any remaining data
if data:
    batch_embeddings = model.encode(data)
    embeddings.extend(batch_embeddings)

embeddings_array = np.vstack(embeddings)

# Save embeddings to a new bucket
output_bucket_name = 'mouseion-embeddings-output'
output_bucket = client.get_bucket(output_bucket_name)

# Create a BytesIO object, save array to it, then upload that to GCS
b = BytesIO()
np.save(b, embeddings_array)
b.seek(0)

blob = output_bucket.blob('wikipediaembeddingsdf.npy')
blob.upload_from_file(b)

# If you need a DataFrame for querying
embeddings_df = pd.DataFrame(embeddings_array)
print(embeddings_df)
