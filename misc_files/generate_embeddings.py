import os
import torch
import json
from transformers import AutoModel, AutoTokenizer
from google.cloud import storage
from tenacity import retry, stop_after_attempt, wait_fixed
from google.api_core.exceptions import NotFound

@retry(stop=stop_after_attempt(5), wait=wait_fixed(10))
def upload_blob(bucket, blob_name, data):
    try:
        print('Starting blob upload...')
        blob = bucket.blob(blob_name)
        blob.upload_from_string(data)
        print('Blob upload completed.')
    except Exception as e:
        print(f'Blob upload failed with error: {e}')
        raise e

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

bucket_name = 'mouseion-embeddings'
client = storage.Client()
bucket = client.get_bucket(bucket_name)

blobs = bucket.list_blobs(prefix='chunk_new_')
counter = 0
BATCH_SIZE = 1000

output_bucket_name = 'mouseion-embeddings'
output_bucket = client.get_bucket(output_bucket_name)

print('Starting script...')

for blob in blobs:
    if blob.name.endswith('.json'):
        # Check if embedding file already exists
        embedding_blob_name = f'{blob.name}_embeddings.json'
        if storage.Blob(bucket=output_bucket, name=embedding_blob_name).exists(client):
            print(f'Skipping blob {blob.name} because embeddings already exist.')
            continue

        try:
            print('Downloading blob...')
            text = blob.download_as_text()
        except NotFound:
            print(f'Blob {blob.name} was not found.')
            continue

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            print(f'Could not decode JSON from blob {blob.name}')
            continue

        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            sentences = [item.get('text', '') for item in data]
            urls = [item.get('url', '') for item in data]
            titles = [item.get('title', '') for item in data]
        else:
            print(f'Unexpected data format in blob {blob.name}')
            continue

        num_batches = int(len(sentences) / BATCH_SIZE) + (len(sentences) % BATCH_SIZE != 0)

        all_output_data = []

        for i in range(num_batches):
            print(f'Processing batch {i+1}/{num_batches}')
            batch_sentences = sentences[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

            print('Tokenizing inputs...')
            inputs = tokenizer(batch_sentences, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                print('Getting model outputs...')
                outputs = model(**inputs)

            embeddings = outputs.pooler_output.tolist()
            print('Creating output data...')
            output_data = [{'embedding': embedding, 'url': url, 'title': title} for embedding, url, title in zip(embeddings, urls[i*BATCH_SIZE:(i+1)*BATCH_SIZE], titles[i*BATCH_SIZE:(i+1)*BATCH_SIZE])]
            all_output_data.extend(output_data)

            counter += len(batch_sentences)
            if counter % 100000 == 0:
                print(f'{counter} sentences embedded')

        # Clean the blob name before adding the suffix
        clean_blob_name = blob.name.replace('_embeddings.json', '')
        blob_name = f'{clean_blob_name}_embeddings.json'
        
        print('Uploading blob...')
        upload_blob(output_bucket, blob_name, json.dumps(all_output_data))

print('Script finished.')
