import os
import torch
import json
from transformers import AutoModel, AutoTokenizer
from google.cloud import storage
from tenacity import retry, stop_after_attempt, wait_fixed

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

# ranges = [(8, 9), (80, 99), (800, 999), (8000, 9999)]
ranges = [(6, 7), (60, 79), (600, 799), (6000, 7999)]
blobs_to_process = []

for start, end in ranges:
    blobs = [blob for blob in bucket.list_blobs(prefix='chunk_new_') if start <= int(blob.name.split('_')[2].split('.')[0]) <= end]
    blobs_to_process.extend(blobs)

blobs_to_process = list(set(blobs_to_process))

counter = 0
BATCH_SIZE = 1000

print('Starting script...')

for blob in blobs_to_process:
    if blob.name.endswith('.json'):
        print(f'Processing blob {blob.name}...')
        text = blob.download_as_text()
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

        output_bucket_name = 'mouseion-embeddings'
        output_bucket = client.get_bucket(output_bucket_name)
        blob_name = f'{blob.name}_embeddings.json'
        print('Uploading blob...')
        upload_blob(output_bucket, blob_name, json.dumps(all_output_data))

print('Script finished.')
