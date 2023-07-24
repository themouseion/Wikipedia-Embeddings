import os
import json
import nltk
from transformers import AutoTokenizer
from google.cloud import storage

# Set up tokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
nltk.download('punkt')

data = []
bucket_name = 'mouseion-embeddings'
client = storage.Client()
bucket = client.get_bucket(bucket_name)

# Define a number of sentences that constitute a chunk
chunk_size = 10000  # adjust this number based on your specific situation

sentences_processed = 0
blobs = bucket.list_blobs(prefix='output_directory_newcode')
for blob in blobs:
    if blob.name.endswith('.json'):
        text = blob.download_as_text()
        for line in text.splitlines():
            json_line = json.loads(line)
            sentences = nltk.sent_tokenize(json_line['text'])
            for sentence in sentences:
                tokens = tokenizer.tokenize(sentence)
                if len(tokens) <= 512:
                    data.append({'text': sentence})
                    sentences_processed += 1
                    if sentences_processed % 100000 == 0:
                        print(f'Processed {sentences_processed} sentences')
                if len(data) >= chunk_size:
                    output_bucket_name = 'mouseion-embeddings'
                    output_bucket = client.get_bucket(output_bucket_name)
                    chunk_num = len(data) // chunk_size
                    blob = output_bucket.blob(f'chunk_{chunk_num}.json')
                    blob.upload_from_string(json.dumps(data))
                    data = []
