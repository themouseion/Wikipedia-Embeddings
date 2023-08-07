import os
import json
import nltk
from transformers import AutoTokenizer
from google.cloud import storage
from tenacity import retry, stop_after_attempt, wait_fixed

# This function will be retried up to 5 times, waiting 10 seconds between attempts
@retry(stop=stop_after_attempt(5), wait=wait_fixed(10))
def upload_blob(bucket, blob_name, data):
    blob = bucket.blob(blob_name)
    blob.upload_from_string(json.dumps(data))

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
chunk_counter = 0  # Initialize a new chunk counter

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
                    # Store the sentence, URL, and title
                    data.append({'text': sentence, 'url': json_line['url'], 'title': json_line['title']})
                    sentences_processed += 1
                    if sentences_processed % 100000 == 0:
                        print(f'Processed {sentences_processed} sentences')
                if len(data) >= chunk_size:
                    output_bucket_name = 'mouseion-embeddings'
                    output_bucket = client.get_bucket(output_bucket_name)
                    chunk_counter += 1  # Increment the chunk counter each time we hit chunk_size
                    upload_blob(output_bucket, f'chunk_new_{chunk_counter}.json', data)  # Use chunk_counter to name the blob
                    data = []
