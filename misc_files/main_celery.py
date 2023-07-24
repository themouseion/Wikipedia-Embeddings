# run this command 'celery -A main_celery worker --concurrency=1 -l debug'


from celery import Celery
import os
import json
from transformers import AutoTokenizer
import numpy as np
from sentence_transformers import SentenceTransformer
from google.cloud import storage
import logging

# Set up logging
logging.basicConfig(filename='file_processing.log', level=logging.INFO)

# Set up Celery
app = Celery('tasks', broker='redis://127.0.0.1:6379/0')

# Disable parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up the transformer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
max_tokens = 512
batch_size = 100

@app.task(bind=True)
def process_file(self, blob_name):
    logging.info(f"Starting to process file {blob_name}")

    # Set up GCS client
    client = storage.Client()
    bucket = client.get_bucket('mouseion-embeddings')

    # Download the file from GCS
    blob = bucket.blob(blob_name)
    json_data = [json.loads(line) for line in blob.download_as_text().splitlines()]

    # Initialize variables
    embeddings = []
    lines_processed = 0
    data = []

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
        
        if len(data) >= batch_size:
            logging.info("Starting to encode batch")
            batch_embeddings = model.encode(data)
            logging.info("Finished encoding batch")
            embeddings.extend(batch_embeddings)
            data = []

        lines_processed += 1

        if lines_processed % 1000 == 0:
            logging.info(f'Processed {lines_processed} lines for file {blob_name}')

    # Process any remaining data
    if data:
        batch_embeddings = model.encode(data)
        embeddings.extend(batch_embeddings)

    embeddings_array = np.vstack(embeddings)

    # Save embeddings to a new bucket
    output_bucket_name = 'mouseion-embeddings-output'
    output_bucket = client.get_bucket(output_bucket_name)
    blob = output_bucket.blob(f'{blob_name}_embeddings.npy')
    blob.upload_from_string(embeddings_array.tobytes())

    logging.info(f'Finished processing file {blob_name}')
    return 'SUCCESS'

