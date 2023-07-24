from main_celery import process_file
from google.cloud import storage

# Set up GCS client
client = storage.Client()
bucket = client.get_bucket('mouseion-embeddings')

# Queue a task for each file
file_count = 0
for blob in bucket.list_blobs(prefix='output_directory_newcode'):
    if blob.name.endswith('.json'):
        process_file.delay(blob.name)
        file_count += 1
        if file_count >= 1000:  # stop after processing 1000 files
            break
