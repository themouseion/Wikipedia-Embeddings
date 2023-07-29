import os
import pandas as pd
import json
from google.cloud import storage
from tenacity import retry, stop_after_attempt, wait_fixed

# This function will be retried up to 5 times, waiting 10 seconds between attempts
@retry(stop=stop_after_attempt(5), wait=wait_fixed(10))
def upload_blob(bucket, blob_name, data):
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data)

bucket_name = 'mouseion-embeddings'
client = storage.Client()
bucket = client.get_bucket(bucket_name)

blobs = bucket.list_blobs(prefix='chunk_1_embeddings')
for blob in blobs:
    if blob.name.endswith('.json'):
        text = blob.download_as_text()
        data = json.loads(text)
        df = pd.DataFrame(data)
        df.to_pickle("embeddings_df.pkl")
        blob = bucket.blob("embeddings_df.pkl")
        upload_blob(blob, df.to_pickle())
