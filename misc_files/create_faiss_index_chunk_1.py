import os
import numpy as np
import pandas as pd
import faiss
from google.cloud import storage
from tenacity import retry, stop_after_attempt, wait_fixed

# This function will be retried up to 5 times, waiting 10 seconds between attempts
@retry(stop=stop_after_attempt(5), wait=wait_fixed(10))
def upload_blob(bucket, blob_name, data):
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data)

df = pd.read_pickle('embeddings_df.pkl')

# Create a matrix X of stored vectors
X = np.array(df['embedding'].tolist()).astype
