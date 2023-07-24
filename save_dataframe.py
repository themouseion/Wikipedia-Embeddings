import numpy as np
import pandas as pd
from google.cloud import storage
from io import BytesIO

# Load the data and embeddings from previous script
data = np.load('data.npy', allow_pickle=True)
embeddings_array = np.load('embeddings.npy')

# Create a DataFrame with the embeddings and the actual data
df = pd.DataFrame(data)
df['embeddings'] = list(embeddings_array)

# Save the DataFrame to GCS
client = storage.Client()
output_bucket_name = 'mouseion-embeddings-output'
output_bucket = client.get_bucket(output_bucket_name)
csv_buffer = BytesIO()
df.to_csv(csv_buffer)
csv_buffer.seek(0)
output_bucket.blob('DataFrameWithEmbeddings.csv').upload_from_file(csv_buffer, content_type='text/csv')
