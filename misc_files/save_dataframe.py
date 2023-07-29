import os
import numpy as np
import pandas as pd
from google.cloud import storage
from io import BytesIO

output_bucket_name = 'mouseion-embeddings-output'
client = storage.Client()
output_bucket = client.get_bucket(output_bucket_name)

num_chunks = len([name for name in os.listdir('.') if name.startswith('data_') and name.endswith('.npy')])

# Process each chunk separately
for chunk_num in range(num_chunks):
    data = np.load(f'data_{chunk_num}.npy', allow_pickle=True)
    embeddings_array = np.load(f'embeddings_{chunk_num}.npy', allow_pickle=True)

    # Create a DataFrame with the embeddings and the actual data
    df = pd.DataFrame(data)
    df['embeddings'] = list(embeddings_array)

    # Save the DataFrame to GCS
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer)
    csv_buffer.seek(0)
    output_bucket.blob(f'DataFrameWithEmbeddings_{chunk_num}.csv').upload_from_file(csv_buffer, content_type='text/csv')
                                                                                                                                             