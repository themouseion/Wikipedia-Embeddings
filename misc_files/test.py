import json
import numpy as np
import pandas as pd
import gcsfs
from sentence_transformers import SentenceTransformer

# Load SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

bucket_name = 'mouseion-embeddings'
folder_path = 'output_directory_newcode'  # without leading or trailing slash
lines_to_process = 1000  # Lines to be processed
df_list = []

# Create a GCSFileSystem object
fs = gcsfs.GCSFileSystem(project='decisive-octane-393214', token='cloud')

# Get the list of all directories in the bucket folder
directories = fs.ls(f'{bucket_name}/{folder_path}')

# Initialize data before the loop
data = []

for directory in directories:
    filenames = fs.ls(directory)  # List all files in the subdirectory
    for filename in filenames:
        if filename.endswith('.json'):
            with fs.open(f'gs://{filename}', 'r') as file:
                for line in file:
                    data.append(json.loads(line))

                    
                    if len(data) == lines_to_process:
                        # Generate embeddings for each string in the data list
                        embeddings = model.encode([d['text'] for d in data])
                        # Convert the list of arrays to a single 2D array
                        embeddings_array = np.vstack(embeddings)
                        # Convert the array to a DataFrame and append it to the df_list
                        df_list.append(pd.DataFrame(embeddings_array))

                        # Save the DataFrame to a .parquet file in GCS
                        with fs.open(f'gs://{bucket_name}/wikipediaembeddings.parquet', 'wb') as f:
                            pd.concat(df_list).to_parquet(f)

                        # Print message and exit
                        print(f'Processed and saved {lines_to_process} embeddings.')
                        exit(0)
