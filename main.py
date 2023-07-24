import os
import json
from transformers import AutoTokenizer
import numpy as np
from sentence_transformers import SentenceTransformer

# Disable parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
max_tokens = 512

folder_path = '/Users/alexcarrabre/Downloads/output_directory_newcode'
batch_size = 100

embeddings = []
lines_processed = 0

for dirpath, dirnames, filenames in os.walk(folder_path):
    for filename in filenames:
        if filename.endswith('.json'):
            with open(os.path.join(dirpath, filename), 'r') as file:
                data = []
                for line in file:
                    json_line = json.loads(line)
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
                    
                    if len(data) == batch_size:
                        batch_embeddings = model.encode(data)
                        embeddings.extend(batch_embeddings)
                        data = []

                    lines_processed += 1

                    if lines_processed % 1000 == 0:
                        print(f'Processed {lines_processed} lines')

# Process any remaining data
if data:
    batch_embeddings = model.encode(data)
    embeddings.extend(batch_embeddings)

embeddings_array = np.vstack(embeddings)
np.save('wikipediaembeddings.npy', embeddings_array)

# Load embeddings
embeddings_loaded = np.load('wikipediaembeddings.npy')

# If you need a DataFrame for querying
import pandas as pd
embeddings_df = pd.DataFrame(embeddings_loaded)
print(embeddings_df)
