import os
import json
from sentence_transformers import SentenceTransformer

# Load SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

data = []
folder_path = '/path/to/your/main/folder'

for dirpath, dirnames, filenames in os.walk(folder_path):
    for filename in filenames:
        if filename.endswith('.json'):
            with open(os.path.join(dirpath, filename), 'r') as file:
                for line in file:
                    data.append(json.loads(line))

# Generate embeddings for each string in the data list
embeddings = model.encode(data)
