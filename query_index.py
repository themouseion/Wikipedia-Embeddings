import numpy as np
from sentence_transformers import SentenceTransformer
import argparse
import json
from google.cloud import storage

# Argument parsing
parser = argparse.ArgumentParser(description='Process a query.')
parser.add_argument('--query', required=True, help='Query for the script')
args = parser.parse_args()

# Load Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Define GCS client
client = storage.Client()
bucket_name = 'mouseion-embeddings'
bucket = client.get_bucket(bucket_name)

# Fetch and load associated_data2 from GCS
blob_data = bucket.blob('associated_data2.json')
data_text = blob_data.download_as_text()
associated_data2 = json.loads(data_text)

# Fetch and load embeddings2 from GCS
blob_emb = bucket.blob('embeddings2.json')
emb_text = blob_emb.download_as_text()
embeddings2 = json.loads(emb_text)

# Convert embeddings2 to list of numpy arrays
embeddings2 = [np.array(emb) for emb in embeddings2]

# Get the query from command line arguments
query = args.query

# Convert the query into a vector using the Sentence Transformer model
query_vector = model.encode([query])[0]

results = []

# Calculate cosine similarity for each vector in our list against the query
for i, emb in enumerate(embeddings2):
    similarity = cosine_similarity(emb, query_vector)
    # Only consider results with similarity > 0.2
    if similarity > 0.1:
        results.append((similarity, associated_data2[i]))

# Sort by similarity
results.sort(key=lambda x: x[0], reverse=True)

# Limit to top 10 results
results = results[:10]

# Print the results
for r in results:
    print(f"Similarity: {r[0]}, Data: {r[1]}")

if not results:
    print("No results found for the given query with similarity greater than 0.2.")
