import json
import numpy as np
import argparse
import logging
from sentence_transformers import SentenceTransformer
from usearch.index import Index

# Set up logging
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

# Load the pre-trained sentence transformer model
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

# Load the index from disk
try:
    index = Index.restore("index_full.usearch", view=True)
    if index is None:
        raise ValueError("Index not found or is None.")
except Exception as e:
    logging.error(f"Failed to load index: {e}")
    exit(1)

# Load metadata from disk
try:
    with open('full_index_metadata.json', 'r') as f:
        metadata_dict = json.load(f)
except Exception as e:
    logging.error(f"Failed to load metadata: {e}")
    exit(1)

embeddings_cache = {}

def get_embedding(text):
    if text in embeddings_cache:
        return embeddings_cache[text]
    embedding = model.encode(text, convert_to_tensor=True).numpy()
    embeddings_cache[text] = embedding
    return embedding

def search_index(query_text, top_k=10):
    query_vector = get_embedding(query_text)
    matches = index.search(query_vector, top_k)
    return matches

def main(query_text):
    results = search_index(query_text, top_k=10)
    for match in results:
        unique_id = match.key
        distance = match.distance
        metadata = metadata_dict.get(str(unique_id), {})  # Convert unique_id to string if necessary
        result_info = f"Unique ID: {unique_id}, Distance: {distance}, Metadata: {metadata}"
        print(result_info)  # Print each result to console

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search in the USearch index using a text query.")
    parser.add_argument("query_text", type=str, help="Text to query in the index")
    args = parser.parse_args()
    main(args.query_text)
