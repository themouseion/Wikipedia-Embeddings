import json
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from usearch.index import Index

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load pre-trained SentenceTransformer model
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

# Load the USearch index from disk
index = Index.restore("500_index.usearch", view=True)

# Load metadata from the correct file
with open('500e_metadata.json') as f:
    metadata = json.load(f)

def get_embedding(text):
    """
    Generate an embedding for the given text using the pre-trained model.
    """
    return model.encode(text, convert_to_tensor=True).numpy()

def search_index(query_text, top_k=10):
    """
    Search the index with the query text and return the top_k results along with their metadata.
    """
    logging.info(f"Generating embedding for query: '{query_text}'")
    query_vector = get_embedding(query_text)

    logging.info("Searching index for top matches")
    matches = index.search(query_vector, top_k)

    results = []
    for match in matches:
        unique_id = match.key
        distance = match.distance
        match_metadata = metadata.get(str(unique_id), {})  # Convert unique_id to string if necessary
        results.append((unique_id, distance, match_metadata))

    return results

def main():
    query_text = input("Enter your query: ")
    results = search_index(query_text, top_k=10)

    logging.info(f"Results for query: {query_text}")
    for unique_id, distance, match_metadata in results:
        print(f"Unique ID: {unique_id}, Distance: {distance}, Metadata: {match_metadata}")

if __name__ == "__main__":
    main()
