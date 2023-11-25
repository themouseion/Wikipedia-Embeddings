import numpy as np
import faiss
import json
import sys
import logging
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_index(index_path):
    return faiss.read_index(index_path)

def load_ids(ids_path):
    with open(ids_path, 'r') as f:
        return json.load(f)

def query_index(index, ids, query_vector, k=10):
    D, I = index.search(np.array([query_vector]).astype('float32'), k)
    return [(ids[i], d) for i, d in zip(I[0], D[0])]

if __name__ == "__main__":
    index_path = "./on_disk_index/trained.index"
    ids_path = "unique_ids.json"
    query = sys.argv[1]

    # Load the sentence transformer model
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

    # Convert the query to a vector
    logger.info(f"Converting query to vector: {query}")
    query_vector = model.encode([query])[0]

    logger.info("Loading index and IDs.")
    index = load_index(index_path)
    ids = load_ids(ids_path)

    logger.info("Querying index.")
    results = query_index(index, ids, query_vector)
    for id, distance in results:
        logger.info(f"ID: {id}, Distance: {distance}")