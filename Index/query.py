import numpy as np
import argparse
import logging
import json
from sentence_transformers import SentenceTransformer
from usearch.index import Index, Indexes
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load the pre-trained sentence transformer model
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

# Load the indexes from disk
indexes = [Index.restore(f"my_usearch_index_{i}.usearch", view=True) for i in range(1, 11)]
multi_index = Indexes(indexes=indexes)

# Load the metadata from disk
metadata_dict = {}
for i in range(1, 324):
    with open(f'metadata_{i}.json', 'r') as f:
        metadata_dict.update(json.load(f))

def get_embedding(text):
    # Encode the text to get the embedding
    return model.encode(text, convert_to_tensor=True).numpy()

def search_index(query_text, top_k=10):
    logging.info(f"Generating embedding for query: '{query_text}'")
    # Convert query text to an embedding
    query_vector = get_embedding(query_text)
    logging.info(f"Query vector: {query_vector}")

    logging.info(f"Searching index for top {top_k} matches")

    # Create a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit a search task for each index to the executor
        future_to_index = {executor.submit(index.search, query_vector, top_k): index for index in indexes}
        matches = []
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                # Get the results of the search
                index_matches = future.result()
                matches.extend(index_matches)
            except Exception as exc:
                logging.error(f"{index} generated an exception: {exc}")

    # Sort the matches by distance and keep only the top k
    matches.sort(key=lambda x: x.distance)
    matches = matches[:top_k]

    if matches:
        logging.info(f"Found {len(matches)} matches")
    else:
        logging.info("No matches found")

    return matches

def main(query_text):
    results = search_index(query_text, top_k=10)
    logging.info(f"Results for query: {query_text}")
    print(f"Results for query: {query_text}")  # Print to console
    for match in results:
        unique_id = match.key
        result_info = f"Key: {match.key}, Unique ID: {unique_id}, Distance: {match.distance}"
        logging.info(result_info)
        print(result_info)  # Print to console

        # Print the associated metadata
        metadata = metadata_dict[str(unique_id)]
        print(f"Metadata: {metadata}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search in the USearch index using a text query."
    )
    parser.add_argument("query_text", type=str, help="Text to query in the index")
    args = parser.parse_args()
    main(args.query_text)