import pandas as pd
import argparse
from typing import List
import gcsfs

def search_wikipedia(query: str) -> List[str]:
    # Initialize a GCSFileSystem object
    fs = gcsfs.GCSFileSystem()

    # Define the path to your parquet file on GCS
    gcs_path = 'gs://mouseion-embeddings/wikipediaembeddings.parquet'

    # Load the Parquet file from GCS
    with fs.open(gcs_path, 'rb') as f:
        df = pd.read_parquet(f)

    # Lowercase the query so the search is case-insensitive
    query = query.lower()

    # Filter the DataFrame based on the query
    results = df[df['content'].str.contains(query, case=False, na=False)]

    # Return the titles of the matching articles
    return results['title'].tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Search a Parquet file for a given query.')
    parser.add_argument('query', type=str, help='The search query.')
    args = parser.parse_args()

    results = search_wikipedia(args.query)
    
    # Print the results
    for title in results:
        print(title)
