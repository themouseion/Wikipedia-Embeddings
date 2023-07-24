# This file is only to be used if you need to convert the wiki dump data to JSON. 
# The command you'd run is 'python -m wikiextractor.WikiExtractor -o output_directory --json enwiki-YYYYMMDD-pages-articles.xml.bz2'

import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Function to rename files without extension in a directory and its subdirectories
def rename_files_to_json(directory):
    """
    Renames files in a directory (and its subdirectories) that don't have an extension to .json.

    Args:
        directory (str): The path to the directory containing the files.

    Returns:
        None
    """
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            # Get the file extension
            _, ext = os.path.splitext(filename)
            
            # If the file doesn't have an extension, rename it to have .json
            if ext == "":
                src = os.path.join(dirpath, filename)
                dst = os.path.join(dirpath, filename + ".json")
                os.rename(src, dst)

# Load SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

data = []
folder_path = '/Users/alexcarrabre/Downloads/output_directory_newcode'

# Rename files in the directory
rename_files_to_json(folder_path)

for dirpath, dirnames, filenames in os.walk(folder_path):
    for filename in filenames:
        if filename.endswith('.json'):
            with open(os.path.join(dirpath, filename), 'r') as file:
                for line in file:
                    data.append(json.loads(line))

# Check if there's any data to process
if data:
    # Generate embeddings for each string in the data list
    embeddings = model.encode(data)

    # Convert the list of arrays to a single 2D array
    embeddings_array = np.vstack(embeddings)

    # Convert the array to a DataFrame
    embeddings_df = pd.DataFrame(embeddings_array)

    # Save the DataFrame to a .parquet file
    embeddings_df.to_parquet('wikipediaembeddings.parquet')
else:
    print("No valid JSON files found in the directory.")

# Load the DataFrame
embeddings_loaded = pd.read_parquet('wikipediaembeddings.parquet')

# Print the loaded embeddings
print(embeddings_loaded)
