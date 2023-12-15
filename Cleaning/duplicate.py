import csv
import json
from google.cloud import storage

# Google Cloud Storage setup
client = storage.Client()
bucket_name = 'mouseion-embeddings'
bucket = client.bucket(bucket_name)

# Parameters
total_files = 13627  # Total number of embedding files
duplicate_ids_file = 'duplicate_ids.csv'  # File to store duplicate IDs

# Function to process files and find duplicates
def process_files_for_duplicates():
    seen_ids = set()
    duplicates = set()

    for file_number in range(1, total_files + 1):
        emb_blob_name = f'embeddings_paraphrase_{file_number}.json'
        emb_blob = bucket.blob(emb_blob_name)

        if emb_blob.exists():
            print(f"Processing file: {emb_blob_name}")
            embeddings_data = json.loads(emb_blob.download_as_text())
            for data in embeddings_data:
                unique_id = data['unique_id']
                if unique_id in seen_ids:
                    duplicates.add(unique_id)
                else:
                    seen_ids.add(unique_id)
        else:
            print(f"File not found: {emb_blob_name}")

    return duplicates

# Running the duplicate ID check
duplicate_ids = process_files_for_duplicates()

# Writing duplicates to a CSV file
with open(duplicate_ids_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Duplicate ID'])
    for duplicate_id in duplicate_ids:
        writer.writerow([duplicate_id])

print(f"Duplicate IDs written to {duplicate_ids_file}")
