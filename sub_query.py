import json
from google.cloud import storage

# Define GCS client
client = storage.Client()
bucket_name = 'mouseion-embeddings' 
bucket = client.get_bucket(bucket_name)

# List of files
file_ids = [1, 10, 100, 1000, 10000, 10001, 10002, 10003, 10004, 10005, 10006]

associated_data2 = []
embeddings2 = []

for file_id in file_ids:
    # Fetch and load associated data
    blob_data = bucket.blob(f'chunk_new_{file_id}.json')
    data_text = blob_data.download_as_text()
    data = json.loads(data_text)

    # Fetch and load embeddings
    blob_emb = bucket.blob(f'chunk_new_{file_id}.json_embeddings.json')
    emb_text = blob_emb.download_as_text()
    emb_data = json.loads(emb_text)
    
    # Store data and embeddings
    for item, emb in zip(data, emb_data):
        associated_data2.append(item)  # associated data
        embeddings2.append(emb['embedding'])  # embeddings

# Save associated_data2 to a JSON file
with open('associated_data2.json', 'w') as f:
    json.dump(associated_data2, f)

# Save embeddings2 to a JSON file
with open('embeddings2.json', 'w') as f:
    json.dump(embeddings2, f)

# Upload associated_data2 to GCS
blob_data = bucket.blob('associated_data2.json')
blob_data.upload_from_filename('associated_data2.json')

# Upload embeddings2 to GCS
blob_emb = bucket.blob('embeddings2.json')
blob_emb.upload_from_filename('embeddings2.json')
