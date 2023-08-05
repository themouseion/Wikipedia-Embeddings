from google.cloud import storage

def list_files(bucket_name):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    chunk_files = [blob.name for blob in bucket.list_blobs(prefix='chunk_new_') if not blob.name.endswith('_embeddings.json')]
    embeddings_files = [blob.name for blob in bucket.list_blobs(prefix='embeddings_paraphrase_')]

    for chunk_file in chunk_files:
        fileID = chunk_file.split('chunk_new_')[-1].split('.json')[0]
        corresponding_embeddings_file = f'embeddings_paraphrase_{fileID}.json'
        
        if corresponding_embeddings_file not in embeddings_files:
            print(f"{chunk_file}")

bucket_name = 'mouseion-embeddings'
list_files(bucket_name)
