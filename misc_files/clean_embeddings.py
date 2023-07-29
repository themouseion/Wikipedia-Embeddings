from google.cloud import storage

def delete_blob(bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.delete()

    print(f'Blob {blob_name} deleted.')

bucket_name = 'mouseion-embeddings'
storage_client = storage.Client()
bucket = storage_client.get_bucket(bucket_name)

blobs = bucket.list_blobs()

for blob in blobs:
    if blob.name.count('_embeddings.json') > 1:
        delete_blob(bucket_name, blob.name)
