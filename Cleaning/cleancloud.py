from google.cloud import storage
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def delete_files(bucket_name, file_prefixes):
    """
    Delete files from the Google Cloud Storage bucket with the specified prefixes.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for prefix in file_prefixes:
        blobs = bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            logging.info(f"Deleting file: {blob.name}")
            blob.delete()

def main():
    # Specify your Google Cloud Storage bucket name
    bucket_name = 'your-bucket-name'  # Replace with your bucket name

    # Prefixes of the files to be deleted
    file_prefixes = ['my_usearch_index_', 'metadata_', 'clustering_']

    # Delete the files
    delete_files(bucket_name, file_prefixes)
    logging.info("Files deletion completed.")

if __name__ == "__main__":
    main()
