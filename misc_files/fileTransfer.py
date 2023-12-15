import os
from google.cloud import storage

def upload_to_bucket(bucket_name, source_file_paths):
    """Uploads specified files to the given GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for source_file_path in source_file_paths:
        if not os.path.isfile(source_file_path):
            print(f"File not found: {source_file_path}")
            continue

        blob = bucket.blob(os.path.basename(source_file_path))
        blob.upload_from_filename(source_file_path)
        print(f"File {source_file_path} uploaded to {bucket_name}.")

def main():
    bucket_name = 'mouseion-embeddings'
    file_paths = []

    # Example: ['path/to/your/file1.txt', 'path/to/your/file2.txt']
    n = int(input("Enter the number of files to upload: "))
    for _ in range(n):
        file_path = input("Enter the path of the file: ")
        file_paths.append(file_path)

    upload_to_bucket(bucket_name, file_paths)

if __name__ == "__main__":
    main()
