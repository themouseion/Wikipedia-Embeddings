from google.cloud import storage
import os

# Define GCS client and bucket
client = storage.Client()
bucket_name = "mouseion-embeddings"
bucket = client.get_bucket(bucket_name)

# Local directory path
local_directory_path = "/home/alexcarrabre/combined_data.parquet" # Change this to the correct directory path

# Iterate through all files in the directory
for filename in os.listdir(local_directory_path):
    local_file_path = os.path.join(local_directory_path, filename)
    
    if os.path.isfile(local_file_path): # Only process files
        # Create blob
        blob = bucket.blob(filename)
        
        # Upload file
        blob.upload_from_filename(local_file_path)
        print(f"File {filename} uploaded.")
