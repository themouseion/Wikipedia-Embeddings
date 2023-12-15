from google.cloud import storage
import json
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Google Cloud Storage Client
client = storage.Client()
bucket_name = 'mouseion-embeddings'
bucket = client.bucket(bucket_name)

def check_file(file_name, start_id, end_id):
    logging.info(f"Checking file: {file_name}")
    blob = bucket.blob(file_name)

    # Check if the file exists in the bucket
    if not blob.exists():
        logging.error(f"File does not exist: {file_name}")
        return False

    data = json.loads(blob.download_as_text())
    unique_ids = [item['unique_id'] for item in data]

    expected_ids = set(range(start_id, end_id + 1))
    actual_ids = set(unique_ids)

    if actual_ids != expected_ids:
        missing_ids = expected_ids - actual_ids
        extra_ids = actual_ids - expected_ids
        logging.warning(f"File {file_name} has incorrect unique IDs. Missing IDs: {missing_ids}, Extra IDs: {extra_ids}")
        return False

    logging.info(f"File {file_name} passed the check.")
    return True

def main():
    start_file_number = 3225  # Start checking from file number 2600
    total_ids = 136270000
    ids_per_file = 10000
    total_files = total_ids // ids_per_file
    error_files = []

    for i in range(start_file_number - 1, total_files):
        start_id = i * ids_per_file
        end_id = start_id + ids_per_file - 1
        file_name = f'embeddings_paraphrase_{i + 1}.json'
        if not check_file(file_name, start_id, end_id):
            logging.error(f"Error in file: {file_name}")
            error_files.append(file_name)

    # Writing error files to disk
    with open('error_files_list.txt', 'w') as file:
        for file_name in error_files:
            file.write(f"{file_name}\n")

    logging.info("Completed checking all files.")

if __name__ == "__main__":
    main()
