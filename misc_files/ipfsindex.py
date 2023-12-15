import subprocess
import json
import logging

# Configure logging to the most detailed level
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def download_file_with_curl(cid, output_filename):
    try:
        ipfs_gateway_url = f"https://dweb.link/ipfs/{cid}"
        logging.debug(f"Preparing to download file from IPFS Gateway URL: {ipfs_gateway_url} to filename '{output_filename}' using curl.")

        # Starting the download process
        logging.info("Downloading file using curl...")
        process = subprocess.run(["curl", "-L", ipfs_gateway_url, "-o", output_filename], capture_output=True, text=True)
        
        # Checking for errors in the download process
        if process.returncode != 0:
            logging.error(f"Curl failed with return code {process.returncode}. Stderr: {process.stderr}")
            raise subprocess.CalledProcessError(process.returncode, process.args, output=process.stdout, stderr=process.stderr)
        
        logging.info(f"File downloaded successfully to '{output_filename}'.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Curl encountered an error: {e.stderr}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during the download process: {e}")
        raise

def extract_metadata(file_path, num_ids):
    try:
        logging.debug(f"Opening file {file_path} to extract metadata")
        with open(file_path, 'r') as file:
            file_contents = file.read().strip()
            if not file_contents:
                logging.error(f"File {file_path} is empty")
                return []
            data = json.loads(file_contents)
            logging.debug(f"File {file_path} opened and JSON data loaded successfully")
            extracted_data = data[:num_ids]
            logging.info(f"Extracted metadata for {num_ids} items from {file_path}")
            return extracted_data
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding error in {file_path}: {e}")
        return []
    except Exception as e:
        logging.error(f"Error while extracting metadata from {file_path}: {e}")
        raise


def main():
    logging.info("Starting main execution")
    
    # CID of the file to download
    cid = "bafybeic7mblcdr4re3ul6m3j4wchub6dxnfezpkyg5g6owydtidh7jml2q"
    output_filename = "metadata_file.json"  # Suitable filename for the JSON data

    # Download the file
    logging.debug("Starting download process")
    download_file_with_curl(cid, output_filename)

    # Extract and print metadata for 10 unique IDs
    logging.debug("Starting metadata extraction")
    num_ids = 10
    metadata = extract_metadata(output_filename, num_ids)
    logging.debug("Printing extracted metadata")
    for item in metadata:
        logging.debug(f"Processing item: {item}")
        print(f"Unique ID: {item['unique_id']}, Title: {item['title']}, URL: {item['url']}, Text: {item['text']}")

    logging.info("Main execution completed")

if __name__ == "__main__":
    main()
