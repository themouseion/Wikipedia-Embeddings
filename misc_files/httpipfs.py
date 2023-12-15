import requests
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def download_file_from_ipfs(node_url, output_filename):
    try:
        logging.info(f"Downloading file from your local IPFS node at: {node_url}")

        # Send a GET request to your local IPFS node with a 30-minute timeout
        response = requests.get(node_url, stream=True, timeout=1800)

        # Check if the request was successful
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192  # 8 KiloBytes
            total_data = 0
            with open(output_filename, 'wb') as file:
                for chunk in response.iter_content(chunk_size=block_size):
                    total_data += len(chunk)
                    file.write(chunk)
                    done = int(50 * total_data / total_size)
                    print("\r[%s%s]" % ('=' * done, ' ' * (50-done)), end='')
                print()
            logging.info(f"File downloaded successfully and saved as '{output_filename}'")
        else:
            logging.error(f"Failed to download file from your local IPFS node. HTTP status code: {response.status_code}")
            response.raise_for_status()

    except requests.RequestException as e:
        logging.error(f"An error occurred while downloading the file: {e}")
        raise

def generate_random_ids(num_ids, lower_bound, upper_bound):
    logging.info(f"Generating {num_ids} random unique IDs between {lower_bound} and {upper_bound}")
    return random.sample(range(lower_bound, upper_bound + 1), num_ids)

def main():
    # URL of the file to download from your local IPFS node
    local_ipfs_url = "https://ipfs.io/ipfs/bafybeic7mblcdr4re3ul6m3j4wchub6dxnfezpkyg5g6owydtidh7jml2q"
    
    # Local filename to save the downloaded file
    output_filename = "downloaded_file"  # You can change this to your preferred file name and extension

    # Download the file from your local IPFS node
    download_file_from_ipfs(local_ipfs_url, output_filename)

    # Generate and print 10 random unique IDs
    num_ids = 10
    lower_bound = 1
    upper_bound = 627000000
    random_ids = generate_random_ids(num_ids, lower_bound, upper_bound)
    logging.info(f"Random unique IDs: {random_ids}")

if __name__ == "__main__":
    main()
