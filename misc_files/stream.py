import subprocess
import json
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def stream_metadata(cid):
    try:
        logging.debug(f"Starting to stream file with CID: {cid}")
        result = subprocess.run(["ipfs", "cat", cid], capture_output=True, text=True, check=True)
        logging.debug(f"Raw streamed data: {result.stdout[:100]}...")  # Log first 100 characters of data for preview
        metadata = json.loads(result.stdout)
        logging.info(f"Successfully streamed and parsed file with CID: {cid}")
        return metadata
    except subprocess.CalledProcessError as e:
        logging.error(f"Error streaming file with CID {cid}: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON data from CID {cid}: {e}")
        return None

def main():
    logging.info("Main execution started")
    
    # CID of the file to stream
    cid = "bafybeic7mblcdr4re3ul6m3j4wchub6dxnfezpkyg5g6owydtidh7jml2q"
    logging.debug(f"Using CID: {cid} for streaming")

    # Stream and process the metadata
    metadata = stream_metadata(cid)
    if metadata:
        logging.info(f"Processing metadata for the first 10 items")
        for item in metadata[:10]:
            logging.debug(f"Processing item with Unique ID: {item['unique_id']}")
            print(f"Unique ID: {item['unique_id']}, Title: {item['title']}, URL: {item['url']}, Text: {item['text']}")
    else:
        logging.warning("No metadata to process")

    logging.info("Main execution completed")

if __name__ == "__main__":
    main()
