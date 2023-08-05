import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List all the files created or used by the FAISS processing code
files_to_preserve = [
    "/home/alexcarrabre/temp_shards",
    "/home/alexcarrabre/join_index.py",
]

# Delete everything on disk except for the preserved files
logger.info("Removing all other files on disk...")
for root, dirs, files in os.walk('/'):
    for file in files:
        full_path = os.path.join(root, file)
        if not any(full_path.startswith(preserve) for preserve in files_to_preserve):
            try:
                os.remove(full_path)
                logger.info(f"Deleted: {full_path}")
            except Exception as e:
                logger.warning(f"Could not delete: {full_path}. Error: {e}")

logger.info("Finished cleaning disk.")
