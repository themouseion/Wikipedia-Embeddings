import os
import json
import nltk
from google.cloud import storage

# If you haven't downloaded the NLTK sentence tokenizer yet, you need to do so
# Uncomment the next line if it's the case
# nltk.download('punkt')

bucket_name = 'mouseion-embeddings'
client = storage.Client()
bucket = client.get_bucket(bucket_name)

sentence_count = 0

blobs = bucket.list_blobs(prefix='output_directory_newcode')
for blob in blobs:
    if blob.name.endswith('.json'):
        text = blob.download_as_text()
        for line in text.splitlines():
            json_line = json.loads(line)
            sentences = nltk.sent_tokenize(json_line['text'])
            sentence_count += len(sentences)
            if sentence_count % 1000000 == 0:
                print(f'Processed {sentence_count} sentences')

print(f'Total sentences: {sentence_count}')
