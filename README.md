# Wikipedia-Index

Data is download from the [wikimedia foundation](https://dumps.wikimedia.org/backup-index.html).

## Getting started

make sure to change the file paths within this repository to your respective paths. 


### Scripts for embedding

- to update requirements.txt

pip freeze > requirements.txt

### Preprocessing

The dataset is massive, first run the file 'splitdata.py'. This also separates the data into URL, Text & Title.

### Embedding
 
Next, run 'paraphrase.py'. This runs the data through the paraphrase embedding model. It's manually separated by files, so that you can run on multiple devices, otherwise it will take multiple days.

### there are various scripts within the "cleaning" file that help with file count etc. 