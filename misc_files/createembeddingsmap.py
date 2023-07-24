from nomic import atlas
import numpy as np

# Let's assume 'your_embeddings' is a numpy array with your embeddings
# It should be 2-dimensional array, with each row being an individual 384-dimensional vector
your_embeddings = np.load('embeddings.npy')

# Check the shape of your embeddings to make sure it's (n, 384)
print(f"Shape of embeddings: {your_embeddings.shape}")

# Create a project with the embeddings
project = atlas.map_embeddings(embeddings=your_embeddings)
