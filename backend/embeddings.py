import numpy as np

# LIGHTWEIGHT MODE: Mock Embeddings
# This generates consistent random vectors so the database runs without AI libraries.

def get_embedding(text):
    """
    Generates a fake vector (list of numbers) based on the text.
    Used to bypass 'sentence-transformers' installation errors.
    """
    if not text:
        return []
    
    # 1. Create a consistent 'seed' from the text
    # (So "Hello" always produces the same vector)
    seed = abs(hash(text)) % (2**32)
    
    # 2. Use Numpy to generate random numbers based on that seed
    rng = np.random.RandomState(seed)
    
    # 3. Return a standard 384-dimension vector
    return rng.rand(384).tolist()