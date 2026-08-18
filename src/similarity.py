import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(embeddings: list[list[float]]) -> np.ndarray:
    return cosine_similarity(np.asarray(embeddings))
