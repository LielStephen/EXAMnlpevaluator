from sentence_transformers import util
import torch

class SimilarityCalculator:
    """
    Computes similarity scores between embeddings.
    """
    
    @staticmethod
    def cosine_similarity(embedding1, embedding2) -> float:
        """
        Compute cosine similarity between two tensors.
        Returns a float between -1 and 1 (usually 0 to 1 for text).
        """
        # util.cos_sim returns a tensor, we extract the item
        return util.cos_sim(embedding1, embedding2).item()