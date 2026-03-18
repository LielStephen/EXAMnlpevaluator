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
        import torch.nn.functional as F

        score = F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
        return score.item()
