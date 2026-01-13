from sentence_transformers import SentenceTransformer
import torch

class EmbeddingModel:
    """
    Wrapper for Hugging Face Sentence Transformers.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def get_embedding(self, text: str):
        """Generates a dense vector embedding for the input text."""
        return self.model.encode(text, convert_to_tensor=True)
    
    def get_embeddings(self, text_list: list[str]):
        """Generates embeddings for a list of texts."""
        return self.model.encode(text_list, convert_to_tensor=True)