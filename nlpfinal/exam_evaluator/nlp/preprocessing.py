import re

class TextPreprocessor:
    """
    Handles lightweight text cleaning and sentence splitting.
    """

    def __init__(self):
        pass

    def clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        return text.strip()

    def get_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        parts = re.split(r'(?<=[.!?])\s+|\n+', text)
        return [part.strip() for part in parts if part.strip()]

    def lemmatize_remove_stopwords(self, text: str) -> str:
        """
        Lightweight fallback normalization without external NLP models.
        """
        tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "in", "is", "it", "of", "on", "or", "that", "the", "to", "was",
            "were", "will", "with",
        }
        return " ".join(token for token in tokens if token not in stopwords)
