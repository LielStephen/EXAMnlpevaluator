import re

class TextPreprocessor:
    """
    Handles text cleaning, sentence splitting, and normalization using spaCy.
    """
    
    def __init__(self, model: str = "en_core_web_sm"):
        import spacy
        try:
            self.nlp = spacy.load(model)
        except OSError:
            # Hosted deployments are more reliable if we avoid requiring an
            # extra downloaded spaCy model package.
            self.nlp = spacy.blank("en")
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")

    def clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        return text.strip()

    def get_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def lemmatize_remove_stopwords(self, text: str) -> str:
        """
        Returns a string of joined lemmas, removing stopwords and punctuation.
        Useful for raw comparison, though Semantic Models often prefer full context.
        """
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return " ".join(tokens)
