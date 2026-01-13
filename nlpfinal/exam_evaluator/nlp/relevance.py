from .similarity import SimilarityCalculator

class RelevanceFilter:
    """
    Filters student answer sentences based on semantic similarity 
    to the model answer.
    """

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.sim_calc = SimilarityCalculator()

    def filter_content(self, student_sentences: list[str], model_answer: str, threshold: float = 0.50):
        """
        Classifies sentences into Relevant and Irrelevant.
        
        Strategy:
        Compare each student sentence embedding against the WHOLE model answer embedding.
        If the conceptual overlap is high enough, it's relevant context.
        """
        relevant_sentences = []
        irrelevant_sentences = []

        model_emb = self.embedding_model.get_embedding(model_answer)

        for sent in student_sentences:
            sent_emb = self.embedding_model.get_embedding(sent)
            score = self.sim_calc.cosine_similarity(sent_emb, model_emb)

            if score >= threshold:
                relevant_sentences.append(sent)
            else:
                irrelevant_sentences.append(sent)

        return relevant_sentences, irrelevant_sentences