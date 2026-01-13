from .similarity import SimilarityCalculator

class AutoGrader:
    """
    Assigns marks based on the similarity of the RELEVANT portions 
    of the student answer to the model answer.
    """

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.sim_calc = SimilarityCalculator()

    def grade_answer(self, relevant_text: str, model_answer: str, max_marks: int):
        """
        Computes final score and returns detailed metrics.
        """
        if not relevant_text.strip():
            return {
                "score": 0.0,
                "marks": 0,
                "reason": "No relevant content detected in the answer."
            }

        rel_emb = self.embedding_model.get_embedding(relevant_text)
        mod_emb = self.embedding_model.get_embedding(model_answer)
        
        similarity_score = self.sim_calc.cosine_similarity(rel_emb, mod_emb)

        # Scoring Rubric
        # 1.0 - 0.85 -> 100%
        # 0.84 - 0.70 -> 70%
        # 0.69 - 0.50 -> 50%
        # < 0.50 -> 20% (for effort/keywords found in noise)

        percentage = 0.0
        reason = ""

        if similarity_score >= 0.85:
            percentage = 1.0
            reason = "Excellent match. Concepts align perfectly."
        elif similarity_score >= 0.70:
            percentage = 0.70
            reason = "Good answer. Major concepts present but lacks detail."
        elif similarity_score >= 0.50:
            percentage = 0.50
            reason = "Average. Some relevant concepts, but significant gaps."
        else:
            percentage = 0.20
            reason = "Weak answer. Concepts loosely related or insufficient."

        marks_awarded = round(percentage * max_marks, 1)

        return {
            "similarity_score": round(similarity_score, 4),
            "percentage": percentage,
            "marks_awarded": marks_awarded,
            "reason": reason
        }