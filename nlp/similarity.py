from __future__ import annotations

from sentence_transformers import util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp.embeddings import get_embedding_model


def semantic_similarity(student_text: str, reference_text: str) -> float:
    if not student_text.strip() or not reference_text.strip():
        return 0.0

    model = get_embedding_model()
    embeddings = model.encode([student_text, reference_text], convert_to_tensor=True)
    return float(util.cos_sim(embeddings[0], embeddings[1]).item())


def lexical_similarity(student_text: str, reference_text: str) -> float:
    if not student_text.strip() or not reference_text.strip():
        return 0.0

    tfidf_matrix = TfidfVectorizer(stop_words="english").fit_transform(
        [student_text, reference_text]
    )
    return float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])


def question_alignment(student_text: str, question_text: str) -> float:
    if not question_text.strip():
        return 1.0
    return semantic_similarity(student_text, question_text)
