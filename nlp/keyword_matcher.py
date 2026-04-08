from __future__ import annotations

import re

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

from config import Settings


stemmer = PorterStemmer()
GENERIC_TERMS = {
    "answer",
    "answers",
    "question",
    "questions",
    "word",
    "words",
    "using",
    "used",
    "use",
    "explain",
    "define",
    "write",
    "mention",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z][a-zA-Z0-9-]*\b", text.lower())


def _normalize_term(term: str) -> str:
    return " ".join(stemmer.stem(token) for token in _tokenize(term))


def extract_reference_keywords(reference_text: str, question_text: str = "") -> list[str]:
    corpus = [reference_text.strip()]
    if question_text.strip():
        corpus.append(question_text.strip())

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=Settings.max_keywords * 3,
    )
    matrix = vectorizer.fit_transform(corpus)
    scores = matrix[0].toarray()[0]
    features = vectorizer.get_feature_names_out()
    ranked = sorted(
        ((feature, score) for feature, score in zip(features, scores) if score > 0),
        key=lambda item: item[1],
        reverse=True,
    )

    keywords: list[str] = []
    seen: set[str] = set()
    for term, _ in ranked:
        tokens = _tokenize(term)
        if not tokens:
            continue
        if all(token in GENERIC_TERMS for token in tokens):
            continue
        if len(tokens) == 1 and len(tokens[0]) < 4:
            continue
        normalized = _normalize_term(term)
        if not normalized or normalized in seen:
            continue
        if any(normalized in existing or existing in normalized for existing in seen):
            continue
        seen.add(normalized)
        keywords.append(term)
        if len(keywords) >= Settings.max_keywords:
            break
    return keywords


def keyword_coverage(student_text: str, reference_keywords: list[str]) -> dict:
    student_lower = student_text.lower()
    student_tokens = set(_tokenize(student_text))
    student_stem_text = _normalize_term(student_text)
    matched: list[str] = []
    missing: list[str] = []

    for keyword in reference_keywords:
        keyword_tokens = _tokenize(keyword)
        keyword_stem = _normalize_term(keyword)
        direct_match = keyword.lower() in student_lower
        token_match = all(token in student_tokens for token in keyword_tokens)
        stem_match = keyword_stem and keyword_stem in student_stem_text

        if direct_match or token_match or stem_match:
            matched.append(keyword)
        else:
            missing.append(keyword)

    coverage = len(matched) / len(reference_keywords) if reference_keywords else 0.0
    return {
        "score": round(coverage, 4),
        "matched_keywords": matched,
        "missing_keywords": missing,
    }
