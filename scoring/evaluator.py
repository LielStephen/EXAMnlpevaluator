from __future__ import annotations

import re

from textblob import TextBlob

from config import Settings
from nlp.keyword_matcher import extract_reference_keywords, keyword_coverage
from nlp.similarity import lexical_similarity, question_alignment, semantic_similarity


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _token_count(text: str) -> int:
    return len(re.findall(r"\b[a-zA-Z][a-zA-Z0-9-]*\b", text))


def grammar_score(text: str) -> float:
    text = _normalize_text(text)
    if not text:
        return 0.0

    try:
        sample_words = text.split()[:40]
        sample_text = " ".join(sample_words)
        corrected = str(TextBlob(sample_text).correct())
        original_tokens = sample_text.split()
        corrected_tokens = corrected.split()
        token_count = max(len(original_tokens), 1)
        changed = sum(
            1
            for left, right in zip(original_tokens, corrected_tokens)
            if left.lower() != right.lower()
        )
        score = max(0.0, 1.0 - (changed / token_count))
        return round(score, 4)
    except Exception:
        sentence_count = max(len(re.split(r"[.!?]+", text)), 1)
        avg_sentence_length = _token_count(text) / sentence_count
        return round(1.0 if 4 <= avg_sentence_length <= 30 else 0.8, 4)


def length_penalty(student_text: str, reference_text: str) -> tuple[float, str]:
    student_len = _token_count(student_text)
    reference_len = max(_token_count(reference_text), 1)
    ratio = student_len / reference_len

    if ratio < Settings.answer_length_floor:
        return max(ratio / Settings.answer_length_floor, 0.5), "Answer is much shorter than the reference."
    if ratio > Settings.answer_length_ceiling:
        return max(Settings.answer_length_ceiling / ratio, 0.7), "Answer is much longer than the reference and may contain irrelevant text."
    return 1.0, "Answer length is within the expected range."


def evaluate_answer(
    *,
    student_text: str,
    reference_text: str,
    question_text: str = "",
    max_marks: int = Settings.default_max_marks,
) -> dict:
    student_text = _normalize_text(student_text)
    reference_text = _normalize_text(reference_text)
    question_text = _normalize_text(question_text)

    if not student_text:
        raise ValueError("Student answer is empty.")
    if not reference_text:
        raise ValueError("Reference answer is empty.")

    semantic_score = semantic_similarity(student_text, reference_text)
    lexical_score = lexical_similarity(student_text, reference_text)
    alignment_score = question_alignment(student_text, question_text)
    grammar = grammar_score(student_text)

    reference_keywords = extract_reference_keywords(reference_text, question_text)
    keyword_result = keyword_coverage(student_text, reference_keywords)
    penalty_factor, penalty_reason = length_penalty(student_text, reference_text)

    weighted_score = (
        (0.60 * semantic_score)
        + (0.20 * keyword_result["score"])
        + (0.10 * lexical_score)
        + (0.05 * grammar)
        + (0.05 * alignment_score)
    )
    final_score = max(0.0, min(weighted_score * penalty_factor, 1.0))
    final_marks = round(final_score * max_marks, 2)

    return {
        "semantic_similarity": round(semantic_score * 100, 2),
        "keyword_match_score": round(keyword_result["score"] * 100, 2),
        "lexical_similarity": round(lexical_score * 100, 2),
        "question_alignment": round(alignment_score * 100, 2),
        "grammar_score": round(grammar * 100, 2),
        "length_penalty_factor": round(penalty_factor, 4),
        "length_penalty_reason": penalty_reason,
        "final_marks": final_marks,
        "max_marks": max_marks,
        "matched_keywords": keyword_result["matched_keywords"],
        "missing_keywords": keyword_result["missing_keywords"],
        "reference_keywords": reference_keywords,
        "student_text": student_text,
        "reference_text": reference_text,
        "question_text": question_text,
    }
