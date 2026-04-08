from __future__ import annotations


def generate_feedback(result: dict) -> list[str]:
    feedback: list[str] = []

    if result["semantic_similarity"] >= 80:
        feedback.append("The answer is conceptually close to the reference answer.")
    elif result["semantic_similarity"] >= 60:
        feedback.append("The answer captures part of the core meaning, but not all of it.")
    else:
        feedback.append("The answer is missing major concepts from the reference answer.")

    if result["keyword_match_score"] < 50 and result["missing_keywords"]:
        feedback.append(
            "Important concepts are missing: "
            + ", ".join(result["missing_keywords"][:5])
            + "."
        )

    if result["question_text"] and result["question_alignment"] < 60:
        feedback.append("The answer is not strongly aligned with the actual question asked.")

    if result["length_penalty_factor"] < 1:
        feedback.append(result["length_penalty_reason"])

    if result["grammar_score"] < 70:
        feedback.append("Improve clarity and sentence quality before submission.")

    if not feedback:
        feedback.append("The answer is balanced across meaning, keywords, and length.")

    return feedback
