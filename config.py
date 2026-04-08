from __future__ import annotations


class Settings:
    page_title = "NLP-Based Exam Evaluator (Lightweight Version)"
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    easyocr_languages = ["en"]
    use_gpu_for_ocr = False
    default_max_marks = 10
    max_image_size_mb = 5
    max_keywords = 12
    answer_length_floor = 0.45
    answer_length_ceiling = 1.85
