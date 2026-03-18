# Exam_evaluatornlp

An AI-assisted descriptive exam evaluation framework combining handwritten OCR, NLP preprocessing, sentence embeddings, and semantic relevance analysis to score student answers while explicitly identifying irrelevant content.

## Local Run

```powershell
cd "Exam_evaluatornlp"
python -m streamlit run streamlit_app.py
```

## Streamlit Cloud Deployment

Use these settings in Streamlit Cloud:

- Repository root: `Exam_evaluatornlp`
- Main file path: `streamlit_app.py`
- Python version: `3.11` via `runtime.txt`
- Dependencies file: `requirements.txt`

Notes:

- Models are loaded lazily on the first evaluation instead of during app startup.
- The app no longer requires downloading the `en_core_web_sm` spaCy package at deploy time.
