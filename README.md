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
- Python version: `3.10` via `runtime.txt`
- Dependencies file: `requirements.txt`

Notes:

- Models are loaded lazily on the first evaluation instead of during app startup.
- The app uses lightweight built-in sentence splitting, so no extra spaCy model download is required.

## Render Deployment

This repo now includes Render configuration:

- `render.yaml` defines a Python web service
- `.python-version` pins Python to `3.10`
- the service starts with `streamlit_app.py`

Manual setup in Render:

1. Create a new `Web Service`
2. Connect `LielStephen/EXAMnlpevaluator`
3. Use branch `main`
4. Render should detect the root `render.yaml`, or you can use:
   `Build Command`: `pip install -r requirements.txt`
   `Start Command`: `python -m streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`
