import streamlit as st
from PIL import Image

from ocr.image_preprocessing import ImagePreprocessor
from ocr.ocr_engine import TrOCREngine
from nlp.embeddings import EmbeddingModel
from nlp.evaluator import AutoGrader
from nlp.preprocessing import TextPreprocessor
from nlp.relevance import RelevanceFilter


st.set_page_config(
    page_title="Handwritten Exam Evaluator",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_ocr_engine():
    return TrOCREngine()


@st.cache_resource
def load_nlp_components():
    embed_model = EmbeddingModel()
    text_prep = TextPreprocessor()
    relevance_filter = RelevanceFilter(embed_model)
    grader = AutoGrader(embed_model)
    return text_prep, relevance_filter, grader


st.sidebar.title("Evaluation Config")
max_marks = st.sidebar.slider("Maximum Marks", 5, 100, 10)
relevance_thresh = st.sidebar.slider(
    "Relevance Threshold",
    0.0,
    1.0,
    0.45,
    help="Similarity score required for a sentence to be considered part of the answer.",
)

st.sidebar.info(
    """
    **System Status:**
    - OCR Model: TrOCR Base Handwritten
    - NLP Model: all-MiniLM-L6-v2
    - Device: Local CPU/GPU
    """
)
st.sidebar.caption("Models load on the first evaluation and stay cached for later runs.")

st.title("Handwritten Exam Evaluator")
st.markdown("### AI-Assisted Descriptive Answer Grading")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Correct (Model) Answer")
    model_answer_text = st.text_area(
        "Enter the reference answer here:",
        height=200,
        placeholder="Type the ideal answer key here...",
    )

with col2:
    st.subheader("2. Student Answer (Upload)")
    uploaded_file = st.file_uploader(
        "Upload Handwritten Answer (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
    )

    student_img = None
    if uploaded_file is not None:
        student_img = Image.open(uploaded_file)
        st.image(student_img, caption="Uploaded Image", use_container_width=True)


if st.button("Evaluate Answer", type="primary"):
    if not model_answer_text:
        st.error("Please provide a Model Answer.")
    elif not student_img:
        st.error("Please upload a Student Answer image.")
    else:
        try:
            with st.spinner("Loading OCR and NLP models... This can take a minute on the first run."):
                ocr_engine = load_ocr_engine()
                text_prep, relevance_filter, grader = load_nlp_components()

            with st.status("Processing Image...", expanded=True) as status:
                st.write("Preprocessing image (denoising/thresholding)...")
                processed_img = ImagePreprocessor.preprocess_image(student_img)

                st.write("Running TrOCR (Handwritten Text Recognition)...")
                extracted_text = ocr_engine.extract_text(processed_img)
                status.update(label="OCR Complete", state="complete", expanded=False)

            with st.spinner("Analyzing Semantics..."):
                clean_student_text = text_prep.clean_text(extracted_text)
                clean_model_text = text_prep.clean_text(model_answer_text)
                student_sentences = text_prep.get_sentences(clean_student_text)

                relevant_sents, irrelevant_sents = relevance_filter.filter_content(
                    student_sentences,
                    clean_model_text,
                    threshold=relevance_thresh,
                )

                relevant_block = " ".join(relevant_sents)
                result = grader.grade_answer(relevant_block, clean_model_text, max_marks)

            st.divider()
            st.header("Evaluation Results")

            with st.expander("Show Raw OCR Output", expanded=False):
                st.text(extracted_text)

            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.success("Relevant Content (Used for Grading)")
                if relevant_sents:
                    for sentence in relevant_sents:
                        st.markdown(f"- {sentence}")
                else:
                    st.warning("No relevant content found.")

            with res_col2:
                st.error("Irrelevant / Unrelated Content")
                if irrelevant_sents:
                    for sentence in irrelevant_sents:
                        st.markdown(f"- {sentence}")
                else:
                    st.info("No irrelevant content detected.")

            st.divider()
            score_col1, score_col2, score_col3 = st.columns(3)

            with score_col1:
                st.metric(
                    label="Semantic Similarity",
                    value=f"{result['similarity_score']:.2f}",
                )

            with score_col2:
                st.metric(
                    label="Marks Awarded",
                    value=f"{result['marks_awarded']} / {max_marks}",
                )

            with score_col3:
                st.info(f"**Verdict:** {result['reason']}")

        except Exception as exc:
            st.error(f"An error occurred during evaluation: {exc}")
            st.exception(exc)


st.markdown("---")
st.caption(
    "**Academic Disclaimer:** This system provides AI-assisted evaluation based on semantic similarity. "
    "Handwritten OCR may introduce noise. Human verification of the extracted text and final grade is recommended."
)
