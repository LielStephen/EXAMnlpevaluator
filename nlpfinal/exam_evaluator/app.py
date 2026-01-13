import streamlit as st
from PIL import Image
import os

# Import custom modules
from ocr.ocr_engine import TrOCREngine
from ocr.image_preprocessing import ImagePreprocessor
from nlp.preprocessing import TextPreprocessor
from nlp.embeddings import EmbeddingModel
from nlp.relevance import RelevanceFilter
from nlp.evaluator import AutoGrader

# Page Config
st.set_page_config(
    page_title="Handwritten Exam Evaluator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CACHING RESOURCES (Singleton Pattern) ---
# We use st.cache_resource to load models only once.

@st.cache_resource
def load_ocr_engine():
    return TrOCREngine()

@st.cache_resource
def load_nlp_components():
    embed_model = EmbeddingModel()
    text_prep = TextPreprocessor()
    return embed_model, text_prep

# Load models immediately
with st.spinner("Loading AI Models (OCR & NLP)... This may take a minute on first run."):
    ocr_engine = load_ocr_engine()
    embed_model, text_prep = load_nlp_components()
    relevance_filter = RelevanceFilter(embed_model)
    grader = AutoGrader(embed_model)

# --- SIDEBAR ---
st.sidebar.title("Evaluation Config")
max_marks = st.sidebar.slider("Maximum Marks", 5, 100, 10)
relevance_thresh = st.sidebar.slider("Relevance Threshold", 0.0, 1.0, 0.45, help="Similarity score required for a sentence to be considered part of the answer.")

st.sidebar.info(
    """
    **System Status:**
    - OCR Model: TrOCR Base Handwritten
    - NLP Model: all-MiniLM-L6-v2
    - Device: Local (CPU/GPU)
    """
)

# --- MAIN UI ---
st.title("📝 Handwritten Exam Evaluator")
st.markdown("### AI-Assisted Descriptive Answer Grading")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Correct (Model) Answer")
    model_answer_text = st.text_area(
        "Enter the reference answer here:", 
        height=200, 
        placeholder="Type the ideal answer key here..."
    )

with col2:
    st.subheader("2. Student Answer (Upload)")
    uploaded_file = st.file_uploader("Upload Handwritten Answer (PNG/JPG)", type=['png', 'jpg', 'jpeg'])
    
    student_img = None
    if uploaded_file is not None:
        student_img = Image.open(uploaded_file)
        st.image(student_img, caption="Uploaded Image", use_column_width=True)

# --- EVALUATION LOGIC ---

if st.button("Evaluate Answer", type="primary"):
    if not model_answer_text:
        st.error("Please provide a Model Answer.")
    elif not student_img:
        st.error("Please upload a Student Answer image.")
    else:
        try:
            # 1. Preprocessing Image
            with st.status("Processing Image...", expanded=True) as status:
                st.write("Preprocessing image (denoising/thresholding)...")
                processed_img = ImagePreprocessor.preprocess_image(student_img)
                
                st.write("Running TrOCR (Handwritten Text Recognition)...")
                extracted_text = ocr_engine.extract_text(processed_img)
                status.update(label="OCR Complete", state="complete", expanded=False)

            # 2. NLP Pipeline
            with st.spinner("Analyzing Semantics..."):
                # Clean and Split
                clean_student_text = text_prep.clean_text(extracted_text)
                clean_model_text = text_prep.clean_text(model_answer_text)
                
                student_sentences = text_prep.get_sentences(clean_student_text)

                # Relevance Filtering
                relevant_sents, irrelevant_sents = relevance_filter.filter_content(
                    student_sentences, clean_model_text, threshold=relevance_thresh
                )
                
                # Reconstruct texts
                relevant_block = " ".join(relevant_sents)
                
                # Grading
                result = grader.grade_answer(relevant_block, clean_model_text, max_marks)

            # --- DISPLAY RESULTS ---
            st.divider()
            st.header("Evaluation Results")

            # OCR Output
            with st.expander("Show Raw OCR Output", expanded=False):
                st.text(extracted_text)

            # Content Analysis Columns
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.success("✅ Relevant Content (Used for Grading)")
                if relevant_sents:
                    for s in relevant_sents:
                        st.markdown(f"- {s}")
                else:
                    st.warning("No relevant content found.")

            with res_col2:
                st.error("⚠️ Irrelevant / Unrelated Content")
                if irrelevant_sents:
                    for s in irrelevant_sents:
                        st.markdown(f"- {s}")
                else:
                    st.info("No irrelevant content detected.")

            st.divider()

            # Scoring Dashboard
            score_col1, score_col2, score_col3 = st.columns(3)
            
            with score_col1:
                st.metric(label="Semantic Similarity", value=f"{result['similarity_score']:.2f}")
            
            with score_col2:
                st.metric(label="Marks Awarded", value=f"{result['marks_awarded']} / {max_marks}")

            with score_col3:
                st.info(f"**Verdict:** {result['reason']}")

        except Exception as e:
            st.error(f"An error occurred during evaluation: {e}")
            st.exception(e)

# --- DISCLAIMER ---
st.markdown("---")
st.caption(
    "**Academic Disclaimer:** This system provides AI-assisted evaluation based on semantic similarity. "
    "Handwritten OCR may introduce noise. Human verification of the extracted text and final grade is recommended."
)