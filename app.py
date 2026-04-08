from __future__ import annotations

import streamlit as st

from config import Settings
from feedback.generator import generate_feedback
from ocr.extractor import OCRExtractor
from scoring.evaluator import evaluate_answer
from utils.preprocessing import (
    highlight_keywords,
    load_image_from_bytes,
    normalize_text,
    validate_image_size,
)


st.set_page_config(page_title=Settings.page_title, layout="wide")
st.title(Settings.page_title)


@st.cache_resource
def load_ocr_engine() -> OCRExtractor:
    return OCRExtractor()


def ensure_session_defaults() -> None:
    st.session_state.setdefault("student_answer_text", "")


ensure_session_defaults()

st.sidebar.header("Configuration")
max_marks = st.sidebar.number_input("Maximum Marks", min_value=1, max_value=100, value=10)
st.sidebar.caption(
    "CPU-first configuration: EasyOCR for extraction and MiniLM embeddings for scoring."
)

st.subheader("Section 1: Upload Answer Sheet Image")
uploaded_file = st.file_uploader(
    "Upload handwritten or printed answer image",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded answer sheet", use_container_width=True)
    if st.button("Extract OCR Text"):
        try:
            validate_image_size(uploaded_file.size)
            image = load_image_from_bytes(uploaded_file.getvalue())
            with st.spinner("Extracting text with EasyOCR..."):
                extracted_text = load_ocr_engine().extract_text(image)
            st.session_state["student_answer_text"] = normalize_text(extracted_text)
        except Exception as exc:
            st.error(f"OCR failed: {exc}")

st.subheader("Section 2: Display OCR Extracted Text (editable textbox)")
st.text_area(
    "Student Answer (OCR output, editable)",
    key="student_answer_text",
    height=220,
    placeholder="OCR extracted text will appear here. You can correct it manually before evaluation.",
)

st.subheader("Section 3: Paste Reference Answer")
reference_answer = st.text_area(
    "Paste the model/reference answer copied from Google, textbook, or internet",
    height=220,
    placeholder="Paste the reference answer here.",
)

st.subheader("Section 4: Optional Question Text Input")
question_text = st.text_area(
    "Question Text (optional but recommended)",
    height=100,
    placeholder="Paste the question here for better accuracy.",
)

st.subheader("Section 5: Evaluate Button")
evaluate_clicked = st.button("Evaluate", type="primary")

if evaluate_clicked:
    try:
        student_answer = normalize_text(st.session_state.get("student_answer_text", ""))
        result = evaluate_answer(
            student_text=student_answer,
            reference_text=reference_answer,
            question_text=question_text,
            max_marks=int(max_marks),
        )
        feedback_items = generate_feedback(result)

        st.subheader("Section 6: Results Display")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Semantic Similarity", f"{result['semantic_similarity']}%")
        metric_cols[1].metric("Keyword Coverage", f"{result['keyword_match_score']}%")
        metric_cols[2].metric("Final Marks", f"{result['final_marks']} / {result['max_marks']}")
        metric_cols[3].metric("Question Alignment", f"{result['question_alignment']}%")

        st.progress(min(result["final_marks"] / max(result["max_marks"], 1), 1.0))

        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.markdown("**Missing Keyword Suggestions**")
            if result["missing_keywords"]:
                st.write(", ".join(result["missing_keywords"]))
            else:
                st.write("No major missing keywords detected.")

            st.markdown("**Short Improvement Feedback**")
            for item in feedback_items:
                st.markdown(f"- {item}")

        with result_col2:
            st.markdown("**Scoring Details**")
            st.write(
                {
                    "lexical_similarity_percent": result["lexical_similarity"],
                    "grammar_score_percent": result["grammar_score"],
                    "length_penalty_factor": result["length_penalty_factor"],
                    "length_penalty_reason": result["length_penalty_reason"],
                }
            )

        st.markdown("**Keyword Highlighting**", help="Green = matched, red = missing.")
        highlighted_reference = highlight_keywords(
            result["reference_text"],
            result["matched_keywords"],
            "#c6f6d5",
        )
        highlighted_reference = highlight_keywords(
            highlighted_reference,
            result["missing_keywords"],
            "#fed7d7",
        )
        highlighted_student = highlight_keywords(
            result["student_text"],
            result["matched_keywords"],
            "#c6f6d5",
        )

        highlight_col1, highlight_col2 = st.columns(2)
        with highlight_col1:
            st.markdown("**Reference Answer with Keyword Highlights**")
            st.markdown(highlighted_reference, unsafe_allow_html=True)
        with highlight_col2:
            st.markdown("**Student Answer with Matched Keywords**")
            st.markdown(highlighted_student, unsafe_allow_html=True)

    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")
