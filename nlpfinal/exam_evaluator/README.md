# Handwritten Exam Evaluation System

A local Streamlit application that uses Computer Vision (TrOCR) and NLP (Sentence Transformers) to grade handwritten student answers by comparing them to a model answer.

## Features
- **OCR:** Converts handwritten images to text using Microsoft's TrOCR.
- **NLP:** Filters content into "Relevant" vs "Irrelevant" based on semantic context.
- **Grading:** Assigns marks based on conceptual similarity, not just keyword matching.
- **Privacy:** Runs 100% locally on your machine.

## Prerequisites
- Python 3.9+
- Windows (Recommended for this setup)
- ~2GB of Disk Space for Models

## Installation

1. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate