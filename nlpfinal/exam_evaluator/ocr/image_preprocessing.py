import cv2
import numpy as np
from PIL import Image

class ImagePreprocessor:
    """
    Handles standard image preprocessing tasks to improve OCR accuracy
    for handwritten text.
    """

    @staticmethod
    def preprocess_image(pil_image: Image.Image) -> Image.Image:
        """
        Convert PIL image to CV2, apply grayscale, denoising, and thresholding,
        then convert back to PIL.
        """
        # Convert PIL to OpenCV format (RGB -> BGR)
        image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # 1. Grayscale
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

        # 2. Noise Reduction (Gaussian Blur)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. Adaptive Thresholding (Binarization)
        # Useful for varying lighting conditions in photos of paper
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )

        # 4. Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, h=10, searchWindowSize=21, templateWindowSize=7)

        # Convert back to PIL Image (Grayscale)
        processed_pil = Image.fromarray(denoised)
        
        return processed_pil