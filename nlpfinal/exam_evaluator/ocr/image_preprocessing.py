from PIL import Image, ImageFilter, ImageOps


class ImagePreprocessor:
    """
    Handles lightweight image preprocessing tasks to improve OCR accuracy
    without requiring OpenCV on hosted deployments.
    """

    @staticmethod
    def preprocess_image(pil_image: Image.Image) -> Image.Image:
        """
        Convert the image to grayscale, improve contrast, reduce small noise,
        and apply a simple threshold using Pillow only.
        """
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        gray = ImageOps.grayscale(pil_image)
        contrasted = ImageOps.autocontrast(gray)
        denoised = contrasted.filter(ImageFilter.MedianFilter(size=3))

        threshold = 160
        binary = denoised.point(lambda px: 255 if px > threshold else 0, mode="L")
        return binary
