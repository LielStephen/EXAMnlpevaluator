from PIL import Image

class TrOCREngine:
    """
    Wrapper for Microsoft's TrOCR (Transformer OCR) model specialized 
    for handwritten text.
    """
    
    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten"):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading OCR Model on: {self.device}...")
        
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        print("OCR Model loaded successfully.")

    def extract_text(self, image: Image.Image) -> str:
        """
        Extracts text from a preprocessed PIL image.
        """
        try:
            # Ensure image is in RGB for the processor
            if image.mode != "RGB":
                image = image.convert("RGB")

            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)

            # Generate text
            generated_ids = self.model.generate(pixel_values, max_new_tokens=256)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return generated_text
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""