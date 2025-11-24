"""
Script for pre-downloading the OCR model to a local directory.
Run this script once to download the model locally.
"""
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from pathlib import Path

model_name = "raxtemur/trocr-base-ru"
BASE_DIR = Path(__file__).parent
local_model_path = BASE_DIR / "models" / model_name.replace("/", "_")

print(f"Downloading model {model_name}...")
print(f"Saving to {local_model_path}...")

try:
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    
    local_model_path.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(str(local_model_path))
    model.save_pretrained(str(local_model_path))
    
    print("Model successfully downloaded and saved!")
    print(f"Model location: {local_model_path}")
except Exception as e:
    print(f"Error downloading model: {str(e)}")
    raise e

