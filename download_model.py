"""
Скрипт для предварительной загрузки OCR модели в локальную директорию.
Запустите этот скрипт один раз, чтобы скачать модель локально.
"""
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from pathlib import Path

model_name = "raxtemur/trocr-base-ru"
BASE_DIR = Path(__file__).parent
local_model_path = BASE_DIR / "models" / model_name.replace("/", "_")

print(f"Загрузка модели {model_name}...")
print(f"Сохранение в {local_model_path}...")

try:
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    
    local_model_path.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(str(local_model_path))
    model.save_pretrained(str(local_model_path))
    
    print("✓ Модель успешно загружена и сохранена!")
    print(f"✓ Модель находится в: {local_model_path}")
except Exception as e:
    print(f"✗ Ошибка загрузки модели: {str(e)}")
    raise e

