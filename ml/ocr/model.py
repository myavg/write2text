from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from pathlib import Path
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple


class OCRModel:
    """OCR модель на основе TrOCR для распознавания русского текста."""
    
    def __init__(self, model_name: str = "raxtemur/trocr-base-ru", local_model_path: str = None):
        """
        Инициализация модели OCR.
        
        Args:
            model_name: Название модели из HuggingFace
            local_model_path: Локальный путь к модели (если None, используется models/ в корне проекта)
        """
        self.model_name = model_name
        
        # Определяем путь к локальной модели
        if local_model_path is None:
            # Используем директорию models в корне проекта
            BASE_DIR = Path(__file__).parent.parent.parent
            local_model_path = BASE_DIR / "models" / model_name.replace("/", "_")
        
        self.local_model_path = Path(local_model_path)
        
        print(f"Загрузка модели {model_name}...")
        try:
            # Проверяем, есть ли модель локально
            if self.local_model_path.exists() and any(self.local_model_path.iterdir()):
                print(f"Используется локальная модель из {self.local_model_path}")
                model_path = str(self.local_model_path)
                local_files_only = True
            else:
                print(f"Модель не найдена локально, загружаем из HuggingFace...")
                model_path = model_name
                local_files_only = False
                # Создаем директорию для сохранения
                self.local_model_path.mkdir(parents=True, exist_ok=True)
            
            # Загружаем processor и model
            self.processor = TrOCRProcessor.from_pretrained(
                model_path, 
                local_files_only=local_files_only
            )
            self.model = VisionEncoderDecoderModel.from_pretrained(
                model_path,
                local_files_only=local_files_only
            )
            
            # Если модель загружена из HuggingFace, сохраняем локально
            if not local_files_only:
                print(f"Сохранение модели в {self.local_model_path}...")
                self.processor.save_pretrained(str(self.local_model_path))
                self.model.save_pretrained(str(self.local_model_path))
                print("Модель сохранена локально!")
            
            # Переводим модель в режим оценки (не обучения)
            self.model.eval()
            print("Модель успешно загружена!")
        except Exception as e:
            print(f"Ошибка загрузки модели: {str(e)}")
            raise e
    
    def predict(self, image_path: str) -> str:
        """
        Распознает текст на изображении.
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Распознанный текст
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл {image_path} не найден")
        
        # Подготовка изображения
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        
        # Генерация текста
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
    
    def predict_directory(self, frames_dir: Path, max_workers: int = 4) -> str:
        """
        Проходит по всем word frames в директории и склеивает распознанный текст.
        Использует параллельную обработку для ускорения.
        Структура директории: frames/row_0/word_0.png, row_0/word_1.png, row_1/word_0.png, ...
        
        Args:
            frames_dir: Путь к директории с frames
            max_workers: Количество параллельных потоков (по умолчанию 4)
            
        Returns:
            Склеенный текст со всех frames
        """
        frames_dir = Path(frames_dir)
        if not frames_dir.exists():
            raise ValueError(f"Директория {frames_dir} не существует")
        
        # Собираем все word frames, отсортированные по row и word
        word_frames = []
        row_dirs = sorted(frames_dir.glob("row_*"), key=lambda x: int(x.name.split("_")[1]))
        
        for row_dir in row_dirs:
            word_files = sorted(row_dir.glob("word_*.png"), key=lambda x: int(x.stem.split("_")[1]))
            for word_file in word_files:
                row_num = int(row_dir.name.split("_")[1])
                word_num = int(word_file.stem.split("_")[1])
                word_frames.append((row_num, word_num, str(word_file)))
        
        if not word_frames:
            return ""
        
        print(f"Обработка {len(word_frames)} изображений в {max_workers} потоках...")
        
        # Распознаем текст параллельно
        results_dict = {}  # {(row_num, word_num): text}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Запускаем все задачи
            future_to_frame = {
                executor.submit(self.predict, word_file): (row_num, word_num)
                for row_num, word_num, word_file in word_frames
            }
            
            # Собираем результаты по мере выполнения
            completed = 0
            for future in as_completed(future_to_frame):
                row_num, word_num = future_to_frame[future]
                try:
                    word_text = future.result()
                    results_dict[(row_num, word_num)] = word_text.strip()
                    completed += 1
                    if completed % 10 == 0:
                        print(f"Обработано: {completed}/{len(word_frames)}")
                except Exception as e:
                    print(f"Ошибка распознавания (row={row_num}, word={word_num}): {str(e)}")
                    results_dict[(row_num, word_num)] = ""
        
        print(f"Обработка завершена: {completed}/{len(word_frames)}")
        
        # Собираем результаты в правильном порядке
        result_lines = []
        current_row = None
        current_line_words = []
        
        for row_num, word_num, _ in word_frames:
            if current_row is not None and row_num != current_row:
                if current_line_words:
                    result_lines.append(" ".join(current_line_words))
                current_line_words = []
            
            word_text = results_dict.get((row_num, word_num), "")
            if word_text:  # Добавляем только непустые слова
                current_line_words.append(word_text)
            
            current_row = row_num
        
        # Добавляем последнюю строку
        if current_line_words:
            result_lines.append(" ".join(current_line_words))
        
        # Склеиваем все строки через перенос строки
        result_text = "\n".join(result_lines)
        
        return result_text

