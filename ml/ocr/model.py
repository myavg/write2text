from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from pathlib import Path
import torch
from typing import List, Tuple, Dict


class OCRModel:
    """OCR модель на основе TrOCR с поддержкой пакетной обработки (Batching)."""

    def __init__(self, model_name: str = "raxtemur/trocr-base-ru", local_model_path: str = None):
        """
        Инициализация модели OCR.
        """
        self.model_name = model_name

        # Определение устройства (GPU или CPU или MPS)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"Устройство для вычислений: {self.device.upper()}")

        # Определяем путь к локальной модели
        if local_model_path is None:
            BASE_DIR = Path(__file__).parent.parent.parent
            local_model_path = BASE_DIR / "models" / model_name.replace("/", "_")

        self.local_model_path = Path(local_model_path)

        print(f"Загрузка модели {model_name}...")
        try:
            if self.local_model_path.exists() and any(self.local_model_path.iterdir()):
                print(f"Используется локальная модель из {self.local_model_path}")
                model_path = str(self.local_model_path)
                local_files_only = True
            else:
                print(f"Модель не найдена локально, загружаем из HuggingFace...")
                model_path = model_name
                local_files_only = False
                self.local_model_path.mkdir(parents=True, exist_ok=True)

            self.processor = TrOCRProcessor.from_pretrained(
                model_path,
                local_files_only=local_files_only
            )
            self.model = VisionEncoderDecoderModel.from_pretrained(
                model_path,
                local_files_only=local_files_only
            )

            # Переносим модель на GPU, если есть
            self.model.to(self.device)

            if not local_files_only:
                print(f"Сохранение модели в {self.local_model_path}...")
                self.processor.save_pretrained(str(self.local_model_path))
                self.model.save_pretrained(str(self.local_model_path))

            self.model.eval()
            print("Модель успешно загружена!")
        except Exception as e:
            print(f"Ошибка загрузки модели: {str(e)}")
            raise e

    def predict_batch(self, image_paths: List[str]) -> List[str]:
        """
        Распознает текст для списка путей к изображениям за один проход (батч).
        """
        images = []
        valid_indices = []  # Чтобы отслеживать, какие картинки загрузились успешно

        # Загружаем картинки в память
        for idx, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Ошибка открытия файла {path}: {e}")
                # Если файл битый, мы его пропускаем, на выходе будет пустая строка для него

        if not images:
            return [""] * len(image_paths)

        # Подготовка батча процессором
        # processor сам сделает ресайз и нормализацию для списка картинок
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)

        # Генерация текста для всего батча
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)

        # Декодирование
        batch_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Собираем полный список результатов, учитывая ошибки загрузки
        final_results = [""] * len(image_paths)
        for i, original_idx in enumerate(valid_indices):
            final_results[original_idx] = batch_texts[i]

        return final_results

    def predict_directory(self, frames_dir: Path, batch_size: int = 16) -> str:
        """
        Проходит по всем word frames в директории и склеивает распознанный текст.
        Использует BATCH PROCESSING вместо потоков.

        Args:
            frames_dir: Путь к директории с frames
            batch_size: Размер пачки.
        """
        frames_dir = Path(frames_dir)
        if not frames_dir.exists():
            raise ValueError(f"Директория {frames_dir} не существует")

        # 1. Сбор всех файлов и метаданных
        # Список кортежей: (row_num, word_num, absolute_path)
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

        total_files = len(word_frames)
        print(f"Найдено {total_files} изображений. Начало обработки батчами по {batch_size}...")

        results_dict = {}  # {(row_num, word_num): text}

        # 2. Обработка батчами
        # Идем по списку с шагом batch_size
        for i in range(0, total_files, batch_size):
            # Вырезаем кусок списка (батч)
            current_batch_meta = word_frames[i: i + batch_size]

            # Достаем только пути для модели
            batch_paths = [meta[2] for meta in current_batch_meta]

            # Предсказываем
            print(f"Обработка батча {i // batch_size + 1}/{(total_files + batch_size - 1) // batch_size}...")
            batch_predictions = self.predict_batch(batch_paths)

            # Сохраняем результаты
            for j, text in enumerate(batch_predictions):
                row_n, word_n, _ = current_batch_meta[j]
                results_dict[(row_n, word_n)] = text.strip()

        print("Обработка завершена.")

        # 3. Склейка текста (логика осталась прежней)
        result_lines = []
        current_row = None
        current_line_words = []

        # word_frames уже отсортирован по row и word при сборе
        for row_num, word_num, _ in word_frames:
            if current_row is not None and row_num != current_row:
                if current_line_words:
                    result_lines.append(" ".join(current_line_words))
                current_line_words = []

            word_text = results_dict.get((row_num, word_num), "")
            if word_text:
                current_line_words.append(word_text)

            current_row = row_num

        if current_line_words:
            result_lines.append(" ".join(current_line_words))

        return "\n".join(result_lines)

    # Оставим метод для одиночного файла для совместимости
    def predict(self, image_path: str) -> str:
        return self.predict_batch([image_path])[0]
