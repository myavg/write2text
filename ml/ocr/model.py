from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from pathlib import Path
import torch
from typing import List, Tuple, Dict


class OCRModel:
    """OCR model based on TrOCR with batch processing support."""
    def __init__(self, model_name: str = "raxtemur/trocr-base-ru", local_model_path: str = None):
        """
        Initializing the OCR model.
        """
        self.model_name = model_name

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"Device for computing: {self.device.upper()}")

        # Определяем путь к локальной модели
        if local_model_path is None:
            BASE_DIR = Path(__file__).parent.parent.parent
            local_model_path = BASE_DIR / "models" / model_name.replace("/", "_")

        self.local_model_path = Path(local_model_path)

        print(f"Loading the model {model_name}...")
        try:
            if self.local_model_path.exists() and any(self.local_model_path.iterdir()):
                print(f"A local model is used from {self.local_model_path}")
                model_path = str(self.local_model_path)
                local_files_only = True
            else:
                print(f"Model not found locally, loading from HuggingFace...")
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

            # Transfer the model to the GPU, if available
            self.model.to(self.device)

            if not local_files_only:
                print(f"Saving the model in {self.local_model_path}...")
                self.processor.save_pretrained(str(self.local_model_path))
                self.model.save_pretrained(str(self.local_model_path))

            self.model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e

    def predict_batch(self, image_paths: List[str]) -> List[str]:
        """
        Recognizes text for a list of image paths in a single pass (batch).
        """
        images = []
        valid_indices = []  # To track which images have loaded successfully

        # Loading images into memory
        for idx, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Ошибка открытия файла {path}: {e}")
                # If the file is broken, we skip it, the output will be an empty line for it.

        if not images:
            return [""] * len(image_paths)

        # Preparing the batch with the processor
        # The processor will automatically resize and normalize the image list
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)

        # Generating text for the entire batch
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)

        # Decoding
        batch_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        # We collect a complete list of results, taking into account loading errors
        final_results = [""] * len(image_paths)
        for i, original_idx in enumerate(valid_indices):
            final_results[original_idx] = batch_texts[i]

        return final_results

    def predict_directory(self, frames_dir: Path, batch_size: int = 16) -> str:
        """
        Iterates through all word frames in a directory and concatenates the recognized text.
        Uses BATCH PROCESSING instead of streams.

        Args:
        frames_dir: Path to the frames directory
        batch_size: Batch size.
        """
        frames_dir = Path(frames_dir)
        if not frames_dir.exists():
            raise ValueError(f"Directory {frames_dir} does not exist")
        
        # 1. Collect all files and metadata
        # List of tuples: (row_num, word_num, absolute_path)
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
        print(f"{total_files} images found. Starting processing in batches of {batch_size}...")

        results_dict = {}  # {(row_num, word_num): text}

        # 2. Batch processing
        # We go through the list in batch_size increments
        for i in range(0, total_files, batch_size):
            # Cut out a piece of the list (batch)
            current_batch_meta = word_frames[i: i + batch_size]

            # We only get the paths for the model
            batch_paths = [meta[2] for meta in current_batch_meta]

            # Predict
            print(f"Batch processing {i // batch_size + 1}/{(total_files + batch_size - 1) // batch_size}...")
            batch_predictions = self.predict_batch(batch_paths)

            # Saving the results
            for j, text in enumerate(batch_predictions):
                row_n, word_n, _ = current_batch_meta[j]
                results_dict[(row_n, word_n)] = text.strip()

        print("Processing complete.")

        #3. Text gluing (the logic remains the same)        
        result_lines = []
        current_row = None
        current_line_words = []

        # word_frames is already sorted by row and word when collected
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

    # Let's leave the method for a single file for compatibility
    def predict(self, image_path: str) -> str:
        return self.predict_batch([image_path])[0]