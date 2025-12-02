from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from PIL import Image
from pathlib import Path
import torch
from typing import List, Tuple, Dict, Optional, Union


class OCRModel:
    def __init__(
        self, 
        model_name: str = "raxtemur/trocr-base-ru", 
        local_model_path: str = None,
        lora_path: Optional[str] = None,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None
    ):
        self.model_name = model_name
        self.use_lora = use_lora
        self.lora_path = lora_path

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

            if self.use_lora:
                if lora_path and Path(lora_path).exists():
                    self.model = PeftModel.from_pretrained(self.model, lora_path)
                else:
                    self.model = self._apply_lora(
                        self.model, 
                        lora_r, 
                        lora_alpha, 
                        lora_dropout, 
                        lora_target_modules
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

    def _apply_lora(
        self, 
        model: VisionEncoderDecoderModel,
        r: int = 8,
        alpha: int = 32,
        dropout: float = 0.1,
        target_modules: Optional[List[str]] = None
    ) -> VisionEncoderDecoderModel:
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
        
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
            modules_to_save=None,
        )
        
        model = get_peft_model(model, lora_config)
        
        return model

    def save_lora_adapters(self, save_path: str) -> None:
        if not self.use_lora:
            raise ValueError("LoRA is not enabled. Cannot save adapters.")
        
        if hasattr(self.model, 'save_pretrained'):
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            if isinstance(self.model, PeftModel):
                self.model.save_pretrained(str(save_path))
            else:
                self.model.save_pretrained(str(save_path))
        else:
            raise ValueError("Model does not support saving LoRA adapters.")

    def enable_training_mode(self) -> None:
        if self.use_lora:
            self.model.train()
        else:
            raise ValueError("LoRA is not enabled. Cannot enable training mode.")

    def disable_training_mode(self) -> None:
        self.model.eval()

    def get_trainable_parameters(self) -> int:
        if self.use_lora:
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            return trainable_params
        else:
            return sum(p.numel() for p in self.model.parameters())

    # Let's leave the method for a single file for compatibility
    def predict(self, image_path: str) -> str:
        return self.predict_batch([image_path])[0]