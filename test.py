from ml.preprocessing.text_segmenter import TextSegmenter
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(BASE_DIR, "storage", "text.jpg")
output_dir = os.path.join(BASE_DIR, "storage", "frames")

segmenter = TextSegmenter()
segmenter.process_and_save(image_path, output_dir)
