import os
import base64
import tempfile
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ml.preprocessing import TextSegmenter
from ml.ocr import OCRModel

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OCR model (lazy loading)
_ocr_model = None

def get_ocr_model():
    """Get or initialize OCR model."""
    global _ocr_model
    if _ocr_model is None:
        try:
            _ocr_model = OCRModel()
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to load OCR model: {str(e)}"
            )
    return _ocr_model

@app.get("/")
def root():
    return {"status": "ok", "message": "Backend is running"}

@app.post("/segment")
async def segment_text(file: UploadFile = File(...)):
    """Process uploaded image and return all word frames as base64 images.
    Word frames are also saved into the frames directory, replacing any previous data."""
    try:
        # Save uploaded file temporarily (preserve original extension)
        file_ext = Path(file.filename).suffix if file.filename else ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Create temporary output directory for processing
        with tempfile.TemporaryDirectory() as output_dir:
            # Process image with TextSegmenter
            segmenter = TextSegmenter()
            segmenter.process_and_save(tmp_path, output_dir)
            
            # Prepare frames directory (clear previous session)
            BASE_DIR = Path(__file__).parent.parent
            frames_dir = BASE_DIR / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            for item in frames_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            
            # Collect all word frames and copy to frames directory
            word_frames = []
            row_dirs = sorted(Path(output_dir).glob("row_*"), key=lambda x: int(x.name.split("_")[1]))
            
            for row_dir in row_dirs:
                word_files = sorted(row_dir.glob("word_*.png"), key=lambda x: int(x.stem.split("_")[1]))
                for word_file in word_files:
                    # Read image for response
                    with open(word_file, "rb") as f:
                        img_bytes = f.read()
                        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                    
                    # Copy to frames directory
                    storage_row_dir = frames_dir / row_dir.name
                    storage_row_dir.mkdir(exist_ok=True)
                    storage_path = storage_row_dir / word_file.name
                    shutil.copy2(word_file, storage_path)
                    
                    word_frames.append({
                        "row": int(row_dir.name.split("_")[1]),
                        "word": int(word_file.stem.split("_")[1]),
                        "image": f"data:image/png;base64,{img_base64}",
                        "saved_path": str(storage_path.relative_to(BASE_DIR))
                    })
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Perform OCR on all word frames
        recognized_text = ""
        try:
            ocr_model = get_ocr_model()
            recognized_text = ocr_model.predict_directory(frames_dir)
        except HTTPException:
            # Re-raise HTTP exceptions (e.g., model loading errors)
            raise
        except Exception as e:
            # Log error but don't fail the request for OCR prediction errors
            print(f"OCR error: {str(e)}")
            recognized_text = f"[OCR Error: {str(e)}]"
        
        return JSONResponse({
            "status": "success",
            "word_count": len(word_frames),
            "word_frames": word_frames,
            "recognized_text": recognized_text,
            "storage_path": str(frames_dir.relative_to(BASE_DIR))
        })
    
    except Exception as e:
        # Clean up temp file if it exists
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
