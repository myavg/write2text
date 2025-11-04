import os
import base64
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ml.preprocessing import TextSegmenter

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "message": "Backend is running"}

@app.post("/segment")
async def segment_text(file: UploadFile = File(...)):
    """Process uploaded image and return all word frames as base64 images.
    Word frames are also saved to storage/frames directory."""
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
            
            # Prepare storage directory
            BASE_DIR = Path(__file__).parent.parent
            storage_dir = BASE_DIR / "storage" / "frames"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = storage_dir / timestamp
            session_dir.mkdir(parents=True, exist_ok=True)
            
            # Collect all word frames and copy to storage
            word_frames = []
            row_dirs = sorted(Path(output_dir).glob("row_*"), key=lambda x: int(x.name.split("_")[1]))
            
            for row_dir in row_dirs:
                word_files = sorted(row_dir.glob("word_*.png"), key=lambda x: int(x.stem.split("_")[1]))
                for word_file in word_files:
                    # Read image for response
                    with open(word_file, "rb") as f:
                        img_bytes = f.read()
                        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                    
                    # Copy to storage
                    storage_row_dir = session_dir / row_dir.name
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
        
        return JSONResponse({
            "status": "success",
            "word_count": len(word_frames),
            "word_frames": word_frames,
            "storage_path": str(session_dir.relative_to(BASE_DIR))
        })
    
    except Exception as e:
        # Clean up temp file if it exists
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
