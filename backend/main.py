from fastapi import FastAPI, UploadFile, File

app = FastAPI()

def dummy_ocr(_: bytes) -> str:
    return "Recognized text (stub)"

@app.get("/")
def root():
    return {"status": "ok", "message": "Backend is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    text = dummy_ocr(image_bytes)

    return {"recognized_text": text}
