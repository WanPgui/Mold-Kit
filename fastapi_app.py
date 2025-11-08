from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

app = FastAPI(title="Mold Detection API")

# Load model
MODEL_PATH = "models/mold_model_final.keras"
MODEL = None

if os.path.exists(MODEL_PATH):
    try:
        MODEL = load_model(MODEL_PATH, compile=False)
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")

def predict_image(image_bytes):
    """Predict mold from image bytes"""
    try:
        img = Image.open(image_bytes).convert("RGB")
        img = img.resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        
        if MODEL:
            pred = MODEL.predict(arr, verbose=0).flatten()
            score = float(pred[0])
            label = "mold" if score >= 0.5 else "clean"
            confidence = score if label == "mold" else (1 - score)
            return label, confidence
        return "unknown", 0.0
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

@app.get("/")
async def root():
    return {"message": "Mold Detection API"}

@app.post("/predict")
async def predict_mold(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    contents = await file.read()
    label, confidence = predict_image(contents)
    
    return JSONResponse({
        "filename": file.filename,
        "prediction": label,
        "confidence": f"{confidence*100:.2f}%"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)