from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests
from typing import Optional

app = FastAPI(title="Mold Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def get_weather(location: str):
    """Get weather data from OpenWeatherMap API for Kenya"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "No weather API key"
    
    # Default to Nairobi if no location provided
    if not location:
        location = "Nairobi,KE"
    elif "," not in location:
        location = f"{location},KE"  # Add Kenya country code
    
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            city = data["name"]
            return f"{city}: {temp}°C, {humidity}% humidity"
        return "Weather unavailable"
    except:
        return "Weather unavailable"

def calculate_risk(confidence: float, weather_info: str):
    """Calculate mold risk based on prediction and weather"""
    if confidence >= 0.8:
        return "high", 3
    elif confidence >= 0.6:
        return "moderate", 2
    elif confidence >= 0.4:
        return "low", 1
    return "minimal", 0

@app.post("/predict")
async def predict_mold(file: UploadFile = File(...), location: Optional[str] = Form(None)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    contents = await file.read()
    label, confidence = predict_image(contents)
    
    weather_info = get_weather(location)
    risk_status, risk_score = calculate_risk(confidence, weather_info)
    
    return JSONResponse({
        "filename": file.filename,
        "prediction": label,
        "confidence": confidence,
        "confidence_display": f"{confidence*100:.2f}%",
        "risk_status": risk_status,
        "weather": weather_info,
        "risk_score": risk_score,
        "message": f"Risk assessment complete. Mold: {label} ({confidence*100:.2f}%), Risk: {risk_status}"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)