from fastapi import FastAPI, File, UploadFile, HTTPException, status, Form
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import os
from typing import Optional
import uvicorn
from PIL import Image
import numpy as np
import tensorflow as tf
from io import BytesIO
import requests

# Initialize FastAPI app
app = FastAPI(
    title="Mold Detection API",
    description="AI-powered mold detection service",
    version="1.0.0"
)

# Configuration
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp'}
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB
MODEL_PATH = "models/mold_model_final.keras"
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "d6159ecc80513ec0218d61d4285e2d52")

# Load model - REQUIRED
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
print(f"✓ Model loaded from {MODEL_PATH}")

# Response models
class PredictionResponse(BaseModel):
    filename: str
    prediction: str
    confidence: float
    confidence_display: str
    risk_status: str
    weather: str
    risk_score: int
    message: str

class HealthResponse(BaseModel):
    status: str
    message: str

# Utility functions
def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)

def predict_mold(file_content: bytes, filename: str) -> tuple:
    """
    Real ML model prediction only
    Returns: (label, confidence_float, confidence_display)
    """
    # Real ML prediction
    img = Image.open(BytesIO(file_content)).convert("RGB")
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    
    pred = MODEL.predict(arr, verbose=0).flatten()
    score = float(np.clip(pred[0], 0.0, 1.0))
    
    label = "mold" if score >= 0.5 else "clean"
    confidence = score if label == "mold" else (1 - score)
    confidence_display = f"{confidence * 100:.2f}%"
    
    return label, confidence, confidence_display

# Routes
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API information"""
    return HealthResponse(
        status="success",
        message="Mold Detection API is running. Visit /docs for API documentation."
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="API is operational"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_mold_endpoint(
    file: UploadFile = File(...),
    location: str = Form(""),
    ventilation: str = Form("moderate"),
    leak: str = Form("no"),
    health: str = Form("no")
):
    """
    Upload an image and get comprehensive mold risk assessment
    """
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not allowed. Supported: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Read file content
    try:
        content = await file.read()
        
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File too large. Maximum size is 8MB"
            )
        
        # Make prediction
        label, confidence, confidence_display = predict_mold(content, file.filename)
        
        # Environmental risk assessment
        weather_data = "No location provided"
        humidity = None
        
        if location.strip():
            try:
                res = requests.get(
                    "https://api.openweathermap.org/data/2.5/weather",
                    params={"q": location, "appid": OPENWEATHER_API_KEY, "units": "metric"},
                    timeout=4
                )
                data = res.json()
                if res.ok and "main" in data:
                    temperature = data["main"].get("temp")
                    humidity = data["main"].get("humidity")
                    weather_data = f"{temperature}°C, {humidity}% humidity"
                else:
                    weather_data = "Invalid location"
            except Exception:
                weather_data = "Weather fetch error"
        
        # Risk score calculation
        ventilation_risk = {"poor": 2, "moderate": 1, "good": 0}
        risk_score = ventilation_risk.get(ventilation.lower(), 1)
        
        if leak.lower() == "yes":
            risk_score += 2
        if humidity is not None:
            risk_score += 2 if humidity > 70 else (1 if humidity > 50 else 0)
        if health.lower() == "yes":
            risk_score += 1
        
        # Final status
        if label == "mold" and risk_score >= 3:
            risk_status = "high"
        elif label == "mold":
            risk_status = "moderate"
        elif risk_score <= 2:
            risk_status = "safe"
        else:
            risk_status = "high"
        
        return PredictionResponse(
            filename=file.filename,
            prediction=label,
            confidence=confidence,
            confidence_display=confidence_display,
            risk_status=risk_status,
            weather=weather_data,
            risk_score=risk_score,
            message=f"Risk assessment complete. Mold: {label} ({confidence_display}), Risk: {risk_status}"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )

# Error handlers
@app.exception_handler(413)
async def request_entity_too_large_handler(request, exc):
    return JSONResponse(
        status_code=413,
        content={"detail": "File too large. Maximum size is 8MB"}
    )

# Remove the if __name__ block for production deployment