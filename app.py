from fastapi import FastAPI, File, UploadFile, HTTPException, status
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import os
from typing import Optional
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Mold Detection API",
    description="AI-powered mold detection service",
    version="1.0.0"
)

# Configuration
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp'}
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB

# Response models
class PredictionResponse(BaseModel):
    filename: str
    prediction: str
    confidence: float
    confidence_display: str
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
    Simple mold prediction based on file characteristics
    Returns: (label, confidence_float, confidence_display)
    """
    try:
        # Simple heuristic based on file size and name
        file_size = len(file_content)
        
        # Basic prediction logic (replace with actual ML model)
        if 'mold' in filename.lower() or 'fungus' in filename.lower():
            confidence = 0.85
            label = "mold"
        elif 'clean' in filename.lower() or 'normal' in filename.lower():
            confidence = 0.90
            label = "clean"
        else:
            # Use file size as a simple heuristic
            confidence = min(0.75, file_size / (1024 * 1024) * 0.3 + 0.5)
            label = "mold" if confidence > 0.6 else "clean"
        
        confidence_display = f"{confidence * 100:.2f}%"
        return label, confidence, confidence_display
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return "unknown", 0.0, "0.00%"

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
async def predict_mold_endpoint(file: UploadFile = File(...)):
    """
    Upload an image and get mold detection prediction
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
        
        return PredictionResponse(
            filename=file.filename,
            prediction=label,
            confidence=confidence,
            confidence_display=confidence_display,
            message=f"Prediction completed successfully. Result: {label} ({confidence_display})"
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)