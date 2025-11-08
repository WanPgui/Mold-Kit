from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Load model
MODEL_PATH = "models/mold_model_final.keras"
MODEL = None

if os.path.exists(MODEL_PATH):
    try:
        MODEL = load_model(MODEL_PATH, compile=False)
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/')
def health():
    return {"status": "healthy", "message": "Mold Detection API"}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400
    
    try:
        # Process image
        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        
        if MODEL:
            pred = MODEL.predict(arr, verbose=0).flatten()
            score = float(pred[0])
            label = "mold" if score >= 0.5 else "clean"
            confidence = score if label == "mold" else (1 - score)
        else:
            label, confidence = "unknown", 0.0
        
        return jsonify({
            "prediction": label,
            "confidence": f"{confidence*100:.1f}%",
            "filename": secure_filename(file.filename)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)