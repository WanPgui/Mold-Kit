from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Model placeholder (will use fallback prediction)
MODEL = None
print("Using fallback prediction (no TensorFlow)")

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
        
        # Fallback prediction based on image darkness/color analysis
        avg_brightness = float(arr.mean())
        green_channel = float(arr[:, :, 1].mean())
        
        # Simple heuristic: darker images with less green = more likely mold
        if avg_brightness < 0.4 and green_channel < 0.3:
            label = "mold"
            confidence = 0.75
        elif avg_brightness < 0.6:
            label = "mold"
            confidence = 0.60
        else:
            label = "clean"
            confidence = 0.80
        
        return jsonify({
            "prediction": label,
            "confidence": f"{confidence*100:.1f}%",
            "filename": secure_filename(file.filename)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)