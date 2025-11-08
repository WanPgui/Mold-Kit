from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import base64
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Model placeholder (will use fallback prediction)
MODEL = None
print("Using fallback prediction (no dependencies)")

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
        # Simple file-based prediction (no image processing)
        file_size = len(file.read())
        file.seek(0)  # Reset file pointer
        
        # Basic heuristic based on file size and name
        filename_lower = file.filename.lower() if file.filename else ""
        
        # Simple prediction logic
        if file_size > 500000:  # Large files more likely to be detailed mold images
            label = "mold"
            confidence = 0.70
        elif "mold" in filename_lower or "fungus" in filename_lower:
            label = "mold"
            confidence = 0.85
        elif "clean" in filename_lower or "normal" in filename_lower:
            label = "clean"
            confidence = 0.85
        else:
            # Random-ish prediction based on file size
            label = "mold" if (file_size % 3) == 0 else "clean"
            confidence = 0.60
        
        return jsonify({
            "prediction": label,
            "confidence": f"{confidence*100:.1f}%",
            "filename": secure_filename(file.filename or "unknown.jpg"),
            "note": "Demo mode - using basic file analysis"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)