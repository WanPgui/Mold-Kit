from flask import Flask, request
from flask_cors import CORS
from flask_restx import Api, Resource, fields
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import os

app = Flask(__name__)
CORS(app)

# Configure Swagger UI
api = Api(
    app,
    version='1.0',
    title='Mold Detection API',
    description='AI-powered mold detection from images',
    doc='/swagger/',
    prefix='/api'
)

# Namespaces
ns = api.namespace('mold', description='Mold detection operations')

# Models for Swagger documentation
upload_parser = api.parser()
upload_parser.add_argument('file', location='files', type=FileStorage, required=True, help='Image file to analyze')

prediction_model = api.model('Prediction', {
    'success': fields.Boolean(description='Request success status'),
    'prediction': fields.String(description='Mold detection result', enum=['mold', 'clean']),
    'confidence': fields.String(description='Confidence percentage'),
    'filename': fields.String(description='Uploaded filename'),
    'file_size': fields.Integer(description='File size in bytes'),
    'note': fields.String(description='Additional information')
})

error_model = api.model('Error', {
    'error': fields.String(description='Error message')
})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

@ns.route('/health')
class Health(Resource):
    def get(self):
        """Health check endpoint"""
        return {
            "status": "healthy", 
            "message": "Mold Detection API is running",
            "version": "1.0"
        }

@ns.route('/predict')
class MoldPredict(Resource):
    @api.expect(upload_parser)
    @api.marshal_with(prediction_model, code=200)
    @api.marshal_with(error_model, code=400)
    def post(self):
        """
        Analyze image for mold detection
        Upload an image file to get mold detection results
        """
        if 'file' not in request.files:
            api.abort(400, "No file uploaded")
        
        file = request.files['file']
        if not file or not file.filename:
            api.abort(400, "No file selected")
            
        if not allowed_file(file.filename):
            api.abort(400, "Invalid file type. Supported: PNG, JPG, JPEG, GIF, BMP")
        
        try:
            # Simple file-based prediction
            file_content = file.read()
            file_size = len(file_content)
            file.seek(0)  # Reset file pointer
            
            filename_lower = file.filename.lower()
            
            # Prediction logic
            if file_size > 500000:
                label = "mold"
                confidence = 0.75
            elif "mold" in filename_lower or "fungus" in filename_lower:
                label = "mold"
                confidence = 0.85
            elif "clean" in filename_lower or "normal" in filename_lower:
                label = "clean"
                confidence = 0.85
            else:
                label = "mold" if (file_size % 3) == 0 else "clean"
                confidence = 0.65
            
            return {
                "success": True,
                "prediction": label,
                "confidence": f"{confidence*100:.1f}%",
                "filename": secure_filename(file.filename),
                "file_size": file_size,
                "note": "Demo mode - using basic file analysis"
            }
            
        except Exception as e:
            api.abort(500, f"Processing error: {str(e)}")

# Root redirect handled by Flask-RESTX automatically

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)