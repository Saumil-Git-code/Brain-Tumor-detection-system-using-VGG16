from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

try:
    model = load_model('model.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMAGE_SIZE = 128
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict(file_path):
    """Predict tumor class from MRI image"""
    if model is None:
        return "Model not loaded", 0.0
    
    try:
        # Load and preprocess image (matching notebook)
        img = load_img(file_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get prediction
        predictions = model.predict(img_array, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        
        return class_labels[class_idx], confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Prediction error", 0.0

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            return render_template('index.html', error="No file selected.")
        
        if not allowed_file(file.filename):
            return render_template('index.html', error="Invalid file format. Use JPG, PNG, or BMP.")
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        result, confidence = predict(file_path)
        return render_template('index.html', result=result, confidence=f"{confidence*100:.2f}", file_path=filename)
    
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, secure_filename(filename))

if __name__ == '__main__':
    app.run(debug=True)
