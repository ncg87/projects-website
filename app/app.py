from flask import Flask, request, jsonify, render_template, Response

# For logging/debugging
import sys

# Translation functions
from utils.translator.translator_utils import predict, tokenizer
# Image Captioning functions
from utils.captioning.image_captioning_utils import predict_caption
# Face detector functions
from utils.face_detection.face_detector_utils import generate_predictions, start_camera, stop_camera

# Create instance of a Flask application
app = Flask(__name__)

####-- Rendering Pages --####

# Render Home Page
@app.get('/')
def home_page():
    return render_template('home.html')

# Render Translation Page
@app.get('/translation')
def predict_page():
    return render_template('translate.html')

# Render Image Captioning Page
@app.get('/image_captioning')
def captioning():
    return render_template('image_captioning.html')

# Render Face Detection Page
@app.get('/face_detector')
def face_detection():
    return render_template('face_detector.html')


####-- Prediction Endpoints --####

# Predict translation for specified text and target language
@app.post('/translate')
def prediction():
    # Check if the request is a POST request, so we don't get error
    if request.method not in ['POST']:
        return jsonify({'error': 'POST requests only'})
    try:
        # Get the data from the POST request
        data = request.get_json()
        # Check if data is empty
        if data is None:
            return jsonify({'error': 'No data found'})
        # Get the text and target language from the data
        text = data.get('text')
        target_lang = data.get('target_language')
        # Get the prediction
        prediction = predict(text, target_lang, tokenizer)
        # Return the prediction
        return jsonify({'translation': prediction})
    except:
        return jsonify({'error': 'Error during prediction'})

# Stream video with face detection
@app.route('/detect')
def detect():
    # Start the camera so we generate predictions
    start_camera()
    return Response(generate_predictions(),mimetype='multipart/x-mixed-replace; boundary=frame')

# Stop the video stream
@app.post('/stop_camera')
def stop_streaming():
    stop_camera()
    return jsonify({'message': 'Camera stopped'})



