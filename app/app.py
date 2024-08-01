# For handling the website routes and requests
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
# For logging/debugging
import sys
import io

#-------------------------------------------------------------------------------------------------------------------------------------------------------------#

# Translation functions
from utils.translator.translator_utils import predict, tokenizer
# Image Captioning functions
from utils.captioning.image_captioning_utils import predict_caption
# Face detector functions
from utils.face_detection.face_detector_utils import generate_prediction

#-------------------------------------------------------------------------------------------------------------------------------------------------------------#

# Create instance of a Flask application
app = Flask(__name__)
# Allowes request from any origin to /caption endpoint
cos = CORS(app, resources={r"/caption": {"origins": "*"}})
# Define the secret key for the application
app.config['SECRET_KEY'] = 'secret_key'

#-------------------------------------------------------------------------------------------------------------------------------------------------------------#

####-- Rendering Pages --####

#-------------------------------------------------------------------------------------------------------------------------------------------------------------#

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

#-------------------------------------------------------------------------------------------------------------------------------------------------------------#

####-- Prediction Endpoints --####

#-------------------------------------------------------------------------------------------------------------------------------------------------------------#

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

#-------------------------------------------------------------------------------------------------------------------------------------------------------------#

@app.route('/caption', methods=['POST'])
def caption():
    
    # Check if the request is a POST request, so we don't get error
    if request.method not in ['POST']:
        return jsonify({'error': 'POST requests only'})
    # Try to get the image and caption it
    try:
        # Get the image from the POST request
        data = request.files
        # Check if data is empty
        if data is None:
            return jsonify({'error': 'No data found'})
        # Get the image from the POST request
        image = data['image'].stream
        # Get the caption of the image
        caption = predict_caption(image)
        # Return caption
        return jsonify({'caption': str(caption)})
    except Exception as e:
        print(e)
        return jsonify({'error': 'Error during prediction'})

#-------------------------------------------------------------------------------------------------------------------------------------------------------------#

# Stream video with face detection
@app.route('/process_frame', methods=['POST'])
def detect():
    # Check if the request is a POST request, so we don't get error
    if request.method not in ['POST']:
        print("Error: POST requests only")
        return jsonify({'error': 'POST requests only'})
    # Try to get the image and caption it
    try:
        # Get the image from the POST request
        data = request.files
        # Check if data is empty
        if data is None:
            return jsonify({'error': 'No data found'})
        # Get the frame from the POST request
        frame = data['image'].stream
        # Get the labeled frame as a byte stream
        labeled_frame = generate_prediction(frame)
        # Return labeled frame
        return send_file(io.BytesIO(labeled_frame), mimetype='image/jpeg')
    except Exception as e:
        print(e)
        return jsonify({'error': 'Error during prediction'})

#-------------------------------------------------------------------------------------------------------------------------------------------------------------#

