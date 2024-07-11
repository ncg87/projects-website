from flask import Flask, request, jsonify, render_template

from translator_utils import predict, tokenizer

# Create instance of a Flask application
app = Flask(__name__)

# Render Home Page
@app.get('/')
def home_page():
    return render_template('home.html')

# Render Translation Page
@app.get('/translate')
def predict_page():
    return render_template('translate.html')

@app.get('/image_captioning')
def captioning():
    return render_template('image_captioning.html')
    
# Predict translation for specified text and target language
@app.post('/predict')
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

