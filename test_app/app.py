# For handling the website routes and requests
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin

# Create instance of a Flask application
app = Flask(__name__)
# Allowes request from any origin to '/' endpoint
cos = CORS(app, resources={r"/file": {"origins": "*"}})

# Render Home Page
@app.get('/')
def home_page():
    return render_template('file_test.html')

# Get the file
@app.route('/file', methods=['POST'])
def get_file():
  
    # Check if the request is a POST request, so we don't get error
    if request.method not in ['POST']:
        return jsonify({'error': 'POST requests only'})
    
    print("Request Received")
    file = request.files
    print(file)
    a  = "The old man walked 5 miles in old shoes for many many hours"
    
    print(type(a))
    print(type(str(a)))
    # I don't know why sending over 'a' compared to 'str(a)' makes a difference
    # but str(a) works
    return jsonify({'caption': a})