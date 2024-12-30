# AI-Powered Web Application

This project is a multi-functional AI web application that integrates Machine Translation, Image Captioning, and Face Detection. It demonstrates the capabilities of modern AI/ML models through a user-friendly web interface, built with Flask and powered by a Dockerized backend architecture.

---

## Features

### 1. Machine Translation
- Translate text into multiple languages.
- Supports English, Filipino, Hindi, Japanese, and Bahasa Indonesia.
- Built with the MT5 model for accurate and context-aware translations.

### 2. Image Captioning
- Automatically generate descriptive captions for uploaded images.
- Uses an Encoder-Decoder architecture with Inception V3 for encoding and an LSTM with attention for decoding.

### 3. Face Detection
- Real-time face detection with live video streaming.
- Powered by a YOLO-based model to identify faces and annotate them with bounding boxes.

---

## Technology Stack

- **Backend**: Flask for API endpoints and rendering web pages.
- **Frontend**: HTML, CSS, and JavaScript for an interactive user interface.
- **AI Models**:
  - MT5 for machine translation.
  - Inception V3 + LSTM for image captioning.
  - YOLO for face detection.
- **Deployment**:
  - Docker and Docker Compose for containerized services.
  - NGINX as a reverse proxy for serving the application.

---

## Getting Started

### Prerequisites

Ensure the following are installed on your machine:
- Python 3.8+
- Docker and Docker Compose

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/ai-powered-web-app.git
   cd ai-powered-web-app

2. **Build and Run the Docker Containers**:
  ```bash
  docker-compose up --build
  ```
3. **Access the Application**:
Open your web browser and navigate to http://localhost to access the homepage.
4. **Explore the Features**:
 - Translation: Go to /translation 
 - Image Captioning: Go to /image_captioning
 - Face Detection: Go to /face_detector

### File Structure
.
├── app/
│   ├── app.py                # Main Flask application
│   ├── dataset.py            # Dataset handling for captioning
│   ├── translator_utils.py   # Translation utilities
│   ├── face_detector_utils.py# Face detection utilities
│   └── image_captioning.py   # Image captioning models
├── static/
│   ├── css/                  # Stylesheets for frontend
│   ├── js/                   # JavaScript files for frontend
│   └── models/               # Trained AI/ML models
├── templates/
│   ├── home.html             # Homepage
│   ├── translation.html      # Translation page
│   ├── image_captioning.html # Image captioning page
│   └── face_detector.html    # Face detection page
├── Dockerfile                # Dockerfile for Flask app
├── docker-compose.yml        # Docker Compose for app and NGINX
└── README.md                 # Project documentation



