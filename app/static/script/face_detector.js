// Face Detector class
class FaceDetector {
    constructor() {
        this.startButton = document.getElementById('start_button');
        this.stopButton = document.getElementById('stop_button');
        this.videoElement = document.getElementById('stream');
    }
    initialize() {
        // Ask for camera permission
        this.askForCameraPermission()
        // Add event listener to start button
        this.startButton.addEventListener('click', () => this.onStartButton());
        
    }
    // Starting the camera through flask
    onStartButton() {
        // Check if permission is granted
        this.checkPermission().then(permission => {
            if(permission){
                // Add event listener to stop button
                this.stopButton.addEventListener('click', () => this.onStopButton());
                console.log('Starting Camera');
                // Set the video element source to the detect route
                this.videoElement.src = "http://127.0.0.1:5000/detect";
            } else{
                // Log error
                console.error('Permission Denied');
                // Tell user to grant camera permission
                alert("Please grant camera permission to use ");
            }
        });
        
    }
    // Stopping the camera through flask
    onStopButton() {
        console.log('Stopping Camera');
        this.videoElement.src = "";
        // Calls flask to stop the camera stream
        fetch('http://127.0.0.1:5000/stop_camera', { method: 'POST' })
        .then(response => {
            if (!response.ok) {
                console.error('Error stopping camera');
            }            
        })
        .catch( error => {
            console.error('Error stopping streaming:', error);
        });
    }
    // Function to ask for camera permission
    askForCameraPermission() {
        // Creates a promise to handle the camera permission
        return new Promise(function(resolve, reject) {
            // Uses navigator.mediaDevices.getUserMedia to ask for camera permission
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(function(stream) {
                    // If permission is granted, resolve the promise with the stream
                    resolve(stream);
                })
                .catch(function(err) {
                    // If permission is denied, reject the promise with the error
                    reject(err);
                });
            });
    }
    //Function to check if permission is granted
    checkPermission() {
        return navigator.permissions.query({name: 'camera'})
        .then(permission => permission.state === 'granted');
    }
}

// Create a new face detector object
const faceDetector = new FaceDetector();
// Initialize the face detector
faceDetector.initialize();

