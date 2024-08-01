
// Face Detector class
class FaceDetector {
    constructor() {
        // Buttons to stop and start camera
        this.startButton = document.getElementById('start_button');
        this.stopButton = document.getElementById('stop_button');
        // Video that captures the feed
        this.video = document.getElementById('video');
        // Canvas to draw the feed
        this.canvas = document.getElementById('video_capture');
        this.context = this.canvas.getContext('2d');
        //
        this.videoElement = document.getElementById('face_detector');
        // Variables to check if the camera is streaming and the stream itself
        this.streaming = false;
        this.stream = null;
        this.initialize();
    }

    initialize() {
        // Add event listener to start and stop button
        this.startButton.addEventListener('click', () => this.onStartButton());
        this.stopButton.addEventListener('click', () => this.onStopButton());
    }
    // Starting the camera through flask
    async onStartButton() {
        try {
            // Waits for camera permission and gets video feed when it does
            this.stream = await navigator.mediaDevices.getUserMedia({ video: true });
            this.video.srcObject = this.stream;
            this.video.addEventListener('loadedmetadata', () => {
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
                this.streaming = true;
                // Starts processing the video feed for face detection
                this.startProcessing();
        });
        } catch (error){
            alert('Error accessing camera:', error);
            console.error('Error accessing camera:', error);
        }
    }
    // Function to start processing the video feed, gets the feed and sends it to Flask app
    async startProcessing() {
        while(this.streaming) {
            // Draws the video feed on the canvas
            this.context.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            // Converts the canvas to a frame for inference
            this.canvas.toBlob(async (blob) => {
                // Creates a form to store image
                const formData = new FormData();
                formData.append('image', blob);
                try{
                    const response = await fetch($SCRIPT_ROOT + '/process_frame', {
                        method: 'POST',
                        body: formData,
                    });
                    // If the response is not ok, throw an error
                    if(!response.ok) {
                        throw new Error('Failed to process frame');
                    }
                    // Get the image blob from the response
                    const imgBlob = await response.blob();
                    // Create a URL for the image blob
                    const imageURL = URL.createObjectURL(imgBlob);
                    // Update the video element with the image
                    this.updateVideo(imageURL);
                } catch (error){
                    console.error('Failed to process frame:', error);
                }
            }, 'image/jpeg');
            await new Promise(resolve => setTimeout(resolve, 175)); // Have to keep the frame low so inference can keep up
        }
    }
    // Stopping the camera through flask
    onStopButton() {
        // Stops processing the video feed
        this.streaming = false;
        // Stops the video feed
        this.video.srcObject.getTracks().forEach(track => track.stop());
    }
    // Function to ask for camera permission
    updateVideo(URL) {
        // Update the video element with the labeled frame
        this.videoElement.src = URL;
    }
}

// Create a new face detector object
const faceDetector = new FaceDetector();

