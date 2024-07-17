
// Function to ask for camera permission
function askForCameraPermission() {
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

// Ask for camera permission
askForCameraPermission()
    // Permission granted
    .then(function(stream) {
        console.log('Permission Granted');
    })
    // Permission denied
    .catch(function(err) {
        // Log error
        console.error("Error accessing camera:", err);
    });

// Event listener for when the page is close to close camera
window.addEventListener('beforeunload', function(event) {
    // Stop the camera stream and release camera
    this.fetch('/stop_camera', { method: 'POST' })
        .then(response => {
            if (!response.ok) {
                console.error('Error stopping camera');
            }            
        })
        .catch( error => {
            console.error('Error stopping streaming:', error);
        });
});