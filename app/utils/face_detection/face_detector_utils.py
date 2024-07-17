# To handle model
import ultralytics
# To draw bounding boxes and labels
import supervision as sv
# To get video
import cv2
# To pause the video stream
import time
import sys

# Load the trained face detector model
model = ultralytics.YOLO('./models/best_face_detector.pt')

# Intialize camera and streaming variables
camera = None
streaming = False

# Bounding box and label annotator
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Run inference on the image and draw bounding boxes
def detect_face(image):
    # Make predictions
    results = model(image)[0]
    # Extract detections
    detections = sv.Detections.from_ultralytics(results)
    # Annotates the boxes with labels 
    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    # Return annotated image
    return annotated_image

# Reads the video stream and generates frames
def generate_predictions():
    
    # Get camera object
    global camera
    
    # Check if the camera is opened
    if not camera.isOpened():
        print("Error: Could not open camera.")
        sys.stdout.flush()
        return
    
    while streaming:
        # Get frame from the camera
        success, frame = camera.read()
        # Checks if frame is captured
        if success:
            # Draws bounding boxes
            frame = detect_face(frame)
            # Converts frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Encodes annotated image
            ret, buffer = cv2.imencode('.jpg', frame)
            # Convert image to bytes
            frame = buffer.tobytes()
        
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
                )
        else:
            print("Error: Could not read frame.")
            
# Start the camera and streaming
def start_camera():
    # Get global variables
    global camera, streaming
    # Check if camera is not opened
    if not streaming:
        # Start camera and streaming
        camera = cv2.VideoCapture(0)
        streaming = True

# Stop the camera and streaming
def stop_camera():
    # Get global variables
    global camera, streaming
    # Check if camera is opened
    if streaming:
        # Release the camera and stop streaming
        camera.release()
        streaming = False