# To handle model
import PIL.Image
import ultralytics
# To draw bounding boxes and labels
import supervision as sv
# To process video frames
import PIL
import io

# Load the trained face detector model
model = ultralytics.YOLO('./models/best_face_detector.pt')


# Bounding box and label annotator
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Converts the images bytes to a image for processing
def process_frame(image):
    # Read image bytes
    image.seek(0)
    image_bytes = image.read()
    # Convert bytes to image
    image = PIL.Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # Return image
    return image

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
def generate_prediction(image):
    image = process_frame(image)
    # Draws bounding boxes
    frame = detect_face(image)
    # Convert PIL image to byte stream
    buffered = io.BytesIO()
    frame.save(buffered, format="JPEG")
    frame_byte_arr = buffered.getvalue()
    
    return frame_byte_arr