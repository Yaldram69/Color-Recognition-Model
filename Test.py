import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import json

# Load the trained model
model = load_model('color_classification_model.h5')

# Load class indices
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Reverse the class indices for prediction
color_labels = {v: k for k, v in class_indices.items()}

# Initialize webcam
cap = cv2.VideoCapture(0)

def predict_color(image):
    """Predict color of the given image using the trained model."""
    resized_image = cv2.resize(image, (16,16))  # Resize to match the model input
    img_array = img_to_array(resized_image) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    return color_labels.get(predicted_class, "Unknown")

def detect_colors(frame):
    """Detect and label colors in the given frame."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges in HSV for segmentation
    color_ranges = {
        'Black': ((0, 0, 0), (180, 255, 50)),
        'Blue': ((100, 150, 0), (140, 255, 255)),
        'Gray': ((0, 0, 80), (180, 50, 200)),
        'Green': ((35, 50, 50), (85, 255, 255)),
        'White': ((0, 0, 200), (180, 30, 255)),
        'Yellow': ((20, 100, 100), (30, 255, 255)),
        'Purple': ((130, 50, 50), (160, 255, 255)),
        'Orange': ((10, 100, 100), (25, 255, 255)),
        'Red': ((0, 100, 100), (10, 255, 255))
    }

    frame_resized = cv2.resize(frame, (640, 480))  # Resize frame for faster processing
    hsv_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

    # Loop through each color range
    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
        mask = cv2.GaussianBlur(mask, (5, 5), 0)  # Apply blur to reduce noise
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter out small contours
                x, y, w, h = cv2.boundingRect(contour)
                roi = frame_resized[y:y + h, x:x + w]

                # Predict color of the region
                predicted_color = predict_color(roi)

                # Draw bounding box and label
                cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame_resized, f"{predicted_color}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame_resized

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Detect and label colors in the frame
    labeled_frame = detect_colors(frame)

    # Display the resulting frame
    cv2.imshow('Color Detection', labeled_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
