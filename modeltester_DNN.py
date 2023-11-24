import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]
# Load the DenseNet201 model
model = load_model('EmotionRecognitionDenseNet201.h5')  # Replace with the actual path to your model

# Load the Haarcascades face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the image
img_path = "test_images/im0.png"
img = Image.open(img_path).convert("RGB")  # Convert to RGB
img_array = np.array(img)

# Convert image to grayscale for face detection
gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Load and preprocess each detected face
for (x, y, w, h) in faces:
    face_img = img_array[y:y + h, x:x + w]  # Extract the face region
    face_img = cv2.resize(face_img, (48,48))  # Resize to match DenseNet201 input size
    face_img = preprocess_input(np.expand_dims(face_img, axis=0))  # Preprocess input for the model

    # Make the emotion prediction
    result = model.predict(face_img)
    img_index = np.argmax(result)

    # Print the predicted emotion label
    print(f"Emotion for face at ({x}, {y}, {w}, {h}): {emotion_labels[img_index]}")

    # Draw a rectangle around the detected face
    cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the image with face detection and emotion recognition
# img = Image.fromarray(img_array)
height, width = (1600, 900)

max_height = 800
max_width = 600
if height > max_height or width > max_width:
    scale = min(max_height / height, max_width / width)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

cv2.imshow('Emotion Recognition using DNN model', img)
cv2.waitKey(0)
cv2.destroyAllWindows()