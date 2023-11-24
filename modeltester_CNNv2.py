import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']
model = load_model('EmotionRecognitionCNN.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
image_path = r"test_images/Rohit-Sharma.png"
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in faces:
    face_img = gray[y:y + h, x:x + w]
    face_img = cv2.resize(face_img, (48, 48))
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=-1)
    result = model.predict(face_img)
    result = list(result[0])
    img_index = result.index(max(result))

    predicted_emotion = emotion_labels[img_index]
    cv2.putText(img, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

height, width = img.shape[:2]
max_height = 800
max_width = 600
if height > max_height or width > max_width:
    scale = min(max_height / height, max_width / width)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

cv2.imshow('Emotion Recognition using CNN model', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
