from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# Load pre-trained emotion detection model
emotion_model = load_model('models/emotion_model.h5')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def detect_emotion(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        return "No face detected"
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float') / 255.0
        roi_gray = img_to_array(roi_gray)
        roi_gray = np.expand_dims(roi_gray, axis=0)
        
        prediction = emotion_model.predict(roi_gray)[0]
        emotion = emotion_labels[np.argmax(prediction)]
        return emotion
    
    return "No emotion detected"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    emotion = detect_emotion(filepath)
    return jsonify({"emotion": emotion})

@app.route('/capture', methods=['POST'])
def capture():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    filename = secure_filename("captured_image.png")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    emotion = detect_emotion(filepath)
    return jsonify({"emotion": emotion})

if __name__ == '__main__':
    app.run(debug=True)
