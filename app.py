from flask import Flask, render_template, jsonify, request, session, redirect, url_for
import tensorflow as tf
import numpy as np
from os import path, walk
import cv2
import base64
from PIL import Image
from io import BytesIO
from collections import deque
from dotenv import load_dotenv
import firebase_admin
import os

from firebase_admin import credentials, storage, db

UPLOAD_FOLDER = 'static/uploads'
cred = credentials.Certificate("credentials.json")
FIREBASE_STORAGE_BUCKET = os.getenv('FIREBASE_STORAGE_BUCKET')
FIREBASE_DATABASE_URL = os.getenv('FIREBASE_DATABASE_URL')
SECRET_KEY = os.getenv('SECRET_KEY')

#import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = SECRET_KEY

firebase_admin.initialize_app(cred, {
    'storageBucket': FIREBASE_STORAGE_BUCKET,
    'databaseURL': FIREBASE_DATABASE_URL
})

# Firebase Storage
bucket = storage.bucket()
# Firebase Realtime Database
root = db.reference()
#config to update when template changes
app.config['TEMPLATES_AUTO_RELOAD'] = True
# Load the model
model_path = './static/MobileNetV2_model4.h5'
mode = tf.keras.models.load_model(model_path)
print('Model loaded. Check http://')

# Set the threshold for the number of frames before prediction
FRAME_THRESHOLD = 16

# Use deque to store frames
frames_deque = deque(maxlen=FRAME_THRESHOLD)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if the username and password are correct
        if is_valid_user(username, password):
            session['logged_in'] = True
            return redirect(url_for('reports'))
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html') 

@app.route('/')
def index():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')
CLASSES_LIST = ["NonViolence", "Violence"]
@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    try:
        data = request.get_json()
        frame_image_data = data.get('frame_data')

        # Convert base64 frame data to a NumPy array
        frame_data = base64.b64decode(frame_image_data)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Resize the frame to fixed dimensions
        resized_frame = cv2.resize(frame, (224, 224))

        # Normalize the resized frame
        normalized_frame = resized_frame / 255

        # Add the frame to the deque
        frames_deque.append(normalized_frame)

        # Check if the deque has reached the threshold
        if len(frames_deque) >= FRAME_THRESHOLD:
            # Call the predict_video function with the processed frames
            predicted_class, confidence = predict_video(list(frames_deque))
            print(predicted_class, confidence,'isworking')
            newconf = confidence.astype(float)
            # Return the predicted values to the client
            return jsonify({'status': 'success', 'predicted_class': predicted_class, 'confidence': newconf})

        return jsonify({'status': 'success'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


def predict_video(frames_list):
    # Placeholder for your video prediction logic
    # This assumes frames_list is a list of pre-processed frames (NumPy arrays)
    print('Predicting video...')
    # Passing the pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities = mode.predict(np.expand_dims(frames_list, axis=0))[0]
    # Convert predicted probabilities to a list of floats
    predicted_probabilities_list = predicted_labels_probabilities.tolist()

    # # Get the index of the class with the highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)
    confidence=predicted_labels_probabilities[predicted_label]

    # # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]

    # # Display the predicted class along with the prediction confidence.
   # print(f'Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')

    return predicted_class_name, confidence

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'frame_data' not in request.json:
        print('No image data found')
        return jsonify({'error': 'No image data found'})

    frame_data = request.json['frame_data']
    room_number = request.json.get('room_number')
    time = request.json.get('time')

    try:
        image_data = base64.b64decode(frame_data)
    except Exception as e:
        return jsonify({'error': 'Invalid image data'})

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    filename = f"{room_number}_{time}.jpg" 
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    print(image_path)

    with open(image_path, 'wb') as f:
        f.write(image_data)


    try:
        bucket = storage.bucket()
        blob = bucket.blob(filename)
        blob.upload_from_filename(image_path)
        blob.make_public()
        image_url = blob.public_url
        print(image_url)
    except Exception as e:
        return jsonify({'error': 'Failed to upload image to Firebase Storage'})

    os.remove(image_path)

    details_ref = root.child('details').push()
    details_ref.set({
        'image_url': image_url,
        'room_number': room_number,
        'time': time
    })

    return jsonify({'success': True})

@app.route('/details', methods=['GET'])
def fetch_details():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 5))

    time = request.args.get('time')
    room_number = request.args.get('room_number')

    details_ref = root.child('details')
    if time:
        details_ref_time = details_ref.order_by_child('time').equal_to(time)
    else:
        details_ref_time = details_ref

    if room_number:
        details_ref_room = details_ref.order_by_child('room_number').equal_to(room_number)
    else:
        details_ref_room = details_ref

    details = {}
    if time and room_number:
        for key, value in details_ref_time.get().items():
            if key in details_ref_room.get().keys():
                details[key] = value
    else:
        details = details_ref_time.get() if time else details_ref_room.get()

    total_count = len(details)
    start_index = (page - 1) * per_page
    end_index = start_index + per_page

    if details:
        details_list = list(details.values())[start_index:end_index]
        return jsonify({
            'total_pages': -(-total_count // per_page),
            'data': details_list
        })
    else:
        return jsonify({'error': 'No details found'}), 404
    
@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

# Reports route
@app.route('/reports')
def reports():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('reports.html')

def is_valid_user(username, password):
    valid_username = 'admin'
    valid_password = 'admin'
    return username == valid_username and password == valid_password
    
if __name__ == '__main__':
    app.run(debug=True)
