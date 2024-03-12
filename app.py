from flask import Flask, render_template, jsonify, request
import tensorflow as tf
import numpy as np
from os import path, walk
import cv2
import base64
from collections import deque
#import matplotlib.pyplot as plt

app = Flask(__name__)
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

@app.route('/')
def index():
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

# ... (your existing code)

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

if __name__ == '__main__':
    app.run(debug=True)
