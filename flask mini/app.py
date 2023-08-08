from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

app = Flask(__name__, static_folder='static')

import tensorflow as tf
import tensorflow_addons as tfa
from keras.models import load_model

# Define custom metric function
f1_score = tfa.metrics.F1Score(num_classes=5)

# Load saved model
with tf.keras.utils.custom_object_scope({'F1Score': f1_score}):
    model = load_model(r'C:/Users/DELL/Downloads/model_alziemers_47.h5')


# Load the machine learning model
class_names = ['Final AD JPEG','Final CN JPEG','Final EMCI JPEG','Final LMCI JPEG','Final MCI JPEG']  # Replace with your class names

# Define a function to preprocess the image
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize the image to match the model's input shape
    image = np.array(image)  # Convert image to numpy array
    image_resized = cv2.resize(image, (128, 128))
    if len(image_resized.shape) == 2:
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    # Add an extra dimension to the array to create a batch of size 1
    image_batch = np.expand_dims(image_rgb, axis=0)
    # Normalize the pixel values
    image_batch = image_batch / 255.0
    return image_batch

# Define the Flask route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['image']       
            # Preprocess the image
        image = Image.open(file)
        preprocessed_image = preprocess_image(image)
        
        # Make prediction using the model
        prediction = model.predict(preprocessed_image)[0]
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)
        
        # Render the result template with the predicted class and confidence
        return render_template('result.html', class_name=predicted_class, confidence=confidence)
    
    # Render the home template for GET requests
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=5000, debug=True)
