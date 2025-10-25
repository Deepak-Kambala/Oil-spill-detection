from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import io
import os
import base64

app = Flask(__name__)

# Lazy loading model
model = None
labels = ['Non Oil Spill', 'Oil Spill']

def get_model():
    global model
    if model is None:
        print("Loading model...")
        model = load_model('m_model.h5')
    return model

def load_and_preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')

    if not file:
        return render_template('index.html', error="No file uploaded. Please upload an image.")

    # Read bytes once
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode('utf-8')  # Keep image persistent

    img = load_and_preprocess_image(image_bytes)
    if img is None:
        return render_template('index.html', error="Error processing image.", image_data=image_data)

    try:
        model = get_model()
        prediction = model.predict(img)
        predicted_class = (prediction > 0.5).astype("int32")[0][0]
        result_label = labels[predicted_class]

        return render_template(
            'index.html',
            prediction=result_label,
            image_data=image_data
        )

    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', error="Error during prediction.", image_data=image_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
