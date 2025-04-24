from flask import Flask, request, render_template, url_for
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Paths to the saved models
DENSENET_MODEL_PATH = "model/densenet121_model.keras"
XCEPTION_MODEL_PATH = "model/xception_model.keras"

# Load the saved models
densenet_model = tf.keras.models.load_model(DENSENET_MODEL_PATH)
xception_model = tf.keras.models.load_model(XCEPTION_MODEL_PATH)

# Create the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Image preprocessing function
def preprocess_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction function
def predict_custom_image(image_path):
    img_array = preprocess_image(image_path)

    # Predict using DenseNet-121
    densenet_prediction = densenet_model.predict(img_array)
    densenet_result = "Tumor" if densenet_prediction[0][0] > 0.5 else "No Tumor"

    # Predict using Xception
    xception_prediction = xception_model.predict(img_array)
    xception_result = "Tumor" if xception_prediction[0][0] > 0.5 else "No Tumor"

    # Combine results
    predictions = {
        "DenseNet-121": {"Prediction": densenet_result, "Confidence": densenet_prediction[0][0]},
        "Xception": {"Prediction": xception_result, "Confidence": xception_prediction[0][0]}
    }
    return predictions

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None
    image_path = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return "No image uploaded", 400

        image = request.files['image']
        if image.filename == '':
            return "No image selected", 400

        # Save the uploaded file
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename).replace("\\", "/")
        image.save(image_path)

        # Perform prediction
        predictions = predict_custom_image(image_path)

    return render_template('index.html', predictions=predictions, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
