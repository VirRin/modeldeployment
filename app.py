from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)

# Load your trained model
model = tf.keras.models.load_model(r'512.h5')

# Preprocessing function
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.convert("RGB")
    image = image.resize((512, 512))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    file.save("uploaded_image.jpg")  # Save the uploaded image
    image_array = preprocess_image("uploaded_image.jpg")
    
    # Make prediction
    prediction = model.predict(image_array)
    confidence = prediction[0][0]
    predicted_class = 'healthy' if confidence > 0.5 else 'disease'
    confidence = confidence if predicted_class == 'healthy' else 1 - confidence
    confidence_percentage = f"{confidence:.2%}"
    
    return jsonify({
        "class": predicted_class,
        "confidence": confidence_percentage
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
