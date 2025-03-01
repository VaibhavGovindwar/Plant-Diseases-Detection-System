from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
import os
import cv2
import gdown
from werkzeug.utils import secure_filename

# Initialize the Flask application
app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to download the model from Google Drive https://drive.google.com/file/d/18mrZAQKbPDXtWRxoa4Zm--06MYbqN3PV/view?usp=sharing
def download_model():
    # The Google Drive file ID from my shareable link
    file_id = '18mrZAQKbPDXtWRxoa4Zm--06MYbqN3PV'  
    if not os.path.exists('plant_disease_model.keras'):
        url = f'https://drive.google.com/uc?id={file_id}'
    
    # Download the model to the local directory
        gdown.download(url, 'plant_disease_model.keras', quiet=False, fuzzy=True)
    else:
        print("Model already exists. Skipping download.")

# Load the model
# MODEL_PATH = (r"plant_disease_model.keras")
download_model()
try:
    model = tf.keras.models.load_model('plant_disease_model.keras')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Class labels
class_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_...', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess the image
        try:
            img = cv2.imread(file_path)  # Read the image file from the provided file path
            img = cv2.resize(img, (224, 224))  # Resize to model input size
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #  image color format from BGR (used by OpenCV) to RGB
            img = img.astype("float32") / 255.0 # Normalizes the image pixel values to a range of 0 to 1 by dividing by 255.0
            img = np.expand_dims(img, axis=0)
            img = img.reshape(1,224,224,3)

            # Predict
            predictions = model.predict(img) # Passes the preprocessed image to the machine learning model for prediction.
            predicted_class = np.argmax(predictions, axis=-1)[0] # the class index with the highest probability
            predicted_label = class_labels[predicted_class] # predicted class index to its corresponding label

            # # Render the result page, passing the predicted label and image path
            return render_template('result.html', label=predicted_label, image_path=file_path)
        except Exception as e: # Catches any exceptions that occur during image preprocessing or model prediction. 
            return f"Error during prediction: {e}"

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
