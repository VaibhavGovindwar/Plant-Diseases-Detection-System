# Plant Disease Detection System

## 🌱 Project Overview

- This project is a Plant Disease Detection System that uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify plant diseases from images.  The trained model is integrated into a Flask-based web application, allowing users to upload images and get real-time predictions. This can help farmers and researchers identify plant diseases early, improving crop health and yield.

## 🚀 Features

- 🌿 Upload an image of a plant leaf

- 🔍 Uses a trained CNN model for disease classification

- 📊 Supports multiple plant species & diseases

- 🖼 Displays the predicted disease with the uploaded image

- 📂 Deployed on Vercel for online access

## 🛠 How It Works

- The user uploads a leaf image to the web application.
- The image is preprocessed and resized to fit the model's input dimensions.
- The trained CNN model predicts the disease based on the image.
- The predicted disease name is displayed along with the uploaded image.
- Users can then take action based on the prediction (e.g., apply treatments).

## Dataset 
- The model was trained using a dataset sourced from Kaggle. You can find the dataset [here](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data).

## 📊 Model Performance

The model was trained on a plant disease dataset using TensorFlow/Keras. Below are some key performance metrics:

Training Accuracy: 95%

Validation Accuracy: 90%

Loss: Low loss achieved through Adam optimizer and categorical cross-entropy loss function.

Optimization: Used EarlyStopping and data augmentation for improved performance.

## 🛠 Tech Stack

- Backend: Flask

- Machine Learning: TensorFlow, Keras

- Image Processing: OpenCV, NumPy

- Visualization: Matplotlib, Seaborn

- Deployment: GitHub, Vercel

## 🖥️ Usage of Web Page

1️⃣ Home Page (index.html)

- Displays a background image with a welcoming message.
- Users can upload an image of a plant leaf using the file upload button.
- Clicking the Predict Leaf button sends the image for processing.

2️⃣ Image Processing & Prediction

- The uploaded image is resized and normalized before being fed into the CNN model.
- The model analyzes the image and predicts whether the plant is healthy or has a disease.

3️⃣ Result Page (result.html)

- Shows the uploaded image for reference.
- Displays the predicted disease name in bold text.
- Provides a "Predict Another Image" button to return to the home page and start a new prediction.

4️⃣ User-Friendly Design

- Responsive UI: Works across different devices.
- Dark-themed background for a modern look.
- Hover effects on buttons for an interactive feel.
- Fast and accurate results powered by TensorFlow.

## 🔧 Setup & Installation

1️⃣ Clone the Repository

  ```bash
  git clone https://github.com/VaibhavGovindwar/Plant-Disease-Detection-System.git
  cd Plant-Disease-Detection-System 
  ```

2️⃣ Install Dependencies

  ```bash 
  pip install -r requirements.txt
  ```

 3️⃣ Run the Flask App

   ```bash
   python app.py
   ```
- Note: When you run the flask app, the model will automatically start downloading to your system.

## 🌍 Deployment on Vercel

1️⃣ Install Vercel CLI
```bash
npm install -g vercel
```

2️⃣ Deploy
```bash
vercel
```

Code overview [here](https://www.kaggle.com/code/vaibhavgovindwar/plant-disease-detection-system)
