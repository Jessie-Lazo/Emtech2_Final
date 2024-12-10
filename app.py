import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Update the path to your saved model
model_path ='fashion_mnist_cnn.h5'

# Load the trained model
model = load_model(model_path)

# Class labels for Fashion MNIST
class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

st.title("Fashion MNIST Classification")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=[0, -1])  # Add batch and channel dimensions

    # Make prediction
    prediction = model.predict(image)
    predicted_label = class_labels[np.argmax(prediction)]

    st.write(f"Predicted Label: **{predicted_label}**")
