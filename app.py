import tensorflow as tf
print("TensorFlow version:", tf.__version__)

import streamlit as st # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model_from_drive():
    model = load_model('best_model.h5') 
    return model

model = load_model_from_drive()

# Constants
IMG_SIZE = 256

# Prediction Function
def predict_uploaded_image(uploaded_img):
    img = uploaded_img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "Real" if prediction >= 0.5 else "Fake"
    confidence = prediction if prediction >= 0.5 else 1 - prediction
    return label, confidence

# Streamlit UI
st.title("Deepfake Detection Web App")
st.write("Upload an image to check whether it is Real or Fake.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)


    label, confidence = predict_uploaded_image(img)

    st.markdown(f"### Prediction: **{label}**")
    st.markdown(f"### Confidence: **{confidence*100:.2f}%**")
