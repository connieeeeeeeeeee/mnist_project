import streamlit as st 
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# load the trained model
model = tf.keras.models.load_model("mnist_model.h5")

st.title("MNIST Digit Classifier")
st.write("Upload a 28*28 grayscale image of a digit")

uploaded_file = st.file_uploader("Choose a digit image...",
type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')   # convert to grayscale
    image = ImageOps.invert(image)   # change to black and white
    image = ImageOps.invert(image)  
    image = image.resize((28,28))   # resize to 28 by 28
    img_array = np.array(image) / 255.0   # normalise
    img_array = img_array.reshape(1,28,28)

    st.image(image, caption="Upload Image", width=150)

    prediction = model.predict(img_array)
    st.write(f"Prediction: **{np.argmax(prediction)}**")

print("connie")
