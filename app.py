import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained model
model = load_model("model.h5")

st.title("Digit Recognition App")
st.write("Upload a 28x28 black & white digit image")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption='Uploaded Image', width=150)

    # Preprocess
    image = image.resize((28, 28))
    img_array = np.array(image)
    img_array = img_array.reshape(1, 28, 28)
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction)

    st.success(f"Predicted Digit: {pred_class}")
