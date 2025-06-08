import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# Set page config
st.set_page_config(
    page_title="Rice Classifier",
    page_icon="🌾",
    layout="centered"
)

# Load model and classes
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model

@st.cache_data
def load_classes():
    with open('classes.json', 'r') as f:
        return json.load(f)

def preprocess_image(image, target_size=(128, 128)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def main():
    st.title("🌾 Rice Variety Classifier")
    st.write("Upload an image of rice to classify its variety.")

    # Load model and classes
    try:
        model = load_model()
        classes = load_classes()
    except Exception as e:
        st.error(f"Error loading model or classes: {str(e)}")
        st.info("Please make sure model.h5 and classes.json are in the correct location.")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make prediction
        if st.button("Classify"):
            with st.spinner("Classifying..."):
                # Preprocess image
                processed_image = preprocess_image(image)

                # Make prediction
                predictions = model.predict(processed_image)[0]
                
                # Get top 3 predictions
                top_3_idx = np.argsort(predictions)[-3:][::-1]
                
                # Display results
                st.subheader("Classification Results")
                
                # Create columns for results
                cols = st.columns(3)
                
                for idx, col in zip(top_3_idx, cols):
                    with col:
                        st.metric(
                            label=classes[idx],
                            value=f"{predictions[idx]*100:.1f}%"
                        )

if __name__ == "__main__":
    main() 