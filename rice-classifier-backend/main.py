from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Updated to include port 3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute path to the model in root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
model_path = os.path.join(parent_dir, 'model.h5')
print("parent_dir", parent_dir)
print("model_path", model_path)

# Load model and classes
model = tf.keras.models.load_model(model_path)
with open(os.path.join(current_dir, 'classes.json'), 'r') as f:
    classes = json.load(f)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess the image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Resize image to match model's expected input size (224x224)
    image = image.resize((224, 224))
    
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    
    # Add batch dimension
    processed_image = np.expand_dims(image_array, axis=0)
    
    # Make prediction
    predictions = model.predict(processed_image)[0]
    
    # Get all predictions with their probabilities
    predictions_list = []
    for i, prob in enumerate(predictions):
        predictions_list.append({
            "class": classes[i],
            "probability": float(prob)
        })
    
    # Sort predictions by probability in descending order
    predictions_list.sort(key=lambda x: x["probability"], reverse=True)
    
    return {"predictions": predictions_list}

@app.get("/classes")
async def get_classes():
    return {"classes": classes} 