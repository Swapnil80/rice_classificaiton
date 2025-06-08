from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and classes
model = tf.keras.models.load_model('model.h5')
with open('classes.json', 'r') as f:
    classes = json.load(f)

def preprocess_image(image_bytes):
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize((128, 128))
    
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    
    # Preprocess image
    processed_image = preprocess_image(contents)
    
    # Make prediction
    predictions = model.predict(processed_image)[0]
    
    # Format results
    results = [
        {
            "class": classes[i],
            "probability": float(predictions[i])
        }
        for i in range(len(classes))
    ]
    
    # Sort results by probability in descending order
    results.sort(key=lambda x: x['probability'], reverse=True)
    
    return {"predictions": results}

@app.get("/classes")
async def get_classes():
    return {"classes": classes} 