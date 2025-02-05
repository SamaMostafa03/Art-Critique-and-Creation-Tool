from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model
try:
    model = tf.keras.models.load_model("genres_styles_classification_model.h5")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise RuntimeError("Failed to load model.")

# Load class labels
def load_classes(filename):
    try:
        with open(filename, "r") as f:
            classes = json.load(f)
        return {int(k): v for k, v in classes.items()}
    except Exception as e:
        logger.error(f"Error loading class labels from {filename}: {e}")
        return {}

style_classes = load_classes("style_classes.json")
genre_classes = load_classes("genre_classes.json")

app = FastAPI(title="WikiArt Classification API", description="Predicts the style and genre of an artwork.")

@app.get("/")
def read_root():
    """Root endpoint to check if API is running."""
    return {"message": "FastAPI is running!"}

def preprocess_image(image: Image.Image):
    """Preprocess the uploaded image."""
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict-style-genre")
async def predict(file: UploadFile = File(...)):
    """Predicts the style and genre of an uploaded artwork."""
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make predictions
        predictions = model.predict(processed_image)
        genre_pred = np.argmax(predictions[0])
        style_pred = np.argmax(predictions[1])
        
        return {
            "genre": genre_classes.get(genre_pred, "Unknown"),
            "style": style_classes.get(style_pred, "Unknown")
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
