import os
# Set writable cache directory
os.environ["HF_HOME"] = "/tmp"
os.environ["TRANSFORMERS_CACHE"] = "/tmp"
os.environ["HF_HUB_CACHE"] = "/tmp"
os.environ["TORCH_HOME"] = "/tmp"

import timm  
from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import io
import logging
import sys
import os
from aes_clip import AesCLIP_reg
import clip
import requests
import urllib.request



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ArtVision Scores API", description="Predicts visual attributes of an artwork.")

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
 
models_urls = {
    "total": "https://huggingface.co/Bambii-03/scores/resolve/main/best_Total_aesthetic_score_model.pth",
    "color": "https://huggingface.co/Bambii-03/scores/resolve/main/best_Color_model.pth",
    "composition": "https://huggingface.co/Bambii-03/scores/resolve/main/best_Layout_and_composition_model.pth",
    "texture": "https://huggingface.co/Bambii-03/scores/resolve/main/best_Details_and_texture_model.pth"
}

models_paths = {
    "total": "/tmp/best_Total_aesthetic_score_model.pth",
    "color": "/tmp/best_Color_model.pth",
    "composition": "/tmp/best_Layout_and_composition_model.pth",
    "texture": "/tmp/best_Details_and_texture_model.pth"
}

def load_model(model_url, clip_name="ViT-B/16"):
    models = {}
    for attr, url in model_url.items():
        model_path = models_paths[attr]
        if not os.path.exists(model_path):
            print(f"Downloading {attr} model...")
            urllib.request.urlretrieve(url, model_path)
        model = AesCLIP_reg(clip_name).to(device)
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        models[attr] = model
    return models

# Now, model is initialized outside of the function.
models = load_model(models_urls,"ViT-B/16")


# Image preprocessing
clip_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])
MEANS = {
    "color": 3.48,
    "total": 64.28,
    "texture": 6.06,
    "composition": 6.22
}

STDS = {
    "color": 3.69,
    "total": 15.24,
    "texture": 1.51,
    "composition": 1.33
}
def preprocess_image(image: Image.Image):
    return clip_preprocess(image).unsqueeze(0).to(device)

def predict_all(image_tensor):
    results = {}
    try:
        with torch.no_grad():
            for attr, model in models.items():
                score = model(image_tensor)  
                logger.info(f"Raw model output for {attr}: {score}")  # ✅ Debugging output
                
                if score.numel() == 1:  # Ensure it's a single value
                    norm_val = score.item()
                    raw = round(norm_val * STDS[attr] + MEANS[attr], 2)
                    results[attr] = raw
                else:
                    logger.error(f"Unexpected output shape for {attr}: {score.shape}")  # ✅ Debugging output
        logger.info(f"Prediction successful: {results}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise RuntimeError("Prediction error")
    return results



@app.get("/")
def home():
    return {"message": "Aesthetic Scoring API Running on Colab"}

@app.post("/predict-scores")
async def predict_image(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = preprocess_image(image)
        predictions = predict_all(image_tensor)
        return {"filename": file.filename, "predictions": predictions}
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    