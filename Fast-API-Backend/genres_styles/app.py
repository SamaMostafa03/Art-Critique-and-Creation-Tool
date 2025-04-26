import os
# Set writable cache directory
os.environ["HF_HOME"] = "/tmp"
os.environ["TRANSFORMERS_CACHE"] = "/tmp"
os.environ["HF_HUB_CACHE"] = "/tmp"
os.environ["TORCH_HOME"] = "/tmp"

import timm  
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import Tuple
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import json
import numpy as np
import requests
import urllib.request

class MultiTaskClassifier(nn.Module):
    def __init__(self, model_name="convnext_base", num_genres=10, num_styles=27, drop_rate=0.1, drop_path_rate=0.1):
        super(MultiTaskClassifier, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="avg")
        in_features = self.backbone.num_features
        self.genre_head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(in_features, num_genres)
        )
        self.style_head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(in_features, num_styles)
        )

    def forward(self, x):
        features = self.backbone(x)
        genre_logits = self.genre_head(features)
        style_logits = self.style_head(features)
        return genre_logits, style_logits

# Initialize and load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize the model object before using it
model = MultiTaskClassifier().to(device)

model_url = {
    "genres_styles": "https://huggingface.co/Bambii-03/wikiart-genre-style-model/resolve/main/genre_style_model_weights.pth"
}
model_path = "/tmp/genre_style_model_weights.pth"

def load_model(model_url):
    for attr, url in model_url.items():
        if not os.path.exists(model_path):
            print(f"Downloading {attr} model...")
            urllib.request.urlretrieve(url, model_path)  # Corrected URL usage here
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    return model

# Now, model is initialized outside of the function.
model = load_model(model_url)

# Image preprocessing
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Converts to [0,1] and CHW format
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])  # Imagenet normalization
    ])
    return transform(image).unsqueeze(0)  # Add batch dim

# Load class labels
def load_classes(filename):
    try:
        with open(filename, "r") as f:
            classes = json.load(f)
        return {int(k): v for k, v in classes.items()}
    except Exception as e:
        print(f"Error loading classes from {filename}: {e}")
        return {}

style_classes = load_classes("style_classes.json")
genre_classes = load_classes("genre_classes.json")

app = FastAPI(title="WikiArt Classification API", description="Predicts the style and genre of an artwork.")

@app.get("/")
async def health_check():
    return "FastAPI is running!"

@app.post("/predict-style-genre/")
async def predict_genre_style(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        input_tensor = preprocess_image(image).to(device)
        with torch.no_grad():
            genre_logits, style_logits = model(input_tensor)
            genre = torch.argmax(genre_logits, dim=1).item()
            style = torch.argmax(style_logits, dim=1).item()
        return {
            "genre": genre_classes.get(genre, "Unknown"),
            "style": style_classes.get(style, "Unknown")
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
