import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import timm
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from typing import Tuple
import io
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import BlipForConditionalGeneration, BlipProcessor
import torch.nn as nn
import json
from aes_clip import AesCLIP_reg
import clip
import urllib.request
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from pydantic import BaseModel
from diffusers import DiffusionPipeline
import re
import base64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

app = FastAPI(title="ArtVision FastAPIs", description="Apis for art critique and creation tool")

@app.get("/")
async def health_check():
    return "FastAPI is running!"

def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ---------------------
# genres-styles
# ---------------------
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

def load_classes(filename):
    try:
        with open(filename, "r") as f:
            classes = json.load(f)
        return {int(k): v for k, v in classes.items()}
    except Exception as e:
        print(f"Error loading classes from {filename}: {e}")
        return {}
style_classes = load_classes(r"Critique-Models\genres-and-styles\style_classes.json")
genre_classes = load_classes(r"Critique-Models\genres-and-styles\genre_classes.json")

genres_styles_model = MultiTaskClassifier().to(device)
genres_styles_model_path = "Critique-Models\\genres-and-styles\\genre_style_multi_model.pth"

if os.path.exists(genres_styles_model_path):
    print(f"Loading model from: {genres_styles_model_path}")
    state_dict = torch.load(genres_styles_model_path, map_location=device)
    genres_styles_model.load_state_dict(state_dict, strict=False)
    genres_styles_model.eval()
else:
    raise FileNotFoundError(f"Model file not found at: {genres_styles_model_path}")


@app.post("/predict-style-genre/")
async def predict_genre_style(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        input_tensor = preprocess_image(image).to(device)
        with torch.no_grad():
            genre_logits, style_logits = genres_styles_model(input_tensor)
            genre = torch.argmax(genre_logits, dim=1).item()
            style = torch.argmax(style_logits, dim=1).item()
        return {
            "genre": genre_classes.get(genre, "Unknown"),
            "style": style_classes.get(style, "Unknown")
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------
# Scores
# ---------------------
scores_models_paths = {
    "total": r"Critique-Models\scores\best_Total_aesthetic_score_model.pth",
    "color": r"Critique-Models\scores\best_Color_model.pth",
    "composition": r"Critique-Models\scores\best_Layout_and_composition_model.pth",
    "texture": r"Critique-Models\scores\best_Details_and_texture_model.pth"
}
def load_scores_models(scores_models_paths, clip_name="ViT-B/16"):
    models = {}
    for attr, model_path in scores_models_paths.items():
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        score_model = AesCLIP_reg(clip_name).to(device)
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        score_model.load_state_dict(state_dict, strict=False)
        score_model.eval()
        models[attr] = score_model
    return models

scores_models = load_scores_models(scores_models_paths,"ViT-B/16")
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
def preprocess_image_for_score(image: Image.Image):
    return clip_preprocess(image).unsqueeze(0).to(device)

def predict_all(image_tensor):
    results = {}
    try:
        with torch.no_grad():
            for attr, model in scores_models.items():
                score = model(image_tensor)                  
                if score.numel() == 1:  # Ensure it's a single value
                    norm_val = score.item()
                    raw = round(norm_val * STDS[attr] + MEANS[attr], 2)
                    results[attr] = raw
    except Exception as e:
        raise RuntimeError("Prediction error")
    return results

@app.post("/predict-scores")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = preprocess_image_for_score(image)
        predictions = predict_all(image_tensor)
        return {"filename": file.filename, "predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
# ---------------------
# Description
# ---------------------
description_model_path = r"Critique-Models\description_generator\blip_model.pth"

def load_description_model(description_model_path):
    if not os.path.exists(description_model_path):
        raise FileNotFoundError(f"Model file not found at: {description_model_path}")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    state_dict = torch.load(description_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)  # Move model to device
    model.eval()
    return processor, model

processor, description_model = load_description_model(description_model_path)

@app.post("/generate_caption")
async def generate_caption(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = description_model.generate(**inputs, max_length=30)
        caption = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return JSONResponse({"caption": caption})

# ---------------------
# Watercolor Generation
# ---------------------
watercolor_weights_path = r"Creation-Models\watercolor\V_0.safetensors"

class PromptRequest(BaseModel): 
    prompt: str

watercolor_pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32,
    use_safetensors=True,
    cache_dir="/"
)
watercolor_pipe.load_lora_weights(watercolor_weights_path)
watercolor_pipe.to("cpu")
watercolor_pipe.enable_attention_slicing("max")
watercolor_pipe.enable_vae_slicing()
watercolor_pipe.enable_vae_tiling()
watercolor_pipe.safety_checker = None

@app.post("/generate_watercolor")
async def generate_watercolor_image(request: PromptRequest):
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            image = watercolor_pipe(request.prompt, height=352, width=352, num_inference_steps=40).images[0]
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {"image_base64": img_str}
    except Exception as e:
        print(f"Error: {e}")  # Log error to terminal
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# # ---------------------
# # Cartoon Generation
# # ---------------------
torch.cuda.empty_cache()
cartoon_pipe = DiffusionPipeline.from_pretrained(
    "stablediffusionapi/disney-pixar-cartoon",
    torch_dtype=torch.float32,
    cache_dir="/",
    safety_checker=None)
cartoon_pipe.to(device)
# cartoon_pipe.enable_model_cpu_offload()
cartoon_pipe.enable_attention_slicing()
cartoon_pipe.enable_vae_slicing()
cartoon_pipe.enable_vae_tiling()

import warnings

@app.post("/generate_cartoon")
async def generate_cartoon_image(request: PromptRequest):
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            image = cartoon_pipe(request.prompt, height=352, width=352, num_inference_steps=40).images[0]
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        torch.cuda.empty_cache()
        return {"image_base64": img_str}
    except Exception as e:
        print(f"Error: {e}")  # Log error to terminal
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# ---------------------
# Style Transfer
# ---------------------
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
styled_model = hub.load(hub_handle)

def load_image_into_tensor(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.expand_dims(img, axis=0)
    return img

@app.post("/stylize")
async def stylize(content_image: UploadFile = File(...), style_image: UploadFile = File(...)):
    content_bytes = await content_image.read()
    style_bytes = await style_image.read()
    content_tensor = load_image_into_tensor(content_bytes)
    style_tensor = load_image_into_tensor(style_bytes)
    outputs = styled_model(content_tensor, style_tensor)
    stylized_image = outputs[0]
    img_array = tf.squeeze(stylized_image).numpy()
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_array)
    buf = BytesIO()
    img_pil.save(buf, format='JPEG')
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")

# ---------------------
# Coloring Generation
# ---------------------

# ---------------------
# Sketch Generation
# ---------------------

#for running -> uvicorn app:app --reload
#pip install --upgrade accelerate
