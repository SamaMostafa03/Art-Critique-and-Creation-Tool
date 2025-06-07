import os
# Set writable cache directory
os.environ["HF_HOME"] = "/tmp"
os.environ["TRANSFORMERS_CACHE"] = "/tmp"
os.environ["HF_HUB_CACHE"] = "/tmp"
os.environ["TORCH_HOME"] = "/tmp"

import timm  
import torch 
from transformers import BlipForConditionalGeneration, BlipProcessor
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import requests
import urllib.request

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="ArtVision Art Desciption Generator API", description="Generates art description for artworks.")

# Download before loading
model_url = {
    "caption": "https://huggingface.co/Toty22/ArtCaption/resolve/main/blip_model.pth"
}
model_path = "/tmp/blip_model.pth"

def load_model(model_url):
    for attr, url in model_url.items():
        if not os.path.exists(model_path):
            print(f"Downloading {attr} model...")
            urllib.request.urlretrieve(url, model_path)  # Corrected URL usage here
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
    return processor, model

# Now, model is initialized outside of the function.
processor, model  = load_model(model_url)

@app.post("/generate_caption")
async def generate_caption(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_length=30)
        caption = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return JSONResponse({"caption": caption})
#uvicorn app:app --reload