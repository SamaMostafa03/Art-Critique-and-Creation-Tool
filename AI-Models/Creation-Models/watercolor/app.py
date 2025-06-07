import os
# Set writable cache directories for Hugging Face Spaces
os.environ["HF_HOME"] = "/tmp"
os.environ["TRANSFORMERS_CACHE"] = "/tmp"
os.environ["HF_HUB_CACHE"] = "/tmp"
os.environ["TORCH_HOME"] = "/tmp"

import torch
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download

# Define input request structure
class PromptRequest(BaseModel):
    prompt: str

# Download LoRA weights
def load_model():
    weights_path = hf_hub_download(
        repo_id="Bambii-03/watercolor-generation",
        filename="V_0.safetensors",
        cache_dir="/tmp"
    )
    return weights_path

# Initialize model
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32,  # Use float32 for CPU
    use_safetensors=True
)
weights_path = load_model()
pipe.load_lora_weights(weights_path)  # Adjust if .safetensors requires special handling
pipe.enable_attention_slicing("max")
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
pipe.safety_checker = None  # Optional: disable safety checker for faster generation


# Initialize FastAPI app
app = FastAPI(
    title="ArtVision Watercolor Art Style Generator API",
    description="Generates watercolor artwork given a text prompt."
)

@app.post("/generate_watercolor")
async def generate_watercolor_image(request: PromptRequest):
    image = pipe(request.prompt, height=352, width=352, num_inference_steps=30).images[0]
    # Safe filename
    safe_prompt = re.sub(r'[^a-zA-Z0-9_\-]', '_', request.prompt)[:100]
    output_path = os.path.join("/tmp", f"{safe_prompt}.jpeg")
    image.save(output_path)
    # Stream image back
    buf = BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")