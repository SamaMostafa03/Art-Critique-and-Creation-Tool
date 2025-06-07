import os
# Set writable cache directories for Hugging Face Spaces
os.environ["HF_HOME"] = "/tmp"
os.environ["TRANSFORMERS_CACHE"] = "/tmp"
os.environ["HF_HUB_CACHE"] = "/tmp"
os.environ["TORCH_HOME"] = "/tmp"

import re
import torch
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from uuid import uuid4
from io import BytesIO
from PIL import Image
from diffusers import DiffusionPipeline

# Define input request structure
class PromptRequest(BaseModel):
    prompt: str

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the diffusion model
pipe = DiffusionPipeline.from_pretrained("stablediffusionapi/disney-pixar-cartoon",torch_dtype=torch.float32)
pipe.enable_attention_slicing("max")
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
pipe.safety_checker = None  # Optional: disable safety checker for faster generation


# Initialize FastAPI app
app = FastAPI(
    title="ArtVision Cartoon Art Style Generator API",
    description="Generates cartoon artwork given a text prompt."
)

# Background task to generate and save the image
def generate_and_save_image(prompt: str, output_path: str):
    image = pipe(prompt, height=352, width=352, num_inference_steps=30).images[0]
    image.save(output_path)

@app.post("/generate_cartoon")
async def generate_cartoon_image(request: PromptRequest, background_tasks: BackgroundTasks):
    # Create a unique task ID
    task_id = str(uuid4())
    safe_prompt = re.sub(r'[^a-zA-Z0-9_\-]', '_', request.prompt)[:100]
    output_path = os.path.join("/tmp", f"{task_id}_{safe_prompt}.jpeg")
    # Launch background task
    background_tasks.add_task(generate_and_save_image, request.prompt, output_path)

    return {"task_id": task_id}

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    # Search for any file with this task_id prefix in /tmp
    matching_files = [f for f in os.listdir("/tmp") if f.startswith(task_id)]
    if not matching_files:
        raise HTTPException(status_code=404, detail="Image not ready yet. Try again shortly.")
    
    image_path = os.path.join("/tmp", matching_files[0])
    return FileResponse(image_path, media_type="image/jpeg")