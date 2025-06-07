from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
# Set writable cache directories for Hugging Face Spaces
os.environ["HF_HOME"] = "/tmp"
os.environ["TRANSFORMERS_CACHE"] = "/tmp"
os.environ["HF_HUB_CACHE"] = "/tmp"
os.environ["TORCH_HOME"] = "/tmp"

# Load TensorFlow Hub model
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
model = hub.load(hub_handle)

# Helper function
def load_image_into_tensor(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.expand_dims(img, axis=0)
    return img
# Initialize FastAPI app
app = FastAPI(
    title="ArtVision Art Style Transfer API",
    description="Transfer styles between two images"
)
# Define endpoint
@app.post("/stylize")
async def stylize(content_image: UploadFile = File(...), style_image: UploadFile = File(...)):
    content_bytes = await content_image.read()
    style_bytes = await style_image.read()
    content_tensor = load_image_into_tensor(content_bytes)
    style_tensor = load_image_into_tensor(style_bytes)
    outputs = model(content_tensor, style_tensor)
    stylized_image = outputs[0]
    img_array = tf.squeeze(stylized_image).numpy()
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_array)
    buf = BytesIO()
    img_pil.save(buf, format='JPEG')
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")

    #uvicorn app:app --reload