# Art-Vision: Art Critique and Creation Tool
Art-Vision is an AI-powered tool designed to offer two core functionalities:
- `Art Creation` - Generates original artwork based on user-defined description, styles, or parameters, providing creative inspiration art pieces.
- `Art Critique` -  Analyzes an artist's work and provides a critique of various elements, such as color palettes, composition, texture, and style, which can help artists improve or refine their works.

For a detailed explanation of the project, please refer to the full documentation: 📌 [Project Documentation (PDF)](./Art-Vision-documentation.pdf)

## AI Models  

### 1. Multi-Class Classification Model  
- **Backbone Model**: Convnext (fine-tuned with transfer learning)
- **Dataset**: WikiArt  
- **Task**: Predicts both **art style** and **genre** from an artwork image  

### 2. Regression Model  
- **Backbone Model**: CLIP+MLP (fine-tuned with transfer learning)
- **Dataset**: APDDv2  
- **Task**: Predicts **color harmony**, **Texture**, **composition** and aesthitic scores from an artwork image  

### 3. Image-to-Text Generation Model  
- **Backbone Model**: BLIP-Base (fine-tuned with transfer learning)
- **Dataset**: WikiArt  
- **Task**: Generates **descriptive caption** of an artwork image  

### 4. Text-to-Image Generation Model  
- **Backbone Model**: Stable Diffusion-XL (fine-tuned with Lora Enabled)
- **Dataset**: Custom dataset  
- **Task**: Generates images in **watercolor** style based on text prompt  

## 📄 Postman API Documentation  
👉 [Fast API Documentation](https://documenter.getpostman.com/view/34266999/2sAYX6oM9L)  

## Running the FastAPI Backend  

```sh
uvicorn app:app --reload
