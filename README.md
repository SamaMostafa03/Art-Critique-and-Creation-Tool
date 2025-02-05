# Art-Vision: Art Critique and Creation Tool
Art-Vision is an AI-powered tool designed to offer two core functionalities:
- `Art Creation` - Generates original artwork based on user-defined description, styles, or parameters, providing creative inspiration art pieces.
- `Art Critique` -  Analyzes an artist's work and provides a critique of various elements, such as color palettes, composition, texture, and style, which can help artists improve or refine their works.

For a detailed explanation of the project, please refer to the full documentation: 📌 [Project Documentation (PDF)](./Art-Vision-documentation.pdf)

## AI Models  

### 1. Multi-Class Classification Model  
- **Backbone Model**: Densenet-121 (fine-tuned with transfer learning)
- **Dataset**: WikiArt  
- **Task**: Predicts both **art style** and **genre** from an artwork image  


## 📄 Postman API Documentation  
👉 [Fast API Documentation](https://documenter.getpostman.com/view/34266999/2sAYX6oM9L)  

## Running the FastAPI Backend  

```sh
uvicorn genre-style-prediction-api:app --reload
