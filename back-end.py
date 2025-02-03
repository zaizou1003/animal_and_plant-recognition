from fastapi import FastAPI, File, UploadFile,Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import json
import io
import torch
from torchvision import transforms
from db_utils import get_species_by_id,search_species_by_name


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development (replace "*" with your frontend's URL in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers (e.g., Content-Type, Authorization)
)

# Load the class map from JSON
with open("classes_map.json", "r") as f:
    class_map = json.load(f)

# Reverse mapping (class_id -> class_name)
id_to_class = {v: k for k, v in class_map.items()}

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your trained model
model = torch.load("full_model.pth")  # Adjust path and loading method
model = model.to(device)
model.eval()


# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust according to your model's input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
@app.get("/search/")
async def search_species(name: str = Query(..., min_length=1)):
    """
    Search for species (animals or plants) by name.
    """
    try:
        # Call the search function
        search_results = search_species_by_name(name)
        return JSONResponse(content=search_results)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Load image
        image = Image.open(io.BytesIO(await file.read()))
        image = preprocess(image).unsqueeze(0)  # Add batch dimension
        image = image.to(device)
        

        # Get model prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted_class = outputs.max(1)

        # Get class ID and name
        class_id = predicted_class.item()
        class_name = id_to_class.get(class_id, "Unknown Class")

        # Fetch animal details from the database
        species_info  = get_species_by_id(class_id)

        # Log the predicted output
        response = {
            "filename": file.filename,
            "predicted_class": int(predicted_class.item()),
            "class_name": class_name,
            "species_info": species_info 
        }

        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
