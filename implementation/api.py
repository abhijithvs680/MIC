from fastapi import FastAPI, File, UploadFile
from io import BytesIO
import torch
from torchvision import transforms
from PIL import Image

# Define your device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = torch.load("implementation/model/trained_model.pth", map_location=device)
model.eval()

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    # Load image from bytes
    image = Image.open(BytesIO(contents)).convert('RGB')
    # Apply transformations
    data_transforms = {
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }
    image_tensor = data_transforms['val'](image).unsqueeze(0).to(device)
    # Disable gradient computation
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
    # Get predicted class
    class_names = ['class1', 'class2', 'class3']  # Define your class names accordingly
    predicted_class = class_names[preds[0]]
    return {"predicted_class": predicted_class}
