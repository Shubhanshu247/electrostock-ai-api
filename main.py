
import json
from pathlib import Path
from io import BytesIO

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mobilenetv2_ewaste.pth"
CLASSES_PATH = BASE_DIR / "class_names.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))

state_dict = torch.load(
    MODEL_PATH, map_location=DEVICE, weights_only=True
)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

app = FastAPI(
    title="ElectroStock AI E-Waste Classifier",
    version="1.0.0",
    description="Classifies electronic-device images into trained categories.",
)


@app.get("/")
def home():
    return {
        "message": "ElectroStock AI API is running",
        "device": str(DEVICE),
        "classes": CLASS_NAMES,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "model_loaded": True,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {
        "image/jpeg", "image/png", "image/webp", "image/bmp"
    }:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, WEBP, or BMP image.",
        )

    try:
        image_bytes = await file.read()

        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image must be 10 MB or smaller.",
            )

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence, index = torch.max(probabilities, dim=0)

        return {
            "predicted_class": CLASS_NAMES[index.item()],
            "confidence": round(confidence.item() * 100, 2),
            "device": str(DEVICE),
        }

    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )
