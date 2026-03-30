import base64
import io
from pathlib import Path
from typing import Optional

import boto3
import mlflow
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from utils.s3_utils import download_model_from_s3
from torchvision import transforms
import uvicorn
from contextlib import asynccontextmanager
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

_model = None
_device = None

def load_model():

    global _model, _device
    
    if _model is not None:
        print("Model has already been loaded")
        return _model, _device
    
    model_path = download_model_from_s3()
    print(f"Loading model from: {model_path}")
    
    model = mlflow.pytorch.load_model(model_path)

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(_device)
    model.eval()
    
    _model = model
    print(f"Model loaded on device: {_device}")
    
    return _model, _device

def preprocess_image(image_bytes):
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    
    return tensor, np.array(image.resize((512, 512))) / 255.0

def get_class_name(prediction_idx):
    classes = ["Healthy", "SickNonTB", "TB"]
    return classes[prediction_idx]

@asynccontextmanager
async def lifespan(app):
    print("Starting up - loading model...")
    load_model()
    print("Startup complete - model ready")
    yield  
    print("Shutting down - cleaning up...")

app = FastAPI(
    title="TB Classifier Inference API",
    description="API for tuberculosis classification from chest X-ray images",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "device": str(_device) if _device else None
    }

@app.post("/predict")
async def predict(file = File(..., description="PNG image file")):

    print(f"Filename: {file.filename}")

    allowed_ext = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
    ext = Path(file.filename).suffix.lower()
    
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"Invalid extension {ext}. Allowed: {allowed_ext}")
    
    # Read image bytes
    image_bytes = await file.read()
    
    # Preprocess
    try:
        input_tensor, image = preprocess_image(image_bytes)
        input_tensor = input_tensor.to(_device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")
    
    # Inference
    try:
        with torch.no_grad():
            outputs = _model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        prob_values = probabilities.cpu().numpy()[0]
        pred_idx = int(np.argmax(prob_values))
        probability = float(prob_values[pred_idx])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # Generate Grad-CAM
    target_layer = get_target_layer(_model) 
    cam = GradCAM(model=_model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(pred_idx)]
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    visualization = show_cam_on_image(
        image, 
        grayscale_cam, 
        use_rgb=True,
        colormap=cv2.COLORMAP_JET
    )
    
    # Convert to base64 for JSON response
    buffer = io.BytesIO()
    Image.fromarray(visualization).save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    original_uint8 = (image * 255).astype("uint8")
    original_pil = Image.fromarray(original_uint8)

    buffer_orig = io.BytesIO()
    original_pil.save(buffer_orig, format="PNG")
    original_base64 = base64.b64encode(buffer_orig.getvalue()).decode()

    return {
        "filename": file.filename,
        "prediction": get_class_name(pred_idx),
        "probability": round(probability, 2),
        "original_image": f"data:image/png;base64,{original_base64}",
        "gradcam_image": f"data:image/png;base64,{img_str}"
    }
    

def get_target_layer(model):
    for name, module in model.named_modules():
        print(f"{name}: {type(module).__name__}")
    return model.backbone.features.denseblock4.denselayer16.conv2


if __name__ == "__main__":  
    uvicorn.run(app, host="0.0.0.0", port=8000)

