import base64
import io
from pathlib import Path
import mlflow
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from utils.s3_utils import download_model_from_s3
from torchvision import transforms
import uvicorn
from contextlib import asynccontextmanager
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

_model = None
_device = None
_gradcam = None

class PredictionResponse(BaseModel):
    filename: str
    prediction: str
    probability: float
    original_image: str
    gradcam_image: str

def load_model():

    global _model, _device, _gradcam
    
    if _model is not None:
        print("Model has already been loaded")
        return
    
    model_path = download_model_from_s3()
    print(f"Loading model from: {model_path}")
    
    model = mlflow.pytorch.load_model(model_path)
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(_device)
    model.eval()
    _model = model

    target_layer = get_target_layer(_model)
    gradcam = GradCAM(model=_model, target_layers=[target_layer])    
    _gradcam = gradcam

    print(f"Model loaded on device: {_device}")

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

    return tensor.to(_device), resize_and_normalize_image(image)   

def resize_and_normalize_image(image):
    return np.array(image.resize((512, 512))) / 255.0  

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

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(..., description="PNG image file")):

    # Validation
    try:
        print(f"Filename: {file.filename}")
        validate_file(file)
    except ValueError as e: 
        raise HTTPException(status_code=400, detail=f"Invalid file: {str(e)}")
    
    # Preprocess
    try:
        image_bytes = await file.read()
        input_tensor, image = preprocess_image(image_bytes)
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
    try:
        targets = [ClassifierOutputTarget(pred_idx)]
        grayscale_cam = _gradcam(input_tensor=input_tensor, targets=targets)[0, :]
        visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {str(e)}")

    return {
        "filename": file.filename,
        "prediction": get_class_name(pred_idx),
        "probability": round(probability, 2),
        "original_image": f"data:image/png;base64,{encode_image_to_base64(image)}",
        "gradcam_image": f"data:image/png;base64,{encode_image_to_base64(visualization)}"
    }

def get_target_layer(model):
    # for name, module in model.named_modules():
    #     print(f"{name}: {type(module).__name__}")
    return model.backbone.features.denseblock4.denselayer16.conv2

def validate_file(file: UploadFile):
    allowed_ext = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
    ext = Path(file.filename).suffix.lower()    
    if ext not in allowed_ext:
        raise ValueError(f"Invalid extension {ext}. Allowed: {allowed_ext}")
    
def encode_image_to_base64(image_array):
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)
    
    with io.BytesIO() as buffer:
        Image.fromarray(image_array).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

if __name__ == "__main__":  
    uvicorn.run(app, host="0.0.0.0", port=8000)

