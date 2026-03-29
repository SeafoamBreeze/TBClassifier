import io
from typing import Optional

import boto3
import mlflow
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Global model cache
_model = None
_device = None

class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    class_name: str
    probability: float
    confidence: str  # high/medium/low based on threshold





def load_model():
    """
    Load model from S3 (download if needed).
    Uses global cache to avoid re-downloading.
    """
    global _model, _device
    
    if _model is not None:
        return _model, _device
    
    # Download from S3
    model_path = download_model_from_s3()
    
    # Load with MLflow
    print(f"Loading model from: {model_path}")
    
    try:
        # Try PyTorch format first
        model = mlflow.pytorch.load_model(model_path)
    except Exception as e:
        print(f"PyTorch load failed, trying pyfunc: {e}")
        # Fallback to pyfunc
        pyfunc_model = mlflow.pyfunc.load_model(model_path)
        # Extract PyTorch model if wrapped
        model = pyfunc_model._model_impl.pytorch_model
    
    # Setup device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(_device)
    model.eval()
    
    _model = model
    print(f"Model loaded on device: {_device}")
    
    return _model, _device


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Preprocess PNG image for model inference.
    Adjust transforms based on your training preprocessing.
    """
    from torchvision import transforms
    
    # Standard ImageNet transforms - adjust to match your training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Apply transforms and add batch dimension
    tensor = transform(image).unsqueeze(0)
    
    return tensor


def get_class_name(prediction_idx: int) -> str:
    """Map prediction index to class name."""
    classes = ["Healthy", "SickNonTB", "TB"]
    return classes[prediction_idx]


def get_confidence_level(probability: float) -> str:
    """Determine confidence level based on probability."""
    if probability >= 0.9:
        return "high"
    elif probability >= 0.7:
        return "medium"
    return "low"


# FastAPI app
app = FastAPI(
    title="TB Classifier Inference API",
    description="API for tuberculosis classification from chest X-ray images",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    print("Starting up - loading model...")
    load_model()
    print("Startup complete - model ready")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "device": str(_device) if _device else None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(..., description="PNG image file")):
    """
    Predict tuberculosis probability from chest X-ray image.
    
    - **file**: PNG image file (chest X-ray)
    - Returns: class name, probability (0-1), and confidence level
    """
    # Validate file type
    if not file.content_type or "image" not in file.content_type:
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read image bytes
    image_bytes = await file.read()
    
    # Validate it's a valid image
    try:
        Image.open(io.BytesIO(image_bytes)).verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Load model if not already loaded
    model, device = load_model()
    
    # Preprocess
    try:
        input_tensor = preprocess_image(image_bytes).to(device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")
    
    # Inference
    try:
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get prediction
            prob_values = probabilities.cpu().numpy()[0]
            pred_idx = int(np.argmax(prob_values))
            confidence = float(prob_values[pred_idx])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    
    # Build response
    response = PredictionResponse(
        class_name=get_class_name(pred_idx),
        probability=round(confidence, 4),
        confidence=get_confidence_level(confidence)
    )
    
    return response


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Batch prediction for multiple images.
    """
    results = []
    
    for file in files:
        try:
            image_bytes = await file.read()
            result = await predict(file)  # Reuse single predict logic
            results.append({
                "filename": file.filename,
                "prediction": result.dict(),
                "success": True
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "success": False
            })
    
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)