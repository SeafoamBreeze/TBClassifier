import base64
import io
from pathlib import Path
from typing import List, Optional
import mlflow
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils.s3_utils import download_model_from_s3
from torchvision import transforms
import uvicorn
from contextlib import asynccontextmanager
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from utils.adversarial_attack_utils import fgsm_attack, apply_mitigation, tensor_to_numpy
import torch.nn.functional as F
import threading
import magic
import uuid

_model_lock = threading.Lock()
_model = None
_device = None
_gradcam = None

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit
ALLOWED_MIME_TYPES = {'image/png', 'image/jpeg', 'image/gif', 'image/bmp', 'image/webp'}

class MitigationResult(BaseModel):
    method: str
    prediction: str
    probability: float
    heatmap: str
    confidence_restored_pct: float

class VariantResult(BaseModel):
    epsilon: float
    prediction: str
    probability: float
    l2_distance: float
    is_attack_successful: bool
    image: str
    heatmap: str
    mitigation: Optional[MitigationResult]

class OriginalResult(BaseModel):
    prediction: str
    probability: float
    image: str
    heatmap: str
    ground_truth: Optional[str] = None

class PredictionResponse(BaseModel):
    filename: str
    original: OriginalResult
    variants: List[VariantResult]

class OutputResponse(BaseModel):
    case_id: str
    classification_label: str
    confidence_score: float
    grad_cam_image: str

def load_model():

    global _model, _device, _gradcam
    
    with _model_lock:
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

    raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((512, 512)) 
    to_tensor = transforms.ToTensor()
    raw_image_tensor = to_tensor(raw_image).unsqueeze(0)                    # (1, 3, 512, 512) range [0, 1]
    standardization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    standardized_tensor = standardization(raw_image_tensor.clone())
    normalized_np = np.array(raw_image) / 255.0                              # (512, 512, 3) for Grad-CAM
    
    return standardized_tensor.to(_device), normalized_np, raw_image_tensor.to(_device)

def get_class_name(prediction_idx):
    classes = ["Healthy", "Others (Not Tuberculosis)", "Tuberculosis"]
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

def run_mitigation_prediction(
    mitigated_tensor: torch.Tensor
) -> torch.Tensor:
    global _model, _gradcam, _device
    
    with torch.no_grad():
        mit_outputs = _model(mitigated_tensor)
        mit_probs = torch.softmax(mit_outputs, dim=1)

    return mit_probs

def run_mitigation_analysis(
    mitigated_tensor: torch.Tensor,
    original_pred_idx: int,
    adv_probability: float,
    orig_probability: float,
    method: str,
    mit_probs: torch.Tensor
) -> MitigationResult:

    mit_prob_values = mit_probs.cpu().numpy()[0]
    mit_pred_idx = int(np.argmax(mit_prob_values))
    mit_probability = float(mit_prob_values[mit_pred_idx])
    
    mit_targets = [ClassifierOutputTarget(mit_pred_idx)]
    mit_cam = _gradcam(input_tensor=mitigated_tensor, targets=mit_targets)[0, :]
    
    mit_np = tensor_to_numpy(mitigated_tensor)
    mit_heatmap = show_cam_on_image(mit_np, mit_cam, use_rgb=True)
    
    if orig_probability != adv_probability:
        gap = orig_probability - adv_probability
        if abs(gap) > 1e-6:
            recovered = mit_probability - adv_probability
            restored_pct = (recovered / gap) * 100
        else:
            restored_pct = 0.0
    else:
        restored_pct = 0.0
    
    restored_pct = max(-100.0, min(100.0, restored_pct))
    
    return MitigationResult(
        method=method,
        prediction=get_class_name(mit_pred_idx),
        probability=round(mit_probability, 4),
        heatmap=encode_image_to_base64(mit_heatmap),
        confidence_restored_pct=round(restored_pct, 2)
    )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "device": str(_device) if _device else None
    }

async def validate_file_secure(file: UploadFile) -> bytes:

    # Check file size first
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise ValueError(f"File too large: {len(contents)} bytes. Max: {MAX_FILE_SIZE}")
    
    # Check MIME type using magic numbers
    mime = magic.from_buffer(contents, mime=True)
    if mime not in ALLOWED_MIME_TYPES:
        raise ValueError(f"Invalid file type: {mime}. Allowed: {ALLOWED_MIME_TYPES}")
    
    # Also check extension as secondary validation
    allowed_ext = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
    ext = Path(file.filename).suffix.lower()    
    if ext not in allowed_ext:
        raise ValueError(f"Invalid extension {ext}. Allowed: {allowed_ext}")
    
    return contents

@app.post("/predict-adverserial", response_model=PredictionResponse)
async def predict(
    mitigation_method: str = Form(...), 
    file: UploadFile = File(..., description="X-ray image to analyze (will be attacked)"),
    ground_truth: Optional[UploadFile] = File(None, description="Optional ground truth image for comparison")
):

    # Validation
    try:
        print(f"Filename: {file.filename}")
        image_bytes = await validate_file_secure(file)
        
        gt_bytes = None
        if ground_truth:
            print(f"Ground truth: {ground_truth.filename}")
            gt_bytes = await validate_file_secure(ground_truth)
    except ValueError as e: 
        raise HTTPException(status_code=400, detail=f"Invalid file: {str(e)}")
    
    # Preprocess
    try:
        standardized_tensor, normalized_np, raw_image_tensor = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")
    
    # Process ground truth if provided
    ground_truth_base64 = None
    if gt_bytes:
        try:
            gt_image = Image.open(io.BytesIO(gt_bytes)).convert("RGB")
            gt_image = gt_image.resize((512, 512))
            gt_np = np.array(gt_image) / 255.0
            ground_truth_base64 = encode_image_to_base64(gt_np)
        except Exception as e:
            print(f"Ground truth processing failed: {e}")
            
    # Store original for comparison
    standardized_tensor2 = standardized_tensor.clone().detach()

    # Configuration
    epsilon_values = [0.0, 0.001, 0.01, 0.1]
    original_pred_idx = None
    original_probability = None
    results = []
    original_heatmap = None

    # Analyze each epsilon
    for epsilon in epsilon_values:
        try:
            if epsilon == 0.0:
                
                print("No adversary attack")
                perturbed_preprocessed  = standardized_tensor2.clone()
                perturbed_unprocessed = raw_image_tensor.clone()

                with torch.no_grad():
                    outputs = _model(perturbed_preprocessed)
                    probs = torch.softmax(outputs, dim=1)
                
                prob_values = probs.cpu().numpy()[0]
                pred_idx = int(np.argmax(prob_values))
                probability = float(prob_values[pred_idx])
                
                original_pred_idx = pred_idx
                original_probability = probability
                
                vis_np = normalized_np
                l2_dist = 0.0
                is_attack_successful = False      

            else:
                
                print(f"FGSM attack with epsilon={epsilon}")

                attack_tensor = raw_image_tensor.clone().requires_grad_(True)  

                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(_device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(_device)
                normalized_attack = (attack_tensor - mean) / std

                outputs = _model(normalized_attack)
                loss = F.cross_entropy(outputs, torch.tensor([original_pred_idx], device=_device))
                
                _model.zero_grad()
                loss.backward()
                data_grad = attack_tensor.grad.data
                
                perturbed_unprocessed = fgsm_attack(attack_tensor, epsilon, data_grad)
                perturbed_preprocessed = (perturbed_unprocessed - mean) / std

                l2_dist = torch.norm(perturbed_preprocessed - standardized_tensor2).item()
                
                with torch.no_grad():
                    outputs = _model(perturbed_preprocessed)
                    probs = torch.softmax(outputs, dim=1)
                
                prob_values = probs.cpu().numpy()[0]
                pred_idx = int(np.argmax(prob_values))
                probability = float(prob_values[pred_idx])
                
                is_attack_successful = (pred_idx != original_pred_idx)
                vis_np = perturbed_unprocessed.squeeze(0).permute(1, 2, 0).cpu().numpy()
                vis_np = np.clip(vis_np, 0, 1).astype(np.float32)
                
            # Grad-CAM
            try:
                print(f"Generating Grad-CAM for epsilon={epsilon}")
                targets = [ClassifierOutputTarget(pred_idx)]
                grayscale_cam = _gradcam(input_tensor=perturbed_preprocessed , targets=targets)[0, :]
                heatmap = show_cam_on_image(vis_np, grayscale_cam, use_rgb=True)
            except Exception as e:
                print(f"Grad-CAM failed for epsilon={epsilon}: {e}")
                heatmap = (vis_np * 255).astype(np.uint8)
            
            if epsilon == 0.0:
                original_heatmap = heatmap.copy()
            
            # Mitigation (skip for epsilon=0)
            mitigation_result = None
            if epsilon > 0:
                print(f"Attempting {mitigation_method} for epsilon={epsilon}")
                try:
                    if mitigation_method == "gaussian_noise":
                        num_samples = 200
                        mitigation_prediction = torch.zeros(1, 3, device=_device)
                        for _ in range(num_samples):
                            mitigated_tensor = apply_mitigation(perturbed_preprocessed, mitigation_method)    
                            mitigation_prediction += run_mitigation_prediction(mitigated_tensor)
                        pred_avg = mitigation_prediction/num_samples
                        print(f"Average prediction after {mitigation_method} is {pred_avg}")
                        mitigation_result = run_mitigation_analysis(
                            mitigated_tensor,
                            original_pred_idx,
                            probability,
                            original_probability,
                            mitigation_method,
                            pred_avg
                        )                             
                    else:
                        mitigated_tensor = apply_mitigation(perturbed_preprocessed, mitigation_method)
                        mitigation_prediction = run_mitigation_prediction(mitigated_tensor)
                        print(f"Prediction after {mitigation_method} is {mitigation_prediction}")
                        mitigation_result = run_mitigation_analysis(
                            mitigated_tensor,
                            original_pred_idx,
                            probability,
                            original_probability,
                            mitigation_method,
                            mitigation_prediction
                        )                            
                except Exception as e:
                    print(f"Mitigation failed for epsilon={epsilon}: {e}")
                    mitigation_result = MitigationResult(
                        method=mitigation_method,
                        prediction="ERROR",
                        probability=0.0,
                        heatmap="",
                        confidence_restored_pct=0.0
                    )

            variant = VariantResult(
                epsilon=epsilon,
                prediction=get_class_name(pred_idx),
                probability=round(probability, 4),
                l2_distance=round(l2_dist, 6),
                is_attack_successful=is_attack_successful,
                image=encode_image_to_base64(vis_np),
                heatmap=encode_image_to_base64(heatmap),
                mitigation=mitigation_result
            )
            results.append(variant)
            
        except Exception as e:
            print(f"Failed to process epsilon={epsilon}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Validate results
    if not results:
        raise HTTPException(status_code=500, detail="All adversarial analyses failed")
    
    if original_heatmap is None:
        raise HTTPException(status_code=500, detail="Original prediction (epsilon=0) failed - no heatmap generated")
    
    # Build original result with ground truth
    original_result = OriginalResult(
        prediction=get_class_name(original_pred_idx),
        probability=round(original_probability, 4),
        image=encode_image_to_base64(normalized_np),
        heatmap=encode_image_to_base64(original_heatmap),
        ground_truth=ground_truth_base64
    )  

    variants = [r for r in results if r.epsilon > 0]
    
    return PredictionResponse(
        filename=file.filename,
        original=original_result,
        variants=variants
    )   

@app.post("/predict", response_model=OutputResponse)
async def predict(
    xray_image_base64: str = Form(..., description="X-ray image in base64"),
    patient_metadata: Optional[str] = Form(None, description="Optional JSON string of patient metadata")
):
    # Generate a unique case ID
    unique_id = str(uuid.uuid4())
    print(f"Generated case_id: {unique_id}")

    # Decode the base64 image and preprocess
    try:
        image_bytes = base64.b64decode(xray_image_base64)
        standardized_tensor, normalized_np, raw_image_tensor = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")

    # Clone tensor for perturbation/prediction
    perturbed_preprocessed = standardized_tensor.clone()

    # Make prediction
    with torch.no_grad():
        outputs = _model(perturbed_preprocessed)
        probs = torch.softmax(outputs, dim=1)

    prob_values = probs.cpu().numpy()[0]
    pred_idx = int(np.argmax(prob_values))
    probability = float(prob_values[pred_idx])

    # Generate Grad-CAM
    try:
        targets = [ClassifierOutputTarget(pred_idx)]
        grayscale_cam = _gradcam(input_tensor=perturbed_preprocessed, targets=targets)[0, :]
        heatmap = show_cam_on_image(normalized_np, grayscale_cam, use_rgb=True)
    except Exception as e:
        print(f"Grad-CAM failed: {e}")
        heatmap = (normalized_np * 255).astype(np.uint8)

    # Convert heatmap to base64
    heatmap_pil = Image.fromarray(heatmap)
    buffered = io.BytesIO()
    heatmap_pil.save(buffered, format="PNG")
    grad_cam_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    classification_label = get_class_name(pred_idx)

    response = OutputResponse(
        case_id=unique_id,
        classification_label=classification_label,
        confidence_score=probability,
        grad_cam_image=grad_cam_base64
    )

    return response

def get_target_layer(model):
    # for name, module in model.named_modules():
    #     print(f"{name}: {type(module).__name__}")
    return model.backbone.features.denseblock4.denselayer16.conv2


def encode_image_to_base64(image_input):

    if isinstance(image_input, np.ndarray):
        if image_input.dtype != np.uint8:
            image_input = (np.clip(image_input, 0, 1) * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_input)
    elif isinstance(image_input, Image.Image):
        pil_image = image_input
    else:
        raise TypeError(f"Expected numpy array or PIL Image, got {type(image_input)}")
    
    with io.BytesIO() as buffer:
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()


if __name__ == "__main__":  
    uvicorn.run(app, host="0.0.0.0", port=8000)

