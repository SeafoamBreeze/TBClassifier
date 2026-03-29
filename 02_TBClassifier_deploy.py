from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import torch
from utils.s3_utils import download_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_version
    try:
        model_path = await download_model()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.load(model_path, map_location=device, weights_only=True)
        model.eval().to(device)
        print(f"Model loaded on {device}")
    except Exception as e:
        print("Startup failed")
        raise
    yield
    del model

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    if model is None:
        raise HTTPException(503, "Model not loaded")
    return {"status": "ok", "version": model_version}