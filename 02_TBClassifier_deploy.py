from fastapi import FastAPI
import torch
from utils.s3_utils import download_production_model

app = FastAPI()

download_production_model()
model = torch.load("model.pth", map_location="cpu")
model.eval()

@app.get("/health")
def health():
    return {"status": "ok"}