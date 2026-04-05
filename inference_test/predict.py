import base64
import requests
from pathlib import Path

# ---------- CONFIG ----------
API_URL = "http://localhost:8000/predict"
IMAGE_PATH = "tb1117.png"
OUTPUT_HTML = "tb1117.html"

# Optional: patient metadata (as JSON string)
PATIENT_METADATA = None  # e.g., '{"age": 45, "gender": "M"}'

# ---------- READ AND ENCODE IMAGE ----------
with open(IMAGE_PATH, "rb") as f:
    image_bytes = f.read()
xray_image_base64 = base64.b64encode(image_bytes).decode("utf-8")

# ---------- SEND REQUEST ----------
data = {
    "filename": Path(IMAGE_PATH).name,
    "xray_image_base64": xray_image_base64,
}
if PATIENT_METADATA:
    data["patient_metadata"] = PATIENT_METADATA

response = requests.post(API_URL, data=data)
response.raise_for_status()  # Raise an error if request failed

result = response.json()
case_id = result["case_id"]
label = result["classification_label"]
confidence = result["confidence_score"]
grad_cam_base64 = result["grad_cam_image"]

# ---------- GENERATE HTML ----------
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>X-ray Prediction Result</title>
    <style>
        body {{ font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }}
        img {{ max-width: 512px; border: 1px solid #ccc; margin-top: 20px; }}
        .label {{ font-size: 24px; margin-top: 20px; }}
        .confidence {{ font-size: 18px; margin-top: 10px; color: #555; }}
    </style>
</head>
<body>
    <h1>Prediction Result</h1>
    <div class="label">Label: {label}</div>
    <div class="confidence">Confidence: {confidence:.2f}</div>
    <img src="data:image/png;base64,{grad_cam_base64}" alt="Grad-CAM">
</body>
</html>
"""

with open(OUTPUT_HTML, "w") as f:
    f.write(html_content)

print(f"HTML result saved to {OUTPUT_HTML}")