import requests
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt

import requests


def show_base64_image(base64_data, title):
    img_bytes = base64.b64decode(base64_data.split(",")[1])
    img = Image.open(io.BytesIO(img_bytes))

    plt.figure()
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)

url = "http://localhost:8000/predict"
with open("chest_xray.png", "rb") as f:
    response = requests.post(url, files={"file": f})
    
result = response.json()
print(result["filename"])
print(result["prediction"])
print(result["probability"])

show_base64_image(result["original_image"], "Original")
show_base64_image(result["gradcam_image"], "GradCAM")

plt.show()