import requests

with open("chest_xray.png", "rb") as f:
    response = requests.post(
        "http://your-ecs-endpoint/predict",
        files={"file": f}
    )
    
result = response.json()
print(f"Prediction: {result['class_name']} ({result['probability']:.2%})")