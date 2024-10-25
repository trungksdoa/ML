import json
import requests

url = 'http://0.0.0.0:8000/predict'

koi_data = [
    {"age": 1, "length": 15.5, "water_quality": "good", "diet": "high protein"},
    {"age": 2, "length": 25.0, "water_quality": "excellent", "diet": "balanced"},
    {"age": 3, "length": 35.5, "water_quality": "fair", "diet": "low protein"}
]

predictions = []
for data in koi_data:
    response = requests.post(url, json=data)
    predictions.append(response.json())

print(json.dumps(predictions, indent=2))