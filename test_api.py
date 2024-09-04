import requests
import json

# API URL
url = 'http://127.0.0.1:5000/api/predict'

# Data to be sent
data = {
    "columns": ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"],
    "data": [5.1, 3.5, 1.4, 0.2]
}

# Send POST request
response = requests.post(url, json=data)

# Print response
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
