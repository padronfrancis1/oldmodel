import requests

# API Endpoint
API_URL = "https://old-modelz-f2chcyhucyahczc8.canadacentral-01.azurewebsites.net/predict"

# Input data
data = {
    "features": [-1, .2, .4, 0.2]  # Example feature set
}

# Send
response = requests.post(API_URL, json=data)

# Print the response    
print(response.json())