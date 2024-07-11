# Importing the requests library
import requests
import json
# Requesting the server to make predictions
resp = requests.post("http://localhost:5000/predict", json={"text": "I am going to the moon", "target_lang": "ja"})
# Print the output
print(json.loads(resp.text))