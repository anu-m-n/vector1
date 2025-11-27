
import requests, json, os, time

# This script assumes the backend is running on port 8000
BASE = "http://localhost:8000"
DATA_FILE = os.path.join(os.path.dirname(__file__), "sample_data.json")

with open(DATA_FILE,'r') as f:
    data = json.load(f)

for n in data['nodes']:
    r = requests.post(BASE + "/nodes", json=n)
    print("Added node:", r.json())
    time.sleep(0.05)

for e in data['edges']:
    r = requests.post(BASE + "/edges", json=e)
    print("Added edge:", r.json())
    time.sleep(0.02)

print("Done. You can now try the frontend at frontend/index.html")
