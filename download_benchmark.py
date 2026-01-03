
import requests
import os

url = "https://huggingface.co/keras-io/conv-lstm/resolve/main/model.keras"
output_path = "results/benchmark_model.weights.h5"

print(f"Downloading benchmark model from {url}...")
response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded to {output_path}")
else:
    print(f"Failed to download. Status: {response.status_code}")
