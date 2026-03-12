import os
import requests

def download_file(url, dest_path):
    print(f"Downloading to {dest_path}...")
    session = requests.Session()
    
    # Handle Google Drive large file confirmation
    response = session.get(url, stream=True)
    
    # Check for virus scan warning page
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            url = url + f"&confirm={value}"
            response = session.get(url, stream=True)
            break
    
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    print(f"✓ Downloaded {dest_path}")

def download_models():
    os.makedirs("models", exist_ok=True)

    if not os.path.exists("models/unet_trained.pth"):
        download_file(
            "https://drive.google.com/uc?export=download&id=1oeGMxAEpaGyXXgiqJPRPSLTey38C1P75",
            "models/unet_trained.pth"
        )

    if not os.path.exists("models/mobilenet_balanced_2k.pth"):
        download_file(
            "https://drive.google.com/uc?export=download&id=1jBKkFp1XRo_XZoqTYErNEEUIyq-XPQL2",
            "models/mobilenet_balanced_2k.pth"
        )

if __name__ == "__main__":
    download_models()
