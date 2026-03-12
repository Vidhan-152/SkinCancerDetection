import os
import requests
import re

def download_from_gdrive(file_id, dest_path):
    """Download a file from Google Drive, handling large file confirmations."""
    print(f"Downloading {dest_path}...")

    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    # First request
    response = session.get(URL, params={"id": file_id}, stream=True)

    # Look for confirmation token in cookies
    confirm_token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            confirm_token = value
            break

    # Check response content for newer Drive confirmation
    if confirm_token is None:
        content_start = b""
        for chunk in response.iter_content(chunk_size=4096):
            content_start += chunk
            if len(content_start) > 4096:
                break
        match = re.search(rb'confirm=([0-9A-Za-z_\-]+)', content_start)
        if match:
            confirm_token = match.group(1).decode()
        # Reset response
        response = session.get(URL, params={"id": file_id}, stream=True)

    # Second request with confirmation if needed
    if confirm_token:
        response = session.get(
            URL,
            params={"id": file_id, "confirm": confirm_token},
            stream=True
        )

    # Write to file
    os.makedirs(os.path.dirname(dest_path) if os.path.dirname(dest_path) else ".", exist_ok=True)
    total = 0
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
                total += len(chunk)

    size_mb = total / (1024 * 1024)
    print(f"✓ Downloaded {dest_path} ({size_mb:.1f} MB)")

    # Verify it's a real model file not an HTML error page
    if total < 1024 * 1024:
        with open(dest_path, 'rb') as f:
            header = f.read(200)
        if b'<!DOCTYPE' in header or b'<html' in header:
            os.remove(dest_path)
            raise RuntimeError(
                f"Got HTML page instead of model weights. "
                f"Please make sure file ID {file_id} is shared as 'Anyone with the link'."
            )


def download_models():
    """Download model weights from Google Drive if not already present."""
    models = {
        "models/unet_trained.pth": "1oeGMxAEpaGyXXgiqJPRPSLTey38C1P75",
        "models/mobilenet_balanced_2k.pth": "1jBKkFp1XRo_XZoqTYErNEEUIyq-XPQL2",
    }

    os.makedirs("models", exist_ok=True)

    for dest_path, file_id in models.items():
        if os.path.exists(dest_path):
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"✓ Already exists: {dest_path} ({size_mb:.1f} MB)")
            continue
        try:
            download_from_gdrive(file_id, dest_path)
        except Exception as e:
            print(f"✗ Failed to download {dest_path}: {e}")


if __name__ == "__main__":
    download_models()