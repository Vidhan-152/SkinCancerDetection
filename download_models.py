import os
import requests
import re

def download_from_gdrive(file_id, dest_path):
    print(f"Downloading {dest_path}...")
    
    session = requests.Session()
    
    # Step 1 — get the confirmation token
    response = session.get(
        "https://drive.google.com/uc",
        params={"export": "download", "id": file_id},
        stream=True
    )
    
    # Step 2 — extract confirmation token from response
    confirm_token = None
    
    # Check cookies
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            confirm_token = value
            break
    
    # Check HTML content for newer Drive format
    if confirm_token is None:
        content = response.content.decode("utf-8", errors="ignore")
        # Try multiple patterns Google uses
        for pattern in [
            r'confirm=([0-9A-Za-z_\-]+)',
            r'"confirm":"([0-9A-Za-z_\-]+)"',
            r'id="downloadForm".*?confirm=([^&"]+)',
        ]:
            match = re.search(pattern, content)
            if match:
                confirm_token = match.group(1)
                break
        
        # Newer Drive uses uuid token
        uuid_match = re.search(r'uuid=([^&"]+)', content)
        if uuid_match:
            # Use the direct download with uuid
            response = session.get(
                "https://drive.usercontent.google.com/download",
                params={
                    "id": file_id,
                    "export": "download",
                    "confirm": "t",
                    "uuid": uuid_match.group(1)
                },
                stream=True
            )
        elif confirm_token:
            response = session.get(
                "https://drive.google.com/uc",
                params={
                    "export": "download",
                    "id": file_id,
                    "confirm": confirm_token
                },
                stream=True
            )
        else:
            # Try direct download endpoint (works for newer Drive)
            response = session.get(
                "https://drive.usercontent.google.com/download",
                params={"id": file_id, "export": "download", "confirm": "t"},
                stream=True
            )
    
    # Step 3 — write file
    total = 0
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
                total += len(chunk)
    
    size_mb = total / (1024 * 1024)
    print(f"✓ Downloaded {dest_path} ({size_mb:.1f} MB)")
    
    # Step 4 — verify not HTML
    if total < 1024 * 1024:
        with open(dest_path, "rb") as f:
            header = f.read(200)
        if b"<!DOCTYPE" in header or b"<html" in header:
            os.remove(dest_path)
            raise RuntimeError(
                f"Got HTML page instead of model weights.\n"
                f"File ID: {file_id}\n"
                f"Please make sure the file is shared as 'Anyone with the link'."
            )


def download_models():
    models = {
        "models/unet_trained.pth": "1oeGMxAEpaGyXXgiqJPRPSLTey38C1P75",
        "models/mobilenet_balanced_2k.pth": "1jBKkFp1XRo_XZoqTYErNEEUIyq-XPQL2",
    }
    os.makedirs("models", exist_ok=True)

    for dest_path, file_id in models.items():
        if os.path.exists(dest_path):
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            # Re-download if suspiciously small
            if size_mb < 1.0:
                print(f"⚠ {dest_path} is too small ({size_mb:.1f}MB), re-downloading...")
                os.remove(dest_path)
            else:
                print(f"✓ Already exists: {dest_path} ({size_mb:.1f} MB)")
                continue
        try:
            download_from_gdrive(file_id, dest_path)
        except Exception as e:
            print(f"✗ Failed: {e}")


if __name__ == "__main__":
    download_models()