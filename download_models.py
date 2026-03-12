import os
import gdown

def download_models():
    os.makedirs("models", exist_ok=True)

    # UNet segmentation weights
    if not os.path.exists("models/unet_trained.pth"):
        print("Downloading UNet weights...")
        gdown.download(
            "https://drive.google.com/uc?id=1oeGMxAEpaGyXXgiqJPRPSLTey38C1P75",
            "models/unet_trained.pth",
            quiet=False
        )
        print("✓ UNet weights downloaded")

    # Classifier weights
    if not os.path.exists("models/mobilenet_balanced_2k.pth"):
        print("Downloading classifier weights...")
        gdown.download(
            "https://drive.google.com/uc?id=1jBKkFp1XRo_XZoqTYErNEEUIyq-XPQL2",
            "models/mobilenet_balanced_2k.pth",
            quiet=False
        )
        print("✓ Classifier weights downloaded")

if __name__ == "__main__":
    download_models()