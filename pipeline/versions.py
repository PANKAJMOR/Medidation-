"""
Simple version checker - just prints what you have installed
"""

packages = [
    # Your direct imports
    "ultralytics",      # from ultralytics import YOLO
    "opencv-python",    # import cv2
    "numpy",            # import numpy as np
    "scipy",            # from scipy.io import wavfile
    "reportlab",        # from reportlab.lib.pagesizes import A4
    "yt-dlp",           # import yt_dlp
    "python-dotenv",    # from dotenv import load_dotenv
    "mediapipe",        # import mediapipe as mp
    
    # Required dependencies
    "torch",            # required by ultralytics
    "torchvision",      # required by ultralytics
]

print("Checking installed versions...\n")

for package in packages:
    try:
        if package == "opencv-python":
            import cv2
            version = cv2.__version__
        elif package == "python-dotenv":
            from dotenv import __version__ as dotenv_version
            version = dotenv_version
        elif package == "yt-dlp":
            import yt_dlp
            version = yt_dlp.version.__version__
        else:
            module = __import__(package)
            version = module.__version__
        
        print(f"{package}=={version}")
    except Exception as e:
        print(f"{package} - NOT INSTALLED (or import error)")

print("\nDone!")