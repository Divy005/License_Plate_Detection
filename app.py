import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
from PIL import Image
import numpy as np

# Title
st.title("🚗 License Plate Detection App")

# Upload YOLOv8 model path
model_path = (r"C:\Users\Divy\OneDrive\Desktop\Projects and Textbooks\Deep-Learning\Projects\license_plate_detection\best.pt")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file and model_path:
    # Load YOLOv8 model
    model = YOLO(model_path)

    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Run detection
    results = model(img)

    # Plot results
    for r in results:
        annotated = r.plot()  # returns BGR numpy image with bounding boxes

    # Convert BGR → RGB
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    st.image(annotated_rgb, caption="Detected License Plates", use_column_width=True)
