import streamlit as st
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

# Path to your trained YOLOv8 model
MODEL_PATH = 'best.pt'

# Class names as per your YAML
CLASS_NAMES = ['Amount_In_Numbers', 'Amount_In_Words', 'Date', 'MICR', 'Payee_Name', 'Sign']

def run_cheque_ocr(image):
    model = YOLO(MODEL_PATH)
    results = model(image)
    detections = results[0].boxes.data.cpu().numpy()
    cheque_detected = False

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf < 0.5:
            continue
        cls = int(cls)
        label = CLASS_NAMES[cls]
        if label not in ['Amount_In_Numbers', 'Amount_In_Words', 'Date', 'MICR', 'Payee_Name']:
            continue
        cheque_detected = True
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return cheque_detected, image

st.title("Cheque Detection App")

uploaded_file = st.file_uploader("Upload a cheque image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    if image is None:
        st.error("Could not read the image. Please upload a valid image file.")
    else:
        cheque_detected, output_image = run_cheque_ocr(image)
        st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), caption="Detected Cheque", use_container_width=True)
        if cheque_detected:
            st.success("The uploaded document is a cheque.")
        else:
            st.warning("No cheque fields detected. The uploaded document may not be a cheque.")