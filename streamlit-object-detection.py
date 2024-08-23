import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_objects(image):
    results = model(image)
    return results.render()[0]

st.title('Real-time Object Detection')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Detecting...")
    result_img = detect_objects(image)
    st.image(result_img, caption='Detected Objects.', use_column_width=True)

st.write("Upload an image to detect objects!")
