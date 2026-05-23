import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO

# 1. Page Config
st.set_page_config(
    page_title="Strawberry Detection Dashboard",
    page_icon="🍓",
    layout="centered"
)

# 2. Cache Model Loading (Speeds up app performance)
@st.cache_resource
def load_yolo_model():
    # After training in Colab, download 'best.pt' and place it in the same directory as app.py
    model_path = "best.pt" 
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Could not load model from '{model_path}'. Ensure you downloaded 'best.pt' from your Colab training run. Error: {e}")
        return None

model = load_yolo_model()

# 3. User Interface Header
st.title("🍓 Strawberry Object Detection")
st.write("Upload an image to detect strawberries using your custom fine-tuned YOLOv8 model.")

# 4. Image Uploader Component
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Open image using PIL
    image = Image.open(uploaded_file)
    
    # Setup columns for side-by-side view
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
    # Process image for YOLOv8 inference
    # Convert PIL Image to OpenCV BGR format
    img_array = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # 5. Run Model Inference
    # conf=0.25 means it filters out detections with less than 25% confidence
    results = model.predict(source=img_bgr, conf=0.25)
    
    # 6. Render Bounding Boxes on Image
    # results[0].plot() returns the image array with boxes, labels, and scores drawn
    annotated_frame = results[0].plot()
    
    # Convert back to RGB for Streamlit rendering
    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    with col2:
        st.subheader("Object Detection")
        st.image(annotated_rgb, use_container_width=True)
        
    # 7. Print Detection Stats
    boxes = results[0].boxes
    st.success(f"🔍 Detected **{len(boxes)}** strawberry object(s) in this image!")