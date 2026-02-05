import streamlit as st
from inference import FoodFreshnessDetector
from PIL import Image

st.title("🍎 Food Freshness Detector")

model = FoodFreshnessDetector(
    model_path="checkpoints/final_model.pth",
    model_type="image_only"
)

uploaded_file = st.file_uploader("Upload Food Image")

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img)

    result = model.predict(img)

    st.write("Prediction:", result["class_name"])
    st.write("Confidence:", result["confidence"])
