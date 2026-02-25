import streamlit as st
from PIL import Image
import time
import os
import sys

# Add the src directory to the Python path so it can find the models package
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# Import the extracted models
from models.clip_model import CLIPPredictor
from models.dino_model import DINOPredictor
from models.vit_model import ViTPredictor
from models.vlm_model import VLMPredictor

# --- Load Models (Cached to avoid reloading on every interaction) ---
@st.cache_resource
def load_clip_model():
    return CLIPPredictor()

@st.cache_resource
def load_dino_model():
    return DINOPredictor()

@st.cache_resource      
def load_vit_model():
    # Assuming the weights file might be placed in a 'weights' folder later
    weights_path = os.path.join(os.path.dirname(__file__), "weights", "vit_multilabel_best.pt")
    return ViTPredictor(model_path=weights_path)

@st.cache_resource
def load_vlm_model():
    return VLMPredictor()

def predict_clip(image):
    model = load_clip_model()
    return model.predict(image)

def predict_dino(image):
    model = load_dino_model()
    return model.predict(image)

def predict_vit(image):
    model = load_vit_model()
    return model.predict(image)

def predict_vlm(image):
    model = load_vlm_model()
    return model.predict(image)

# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="Vehicle Classification Solutions", page_icon="🚗")
    
    st.title(" Few-Shot Vehicle Classification ")
    st.write("Compare different solutions (CLIP, DINO, VLM) for vehicle classification.")

    # 1. Select Solution
    st.header("1. Choose a Solution")
    model_choice = st.selectbox(
        "Select the model you want to use for prediction:",
        ("CLIP", "DINO", "VLM")
    )

    # 2. Upload Image
    st.header("2. Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        
        # Create columns to make the layout look nicer
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption='Uploaded Image', use_container_width=True)

        with col2:
            # 3. Predict Button
            st.header("3. Predict")
            if st.button("Predict Class", type="primary"):
                with st.spinner(f'Running prediction using {model_choice}...'):
                    
                    # Route to the correct model based on selection
                    if model_choice == "CLIP":
                        prediction = predict_clip(image)
                    elif model_choice == "DINO":
                        prediction = predict_dino(image)
                    elif model_choice == "VLM":
                        prediction = predict_vlm(image)
                    else:
                        prediction = "Unknown Model"

                # 4. Show Result
                st.success("Prediction Complete!")
                st.subheader("Classification Result:")
                st.info(f"**{prediction}**")

if __name__ == "__main__":
    main()
