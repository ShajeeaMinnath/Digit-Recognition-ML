import streamlit as st
from streamlit_drawable_canvas import st_canvas
import joblib
import cv2
from utils import preprocess_image # Logic from Member B

# --- 1. Load the "Brain" ---
try:
    model = joblib.load('mnist_svm_model.pkl')
    pca = joblib.load('pca_transformer.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Error: .pkl files missing! Ensure Member B merged them to main.")

st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")
st.title("🔢 Hand-Drawn Digit Classifier")
st.markdown("### Powered by HOG Features + SVM")
st.write("Draw a digit (0-9) in the center of the box below.")

# --- 2. The Drawing Interface ---
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=18, # Thicker strokes work better for MNIST
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# --- 3. The Prediction Logic ---
if canvas_result.image_data is not None:
    # Get the image data
    img = canvas_result.image_data.astype('uint8')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if st.button("Classify Digit"):
        # Preprocess using the team's shared utility
        features = preprocess_image(img_gray)
        
        # Apply the trained scaling and PCA
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)
        
        # Predict!
        prediction = model.predict(features_pca)
        
        st.header(f"The model thinks this is a: {prediction[0]}")
        st.balloons()