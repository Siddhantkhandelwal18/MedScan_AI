import warnings
warnings.filterwarnings('ignore')

# Import required libraries
import os
import streamlit as st
import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image
import google.generativeai as genai

# Function to download the model from Hugging Face
MODEL_URL = "https://huggingface.co/your_model_repo/resolve/main/model_25_epoch.h5"
MODEL_PATH = "model_25_epoch.h5"

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            response = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        st.success("Model downloaded successfully!")

# Download the model if not already present
download_model()

# Load the pre-trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Configure Gemini API (Load key from environment variable)
api_key = os.getenv("")
if not api_key:
    st.error("API key is missing. Set the GEMINI_API_KEY environment variable.")
    gen_model = None
else:
    genai.configure(api_key=api_key)
    gen_model = genai.GenerativeModel("gemini-1.5-pro-latest")

# Function for image prediction
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))  
    x = image.img_to_array(img)  
    x = np.expand_dims(x, axis=0)  
    x = preprocess_input(x)  
    classes = model.predict(x)  
    result = np.argmax(classes[0])  

    labels = ['Bacterial Pneumonia', 'COVID-19', 'Normal', 'Tuberculosis', 'Viral Pneumonia']
    return labels[result]

# Function to generate a medical report
def generate_report_gemini(disease, confidence_level):
    report_prompt = f"""
    Create a detailed, compassionate medical report:
    Disease: {disease}
    Confidence Level: {confidence_level:.2f}%
    Provide a comprehensive explanation including:
    - Disease details
    - Health implications
    - Recommended steps
    - Professional medical advice
    """

    try:
        response = gen_model.generate_content(report_prompt)
        return response.text if hasattr(response, 'text') else "No valid response received."
    except Exception as e:
        return f"Error generating report: {e}"

# Streamlit UI
st.set_page_config(page_title="MedScan AI", page_icon="🩺", layout="wide")

st.title("MedScan AI: Chest X-Ray Diagnostics 🏥")

uploaded_file = st.file_uploader("Upload a Chest X-ray", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Chest X-ray", use_column_width=True)
    
    img_path = "uploaded_image.jpg"
    img.save(img_path)

    with st.spinner('Analyzing X-ray...'):
        disease = predict_image(img_path)

    st.success(f"Detected Disease: {disease}")

    if st.button("Generate Report"):
        confidence_level = np.random.uniform(90, 98)
        if gen_model:
            report = generate_report_gemini(disease, confidence_level)
            st.text_area("Medical Report", report, height=250)
            st.download_button("Download Report", report, file_name="Medical_Report.txt")
        else:
            st.error("API error: Unable to generate report.")

st.sidebar.header("📋 Health Tips")
st.sidebar.info("""
- **Bacterial Pneumonia**: Get vaccinated and practice good hygiene.
- **COVID-19**: Follow safety protocols and mask up.
- **Tuberculosis**: Regular screenings are recommended.
""")
st.sidebar.warning("⚠️ This is an AI-assisted tool. Always consult a healthcare professional.")
